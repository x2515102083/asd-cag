from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

from cag.losses import gradient_reverse


class NodeFeatureFilter(nn.Module):
    def __init__(self, n_nodes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_nodes, 2 * n_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_nodes, 4 * n_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_nodes, 2 * n_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_nodes, n_nodes),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GumbelSigmoidMask(nn.Module):
    def __init__(self, n_nodes: int, hidden_dim: int, temperature: float = 1.0, mask_type: str = "node") -> None:
        super().__init__()
        if mask_type not in {"node", "edge", "node_edge"}:
            raise ValueError("mask_type must be one of: node, edge, node_edge")
        self.temperature = float(temperature)
        self.mask_type = mask_type
        self.node_mlp = nn.Sequential(
            nn.Linear(n_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_nodes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _gumbel_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return torch.sigmoid(logits)
        eps = 1e-8
        uniform = torch.rand_like(logits).clamp(eps, 1.0 - eps)
        noise = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((logits + noise) / self.temperature)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor | None]:
        node_logits = self.node_mlp(x).squeeze(-1)
        node_mask = self._gumbel_sigmoid(node_logits)
        edge_mask = None
        if self.mask_type in {"edge", "node_edge"}:
            src, dst = edge_index
            edge_logits = self.edge_mlp(torch.cat([x[src], x[dst]], dim=-1)).squeeze(-1)
            edge_mask = self._gumbel_sigmoid(edge_logits)
        if self.mask_type == "edge":
            node_mask = torch.ones_like(node_mask)
        return {"node_mask": node_mask, "edge_mask": edge_mask}


def _gin_mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        current_dim = in_dim
        for _ in range(n_layers):
            self.convs.append(GINConv(_gin_mlp(current_dim, hidden_dim)))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return global_mean_pool(x, batch)


class CAGModel(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_env: int = 2,
        lambda_e1: float = 1.0,
        lambda_e2: float = 1.0,
        lambda_s: float = 10.0,
        mask_type: str = "node",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.lambda_e1 = float(lambda_e1)
        self.lambda_e2 = float(lambda_e2)
        self.lambda_s = float(lambda_s)
        self.mask_type = mask_type

        self.filter = NodeFeatureFilter(n_nodes=n_nodes, dropout=dropout)
        self.masker = GumbelSigmoidMask(
            n_nodes=n_nodes,
            hidden_dim=hidden_dim,
            temperature=temperature,
            mask_type=mask_type,
        )
        self.clf_encoder = GINEncoder(n_nodes, hidden_dim, n_layers=n_layers, dropout=dropout)
        self.env1_encoder = GINEncoder(n_nodes, hidden_dim, n_layers=n_layers, dropout=dropout)
        self.env2_encoder = GINEncoder(n_nodes, hidden_dim, n_layers=n_layers, dropout=dropout)
        self.s_encoder = GINEncoder(n_nodes, hidden_dim, n_layers=n_layers, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.env1_head = nn.Linear(hidden_dim, n_env)
        self.env2_head = nn.Linear(hidden_dim, n_env)
        self.spurious_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch) -> dict[str, torch.Tensor | None]:
        filtered_x = self.filter(batch.x)
        mask_out = self.masker(filtered_x, batch.edge_index)
        node_mask = mask_out["node_mask"]
        edge_mask = mask_out["edge_mask"]

        effective_node_mask = node_mask
        if edge_mask is not None and self.mask_type in {"edge", "node_edge"}:
            src, _ = batch.edge_index
            support = torch.zeros(filtered_x.shape[0], device=filtered_x.device, dtype=filtered_x.dtype)
            support.scatter_add_(0, src, edge_mask.to(filtered_x.dtype))
            degree = torch.zeros_like(support)
            degree.scatter_add_(0, src, torch.ones_like(edge_mask, dtype=filtered_x.dtype))
            support = support / degree.clamp_min(1.0)
            effective_node_mask = node_mask * support

        gc_x = filtered_x * effective_node_mask.unsqueeze(-1)
        gs_x = filtered_x * (1.0 - effective_node_mask).unsqueeze(-1)

        emb_gc = self.clf_encoder(gc_x, batch.edge_index, batch.batch)
        emb_g_prime = self.env1_encoder(filtered_x, batch.edge_index, batch.batch)
        emb_gc_env = self.env2_encoder(gc_x, batch.edge_index, batch.batch)
        emb_gs = self.s_encoder(gs_x, batch.edge_index, batch.batch)

        return {
            "logits_c": self.classifier(emb_gc),
            "logits_s": self.spurious_head(gradient_reverse(emb_gs, self.lambda_s)),
            "logits_env1": self.env1_head(gradient_reverse(emb_g_prime, self.lambda_e1)),
            "logits_env2": self.env2_head(gradient_reverse(emb_gc_env, self.lambda_e2)),
            "emb_g_prime": emb_g_prime,
            "emb_gc": emb_gc,
            "emb_gs": emb_gs,
            "mask": effective_node_mask,
            "edge_mask": edge_mask,
            "filtered_x": filtered_x,
        }
