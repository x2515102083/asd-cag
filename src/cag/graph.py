from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch


@lru_cache(maxsize=16)
def build_full_edge_index(n_nodes: int) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.arange(n_nodes, dtype=torch.long),
        torch.arange(n_nodes, dtype=torch.long),
        indexing="ij",
    )
    return torch.stack([rows.reshape(-1), cols.reshape(-1)], dim=0)


def build_full_adjacency(n_nodes: int) -> torch.Tensor:
    return torch.ones((n_nodes, n_nodes), dtype=torch.float32)


def fc_to_pyg_data(
    fc: np.ndarray,
    label: int,
    site_idx: int,
    subject_id: str,
    record_index: int | None = None,
) -> Data:
    from torch_geometric.data import Data

    matrix = np.asarray(fc, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"FC matrix must be square, got shape={matrix.shape}")
    n_nodes = int(matrix.shape[0])
    edge_index = build_full_edge_index(n_nodes)
    data = Data(
        x=torch.as_tensor(matrix, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.ones(edge_index.shape[1], dtype=torch.float32),
        y=torch.tensor([int(label)], dtype=torch.long),
        site=torch.tensor([int(site_idx)], dtype=torch.long),
        subject_uid=torch.tensor([-1 if record_index is None else int(record_index)], dtype=torch.long),
    )
    data.subject_id = str(subject_id)
    return data


def batch_subject_ids(batch: Any) -> list[str]:
    value = getattr(batch, "subject_id", [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]
