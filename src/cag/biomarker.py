from __future__ import annotations

from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_top_nodes(
    model: torch.nn.Module,
    loader: DataLoader,
    top_percent: float = 0.03,
    asd_label: int = 1,
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    model.eval()
    device = device or next(model.parameters()).device
    probability_sum: np.ndarray | None = None
    asd_count = 0

    for batch in tqdm(loader, desc="biomarkers", leave=False):
        batch = batch.to(device)
        outputs = model(batch)
        mask = outputs["mask"].detach().cpu().numpy()
        labels = batch.y.view(-1).detach().cpu().numpy().astype(int)
        batch_size = int(labels.shape[0])
        if batch_size == 0:
            continue
        n_nodes = int(mask.shape[0] // batch_size)
        mask = mask.reshape(batch_size, n_nodes)
        selected = labels == int(asd_label)
        if not selected.any():
            continue
        selected_mask = mask[selected]
        if probability_sum is None:
            probability_sum = np.zeros(n_nodes, dtype=np.float64)
        probability_sum += selected_mask.sum(axis=0)
        asd_count += int(selected_mask.shape[0])

    if probability_sum is None:
        raise ValueError("No ASD subjects were found for biomarker extraction.")

    probability_mean = probability_sum / max(asd_count, 1)
    top_k = max(1, ceil(len(probability_sum) * float(top_percent)))
    order = np.argsort(-probability_sum)
    selected = np.zeros(len(probability_sum), dtype=bool)
    selected[order[:top_k]] = True
    dataframe = pd.DataFrame(
        {
            "node_index": np.arange(len(probability_sum), dtype=int),
            "probability_sum": probability_sum,
            "probability_mean": probability_mean,
            "selected_top3_percent": selected,
        }
    )
    return dataframe, probability_sum


def save_biomarker_outputs(dataframe: pd.DataFrame, probability_vector: np.ndarray, out_dir: str | Path) -> None:
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output / "causal_nodes.csv", index=False)
    np.save(output / "mask_probabilities.npy", probability_vector)
