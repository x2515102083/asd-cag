from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cag.losses import classification_metrics


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = batch.to(device)
        outputs = model(batch)
        probs = torch.softmax(outputs["logits_c"], dim=-1)[:, 1].detach().cpu().numpy()
        labels = batch.y.view(-1).detach().cpu().numpy().astype(int)
        subject_indices = batch.subject_uid.view(-1).detach().cpu().numpy().astype(int)
        subject_ids = list(getattr(batch, "subject_id", [str(idx) for idx in subject_indices]))
        sites = batch.site.view(-1).detach().cpu().numpy().astype(int)
        for sid, index, label, prob, site in zip(subject_ids, subject_indices, labels, probs, sites):
            rows.append(
                {
                    "subject_id": str(sid),
                    "subject_index": int(index),
                    "label": int(label),
                    "probability": float(prob),
                    "site_idx": int(site),
                }
            )
    return pd.DataFrame(rows)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    predictions = predict(model, loader, device)
    if predictions.empty:
        return {"ACC": 0.0, "AUC": float("nan"), "SPE": 0.0, "SEN": 0.0}, predictions
    metrics = classification_metrics(
        labels=predictions["label"].to_numpy(),
        probabilities=predictions["probability"].to_numpy(),
        threshold=threshold,
    )
    return metrics, predictions


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    return checkpoint if isinstance(checkpoint, dict) else {}
