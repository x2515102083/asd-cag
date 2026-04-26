from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("torch_geometric")
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cag.data import BrainFCDataset, read_subject_records
from cag.losses import classification_metrics
from cag.models import CAGModel
from cag.train import build_pseudo_env_labels, train_one_epoch


def _write_synthetic_dataset(root: Path, n_subjects: int = 8, n_nodes: int = 8) -> tuple[Path, Path]:
    fc_dir = root / "fc"
    fc_dir.mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(7)
    for idx in range(n_subjects):
        subject_id = f"sub_{idx:04d}"
        label = idx % 2
        site = f"SITE_{idx % 2}"
        matrix = rng.normal(0.0, 0.2, size=(n_nodes, n_nodes)).astype(np.float32)
        matrix = 0.5 * (matrix + matrix.T)
        matrix += label * 0.1
        np.fill_diagonal(matrix, 1.0)
        np.save(fc_dir / f"{subject_id}.npy", matrix)
        rows.append({"subject_id": subject_id, "label": label, "site": site})
    subjects_csv = root / "subjects.csv"
    pd.DataFrame(rows).to_csv(subjects_csv, index=False)
    return subjects_csv, fc_dir


def test_dataset_forward_training_step_and_metrics(tmp_path: Path) -> None:
    subjects_csv, fc_dir = _write_synthetic_dataset(tmp_path)
    records = read_subject_records(subjects_csv, fc_dir)
    dataset = BrainFCDataset(records)
    assert len(dataset) == 8

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    model = CAGModel(n_nodes=8, hidden_dim=16, n_layers=2, dropout=0.1, n_env=2)
    outputs = model(batch)
    assert outputs["logits_c"].shape == (4, 2)
    assert outputs["logits_s"].shape == (4, 1)
    assert outputs["logits_env1"].shape == (4, 2)
    assert outputs["logits_env2"].shape == (4, 2)
    assert outputs["mask"].shape == (4 * 8,)

    optimizer = AdamW(model.parameters(), lr=1e-3)
    pseudo_env, _, _, _ = build_pseudo_env_labels(model, loader, n_env=2, seed=7, device=torch.device("cpu"))
    train_metrics = train_one_epoch(model, loader, optimizer, pseudo_env, device=torch.device("cpu"))
    assert train_metrics["loss"] > 0.0

    metrics = classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    assert metrics["ACC"] == 1.0
    assert metrics["SEN"] == 1.0
    assert metrics["SPE"] == 1.0
