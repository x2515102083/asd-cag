from __future__ import annotations

import json
import math
import time as time_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from cag.data import BrainFCDataset, SubjectRecord, infer_n_nodes, make_dataset
from cag.evaluate import evaluate_loader
from cag.graph import batch_subject_ids
from cag.losses import FocalLoss
from cag.models import CAGModel
from cag.split import external_site_split, stratified_cv_indices
from cag.utils import save_json, select_device, set_seed


def make_model(n_nodes: int, config: dict[str, Any]) -> CAGModel:
    return CAGModel(
        n_nodes=n_nodes,
        hidden_dim=int(config.get("hidden_dim", 128)),
        n_layers=int(config.get("n_layers", 2)),
        dropout=float(config.get("dropout", 0.2)),
        n_env=int(config.get("n_env", 2)),
        lambda_e1=float(config.get("lambda_e1", 1.0)),
        lambda_e2=float(config.get("lambda_e2", 1.0)),
        lambda_s=float(config.get("lambda_s", 10.0)),
        mask_type=str(config.get("mask_type", "node")),
        temperature=float(config.get("temperature", 1.0)),
    )


@torch.no_grad()
def build_pseudo_env_labels(
    model: CAGModel,
    train_loader: DataLoader,
    n_env: int = 2,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[dict[str, int], int, int, float]:
    model.eval()
    device = device or next(model.parameters()).device
    embeddings: list[np.ndarray] = []
    subject_ids: list[str] = []
    for batch in tqdm(train_loader, desc="pseudo-env", leave=False):
        batch = batch.to(device)
        outputs = model(batch)
        embeddings.append(outputs["emb_gc"].detach().cpu().numpy())
        subject_ids.extend(batch_subject_ids(batch))
    if not embeddings:
        return {}, 0, 0, 0.0
    matrix = np.concatenate(embeddings, axis=0)
    clusters = min(int(n_env), matrix.shape[0])
    if clusters <= 1:
        labels = np.zeros(matrix.shape[0], dtype=int)
        silhouette = 0.0
    else:
        labels = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(matrix)
        if len(set(labels)) > 1:
            silhouette = silhouette_score(matrix, labels)
        else:
            silhouette = 0.0
    cluster_0_count = int(np.sum(labels == 0))
    cluster_1_count = int(np.sum(labels == 1))
    return {str(subject_id): int(label) for subject_id, label in zip(subject_ids, labels)}, cluster_0_count, cluster_1_count, float(silhouette)


def train_one_epoch(
    model: CAGModel,
    train_loader: DataLoader,
    optimizer: AdamW,
    pseudo_env: dict[str, int],
    device: torch.device,
    use_le1: bool = True,
    use_le2: bool = True,
    use_ls: bool = True,
) -> dict[str, float]:
    model.train()
    focal = FocalLoss()
    totals: dict[str, float] = {"loss": 0.0, "Lc": 0.0, "Le1": 0.0, "Le2": 0.0, "Ls": 0.0}
    mask_totals: list[float] = []
    emb_totals: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_scores: list[float] = []
    steps = 0
    for batch in tqdm(train_loader, desc="train", leave=False):
        batch = batch.to(device)
        y = batch.y.view(-1).long()
        subject_ids = batch_subject_ids(batch)
        eps = torch.tensor([pseudo_env[str(subject_id)] for subject_id in subject_ids], dtype=torch.long, device=device)

        outputs = model(batch)
        loss_c = focal(outputs["logits_c"], y)
        loss_e1 = F.cross_entropy(outputs["logits_env1"], eps)
        loss_e2 = F.cross_entropy(outputs["logits_env2"], eps)
        loss_s = F.binary_cross_entropy_with_logits(outputs["logits_s"].view(-1), y.float())
        loss = loss_c
        if use_le1:
            loss = loss + loss_e1
        if use_le2:
            loss = loss + loss_e2
        if use_ls:
            loss = loss + loss_s

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["Lc"] += float(loss_c.detach().cpu())
        totals["Le1"] += float(loss_e1.detach().cpu())
        totals["Le2"] += float(loss_e2.detach().cpu())
        totals["Ls"] += float(loss_s.detach().cpu())

        mask_totals.append(outputs["mask"].detach().cpu().mean().item())
        emb_totals.append(outputs["emb_gc"].detach().cpu().std().item())

        probs = torch.softmax(outputs["logits_c"], dim=1)
        scores = probs[:, 1].detach().cpu().numpy()
        preds = torch.argmax(outputs["logits_c"], dim=1).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_scores.extend(scores)

        steps += 1

    result = {key: value / max(steps, 1) for key, value in totals.items()}
    result["mask_mean"] = float(np.mean(mask_totals))
    result["mask_std"] = float(np.std(mask_totals))
    result["emb_gc_std"] = float(np.mean(emb_totals))

    if all_labels:
        result["train_ACC"] = float(np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels))
        if len(set(all_labels)) > 1:
            from sklearn.metrics import roc_auc_score
            result["train_AUC"] = float(roc_auc_score(all_labels, all_scores))
        else:
            result["train_AUC"] = float("nan")
    else:
        result["train_ACC"] = float("nan")
        result["train_AUC"] = float("nan")

    return result


def _save_checkpoint(path: Path, model: CAGModel, optimizer: AdamW, epoch: int, val_metrics: dict[str, float], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "config": config,
        },
        path,
    )


def train_fold(
    records: list[SubjectRecord],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int] | None,
    config: dict[str, Any],
    out_dir: str | Path,
    checkpoint_dir: str | Path,
    test_site: str = "Unknown",
    fold_idx: int = 1,
) -> dict[str, Any]:
    seed = int(config["training"].get("seed", 42))
    set_seed(seed)
    n_nodes = infer_n_nodes(records)
    batch_size = int(config["training"].get("batch_size", 32))
    num_workers = int(config["training"].get("num_workers", 0))
    device = select_device(str(config["training"].get("device", "auto")))

    use_le1 = bool(config["model"].get("use_le1", True))
    use_le2 = bool(config["model"].get("use_le2", True))
    use_ls = bool(config["model"].get("use_ls", True))
    lambda_e1 = float(config["model"].get("lambda_e1", 1.0))
    lambda_e2 = float(config["model"].get("lambda_e2", 1.0))
    lambda_s = float(config["model"].get("lambda_s", 10.0))
    n_splits = int(config["training"].get("n_splits", 10))

    print(f"\nFold {fold_idx}: test_site={test_site}, use_le1={use_le1}, use_le2={use_le2}, use_ls={use_ls}, lambda_e1={lambda_e1}, lambda_e2={lambda_e2}, lambda_s={lambda_s}, device={device}")

    train_dataset = make_dataset(records, train_indices)
    site_to_idx = train_dataset.site_to_idx
    val_dataset = BrainFCDataset(records, site_to_idx=site_to_idx, indices=val_indices)
    test_dataset = BrainFCDataset(records, site_to_idx=site_to_idx, indices=test_indices or [])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    pseudo_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_indices else None

    model = make_model(n_nodes=n_nodes, config=config["model"]).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 0.0001)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    fold_out = Path(out_dir)
    fold_ckpt = Path(checkpoint_dir)
    fold_out.mkdir(parents=True, exist_ok=True)
    fold_ckpt.mkdir(parents=True, exist_ok=True)
    best_auc = float("-inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    n_epochs = int(config["training"].get("epochs", 50))

    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time_module.time()
        pseudo_env, cluster_0_count, cluster_1_count, silhouette = build_pseudo_env_labels(
            model=model,
            train_loader=pseudo_loader,
            n_env=int(config["model"].get("n_env", 2)),
            seed=seed + epoch,
            device=device,
        )
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            pseudo_env=pseudo_env,
            device=device,
            use_le1=use_le1,
            use_le2=use_le2,
            use_ls=use_ls,
        )
        val_metrics, val_predictions = evaluate_loader(
            model=model,
            loader=val_loader,
            device=device,
            threshold=float(config.get("evaluation", {}).get("threshold", 0.5)),
        )
        epoch_time = time_module.time() - epoch_start_time
        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "Lc": train_metrics["Lc"],
            "Le1": train_metrics["Le1"],
            "Le2": train_metrics["Le2"],
            "Ls": train_metrics["Ls"],
            "train_ACC": train_metrics["train_ACC"],
            "train_AUC": train_metrics["train_AUC"],
            "val_ACC": val_metrics["ACC"],
            "val_AUC": val_metrics["AUC"],
            "pseudo_env_cluster_0_count": cluster_0_count,
            "pseudo_env_cluster_1_count": cluster_1_count,
            "pseudo_env_silhouette": silhouette,
            "mask_mean": train_metrics["mask_mean"],
            "mask_std": train_metrics["mask_std"],
            "emb_gc_std": train_metrics["emb_gc_std"],
            "epoch_time": epoch_time,
        }
        history.append(row)

        print(f"Fold {fold_idx}/{n_splits} Epoch {epoch}/{n_epochs} loss={train_metrics['loss']:.4f} Lc={train_metrics['Lc']:.4f} Le1={train_metrics['Le1']:.4f} Le2={train_metrics['Le2']:.4f} Ls={train_metrics['Ls']:.4f} val_AUC={val_metrics['AUC']:.4f} cluster=[{cluster_0_count},{cluster_1_count}] mask_mean={train_metrics['mask_mean']:.4f} emb_gc_std={train_metrics['emb_gc_std']:.4f}", flush=True)

        current_auc = val_metrics["AUC"]
        if not math.isnan(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            best_metrics = val_metrics
            _save_checkpoint(fold_ckpt / "best.pt", model, optimizer, epoch, val_metrics, config)
            val_predictions.to_csv(fold_out / "val_predictions.csv", index=False)
        _save_checkpoint(fold_ckpt / "last.pt", model, optimizer, epoch, val_metrics, config)

    pd.DataFrame(history).to_csv(fold_out / "history.csv", index=False)
    save_json(best_metrics, fold_out / "metrics_val.json")

    test_metrics: dict[str, float] | None = None
    if test_loader is not None:
        checkpoint = torch.load(fold_ckpt / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics, test_predictions = evaluate_loader(
            model=model,
            loader=test_loader,
            device=device,
            threshold=float(config.get("evaluation", {}).get("threshold", 0.5)),
        )
        test_predictions.to_csv(fold_out / "test_predictions.csv", index=False)
        save_json(test_metrics, fold_out / "metrics_test.json")

    return {
        "best_checkpoint": str(fold_ckpt / "best.pt"),
        "val": best_metrics,
        "test": test_metrics,
    }


def _summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        for metric in ("ACC", "AUC", "SPE", "SEN"):
            values = [
                row[f"{split}_{metric}"]
                for row in rows
                if row.get(f"{split}_{metric}") is not None
                and not pd.isna(row[f"{split}_{metric}"])
            ]
            if values:
                summary[f"{split}_{metric}_mean"] = float(np.mean(values))
                summary[f"{split}_{metric}_std"] = float(np.std(values, ddof=0))
    return summary


def train_external_site_cv(
    records: list[SubjectRecord],
    test_site: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    base_train_indices, test_indices = external_site_split(records, test_site=test_site)
    folds = stratified_cv_indices(
        records=records,
        candidate_indices=base_train_indices,
        n_splits=int(config["training"].get("n_splits", 10)),
        seed=int(config["training"].get("seed", 42)),
    )
    output_root = Path(config["training"].get("output_dir", "outputs/abide1")) / str(test_site)
    checkpoint_root = Path(config["training"].get("checkpoint_dir", "checkpoints/abide1")) / str(test_site)
    rows: list[dict[str, Any]] = []
    for fold_idx, (train_indices, val_indices) in enumerate(folds, start=1):
        result = train_fold(
            records=records,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            config=config,
            out_dir=output_root / f"fold_{fold_idx:02d}",
            checkpoint_dir=checkpoint_root / f"fold_{fold_idx:02d}",
            test_site=test_site,
            fold_idx=fold_idx,
        )
        row = {"fold": fold_idx, "checkpoint": result["best_checkpoint"]}
        for split in ("val", "test"):
            metrics = result.get(split) or {}
            for metric in ("ACC", "AUC", "SPE", "SEN"):
                row[f"{split}_{metric}"] = metrics.get(metric)
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / "fold_metrics.csv", index=False)
    summary = _summarize_metrics(rows)
    save_json(summary, output_root / "summary.json")
    return {"folds": rows, "summary": summary}


def dump_metrics_csv_json(metrics: dict[str, Any], csv_path: str | Path, json_path: str | Path) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    with Path(json_path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
