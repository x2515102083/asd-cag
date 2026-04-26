#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.loader import DataLoader

from cag.data import read_subject_records, make_dataset, infer_n_nodes
from cag.models import CAGModel
from cag.losses import FocalLoss


def stratified_sample(records, n_subjects, seed=42):
    """Stratified sample to ensure ASD/TDC balance."""
    np.random.seed(seed)
    asd_records = [r for r in records if r.label == 1]
    tdc_records = [r for r in records if r.label == 0]
    n_per_class = n_subjects // 2
    sampled_asd = np.random.choice(asd_records, min(n_per_class, len(asd_records)), replace=False)
    sampled_tdc = np.random.choice(tdc_records, min(n_per_class, len(tdc_records)), replace=False)
    return list(sampled_asd) + list(sampled_tdc)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects_csv", type=str, required=True)
    parser.add_argument("--fc_dir", type=str, required=True)
    parser.add_argument("--n_subjects", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_adversarial", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs/overfit_tiny")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Print device information
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Load data
    records = read_subject_records(args.subjects_csv, args.fc_dir)
    sampled_records = stratified_sample(records, args.n_subjects, args.seed)
    dataset = make_dataset(sampled_records)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Infer n_nodes
    n_nodes = infer_n_nodes(sampled_records)

    # Create model
    model = CAGModel(
        n_nodes=n_nodes,
        hidden_dim=64,
        n_layers=2,
        dropout=0.2,
        n_env=2,
        lambda_e1=50.0,
        lambda_e2=20.0,
        lambda_s=5.0,
        mask_type="node",
        temperature=1.0,
    )

    # Move model to device
    model = model.to(args.device)

    # Count parameters
    total_params = count_parameters(model)
    filter_params = count_parameters(model.filter)
    masker_params = count_parameters(model.masker)
    encoder_params = count_parameters(model.clf_encoder)
    classifier_params = count_parameters(model.classifier)

    print(f"Total trainable parameters: {total_params}")
    print(f"Graph filter params: {filter_params}")
    print(f"Subgraph extractor params: {masker_params}")
    print(f"Classifier encoder params: {encoder_params}")
    print(f"Classifier head params: {classifier_params}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loss
    criterion = FocalLoss()

    # Training history
    history: List[Dict[str, Any]] = []

    # First batch debug flag
    first_batch = True

    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_scores = []

        for batch in dataloader:
            # Move batch to device
            batch = batch.to(args.device)
            optimizer.zero_grad()
            outputs = model(batch)
            logits_c = outputs["logits_c"]
            y = batch.y

            # First batch debug
            if first_batch:
                print("\nFirst batch debug:")
                print(f"logits_c.shape: {logits_c.shape}")
                print(f"y.shape: {y.shape}")
                print(f"y dtype: {y.dtype}")
                print(f"y unique values: {torch.unique(y).tolist()}")
                print(f"First 10 logits_c: {logits_c[:10].detach().cpu().numpy()}")
                probs = torch.softmax(logits_c, dim=1)
                print(f"First 10 softmax probabilities: {probs[:10].detach().cpu().numpy()}")
                print(f"First 10 labels: {y[:10].detach().cpu().numpy()}")
                
                # Check devices
                print(f"logits_c.device: {logits_c.device}")
                print(f"y.device: {y.device}")
                print(f"batch.x.device: {batch.x.device}")
                
                # Check embeddings
                print(f"filtered_x.std(): {outputs['filtered_x'].std().item()}")
                print(f"gc_x.std(): {outputs['gc_x'].std().item() if 'gc_x' in outputs else 'N/A'}")
                print(f"mask.mean(): {outputs['mask'].mean().item()}")
                print(f"mask.std(): {outputs['mask'].std().item()}")
                print(f"emb_gc.std(): {outputs['emb_gc'].std().item()}")
                
                first_batch = False

            # Compute loss
            loss = criterion(logits_c, y)
            
            # Print loss for first batch
            if not first_batch and epoch == 0:
                print(f"First batch loss: {loss.item()}")

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute metrics
            probs = torch.softmax(logits_c, dim=1)
            scores = probs[:, 1].detach().cpu().numpy()
            preds = torch.argmax(logits_c, dim=1).detach().cpu().numpy()
            labels = y.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_scores.extend(scores)

        # Compute epoch metrics
        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_scores)
        epoch_time = time.time() - start_time

        # Save history
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_ACC": acc,
            "train_AUC": auc,
            "time": epoch_time,
        })

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, train_ACC={acc:.4f}, train_AUC={auc:.4f}, time={epoch_time:.2f}s")

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)

    # Save summary
    summary = {
        "n_subjects": args.n_subjects,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "disable_adversarial": args.disable_adversarial,
        "final_loss": history[-1]["loss"],
        "final_train_ACC": history[-1]["train_ACC"],
        "final_train_AUC": history[-1]["train_AUC"],
        "max_train_ACC": max(h["train_ACC"] for h in history),
        "max_train_AUC": max(h["train_AUC"] for h in history),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\nFinal Summary:")
    print(f"Final loss: {summary['final_loss']:.4f}")
    print(f"Final train ACC: {summary['final_train_ACC']:.4f}")
    print(f"Final train AUC: {summary['final_train_AUC']:.4f}")
    print(f"Max train ACC: {summary['max_train_ACC']:.4f}")
    print(f"Max train AUC: {summary['max_train_AUC']:.4f}")


if __name__ == "__main__":
    main()
