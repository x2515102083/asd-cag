#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from cag.data import read_subject_records, load_fc_matrix
from cag.split import external_site_split


def extract_upper_triangle(fc_matrix: np.ndarray) -> np.ndarray:
    """Extract upper triangle without diagonal."""
    n = fc_matrix.shape[0]
    return fc_matrix[np.triu_indices(n, k=1)]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    sen = recall_score(y_true, y_pred, pos_label=1)
    spe = recall_score(y_true, y_pred, pos_label=0)
    return {
        "ACC": float(acc),
        "AUC": float(auc),
        "SEN": float(sen),
        "SPE": float(spe),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects_csv", type=str, required=True)
    parser.add_argument("--fc_dir", type=str, required=True)
    parser.add_argument("--test_site", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["logistic", "linear_svm", "rbf_svm"], required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="outputs/baseline_ml")
    args = parser.parse_args()

    # Load data
    records = read_subject_records(args.subjects_csv, args.fc_dir)
    train_indices, test_indices = external_site_split(records, args.test_site)

    # Prepare test data
    test_X = []
    test_y = []
    for idx in test_indices:
        record = records[idx]
        fc = load_fc_matrix(record.fc_path)
        feat = extract_upper_triangle(fc)
        test_X.append(feat)
        test_y.append(record.label)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    # Prepare train_val data
    train_val_X = []
    train_val_y = []
    for idx in train_indices:
        record = records[idx]
        fc = load_fc_matrix(record.fc_path)
        feat = extract_upper_triangle(fc)
        train_val_X.append(feat)
        train_val_y.append(record.label)
    train_val_X = np.array(train_val_X)
    train_val_y = np.array(train_val_y)

    # Initialize model
    if args.model == "logistic":
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear"
        )
    elif args.model == "linear_svm":
        model = LinearSVC(
            class_weight="balanced"
        )
    elif args.model == "rbf_svm":
        model = SVC(
            class_weight="balanced",
            kernel="rbf",
            probability=True
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Run cross-validation
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(train_val_X, train_val_y)):
        # Split data
        X_train, X_val = train_val_X[train_idx], train_val_X[val_idx]
        y_train, y_val = train_val_y[train_idx], train_val_y[val_idx]

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(test_X)

        # Train model
        model.fit(X_train_scaled, y_train)

        # Predict
        if args.model == "linear_svm":
            y_train_score = model.decision_function(X_train_scaled)
            y_val_score = model.decision_function(X_val_scaled)
            y_test_score = model.decision_function(X_test_scaled)
        else:
            y_train_score = model.predict_proba(X_train_scaled)[:, 1]
            y_val_score = model.predict_proba(X_val_scaled)[:, 1]
            y_test_score = model.predict_proba(X_test_scaled)[:, 1]

        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_score)
        val_metrics = compute_metrics(y_val, y_val_pred, y_val_score)
        test_metrics = compute_metrics(test_y, y_test_pred, y_test_score)

        fold_metric = {
            "fold": fold_idx + 1,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        fold_metrics.append(fold_metric)

        print(f"Fold {fold_idx + 1}:")
        print(f"  Train: ACC={train_metrics['ACC']:.4f}, AUC={train_metrics['AUC']:.4f}")
        print(f"  Val: ACC={val_metrics['ACC']:.4f}, AUC={val_metrics['AUC']:.4f}")
        print(f"  Test: ACC={test_metrics['ACC']:.4f}, AUC={test_metrics['AUC']:.4f}")

    # Compute summary
    summary = {
        "model": args.model,
        "test_site": args.test_site,
        "n_folds": args.folds,
        "n_train_val": len(train_indices),
        "n_test": len(test_indices),
        "folds": fold_metrics,
    }

    # Compute mean metrics
    mean_train = {k: np.mean([f["train"][k] for f in fold_metrics]) for k in ["ACC", "AUC", "SEN", "SPE"]}
    mean_val = {k: np.mean([f["val"][k] for f in fold_metrics]) for k in ["ACC", "AUC", "SEN", "SPE"]}
    mean_test = {k: np.mean([f["test"][k] for f in fold_metrics]) for k in ["ACC", "AUC", "SEN", "SPE"]}

    summary["mean_train"] = mean_train
    summary["mean_val"] = mean_val
    summary["mean_test"] = mean_test

    # Save results
    out_dir = Path(args.out_dir) / args.test_site / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save fold metrics
    fold_df = pd.DataFrame()
    for fold in fold_metrics:
        row = {
            "fold": fold["fold"],
            **{f"train_{k}": fold["train"][k] for k in ["ACC", "AUC", "SEN", "SPE"]},
            **{f"val_{k}": fold["val"][k] for k in ["ACC", "AUC", "SEN", "SPE"]},
            **{f"test_{k}": fold["test"][k] for k in ["ACC", "AUC", "SEN", "SPE"]},
        }
        fold_df = pd.concat([fold_df, pd.DataFrame([row])], ignore_index=True)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\nSummary:")
    print(f"Model: {args.model}")
    print(f"Test site: {args.test_site}")
    print(f"Mean Train: ACC={mean_train['ACC']:.4f}, AUC={mean_train['AUC']:.4f}")
    print(f"Mean Val: ACC={mean_val['ACC']:.4f}, AUC={mean_val['AUC']:.4f}")
    print(f"Mean Test: ACC={mean_test['ACC']:.4f}, AUC={mean_test['AUC']:.4f}")


if __name__ == "__main__":
    main()
