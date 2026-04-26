from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cag.data import read_subject_records
from cag.train import train_fold
from cag.utils import load_yaml, str_to_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small synthetic CAG smoke training job.")
    parser.add_argument("--config", type=str, default="configs/cag_synthetic.yaml")
    parser.add_argument("--subjects_csv", type=str, default=None)
    parser.add_argument("--fc_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mask_type", type=str, choices=["node", "edge", "node_edge"], default=None)
    parser.add_argument("--use_le1", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use_le2", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use_ls", type=int, choices=[0, 1], default=None)
    parser.add_argument("--n_env", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    return parser.parse_args()


def split_indices_from_csv(subjects_csv: str) -> tuple[list[int], list[int], list[int]]:
    dataframe = pd.read_csv(ROOT / subjects_csv if not Path(subjects_csv).is_absolute() else subjects_csv)
    if "split" in dataframe.columns:
        values = dataframe["split"].astype(str).str.lower()
        train = values[values == "train"].index.astype(int).tolist()
        val = values[values == "val"].index.astype(int).tolist()
        test = values[values == "test"].index.astype(int).tolist()
    else:
        total = len(dataframe)
        train = list(range(0, int(total * 0.7)))
        val = list(range(int(total * 0.7), int(total * 0.85)))
        test = list(range(int(total * 0.85), total))
    if not train or not val:
        raise ValueError("Synthetic smoke training requires non-empty train and val splits.")
    return train, val, test


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    if args.subjects_csv:
        config["data"]["subjects_csv"] = args.subjects_csv
    if args.fc_dir:
        config["data"]["fc_dir"] = args.fc_dir
    for key in ("epochs", "batch_size", "lr"):
        value = getattr(args, key)
        if value is not None:
            config["training"][key] = value
    for key in ("mask_type", "n_env", "n_layers"):
        value = getattr(args, key)
        if value is not None:
            config["model"][key] = value
    for key in ("use_le1", "use_le2", "use_ls"):
        value = getattr(args, key)
        if value is not None:
            config["model"][key] = str_to_bool(value)

    records = read_subject_records(config["data"]["subjects_csv"], config["data"]["fc_dir"])
    train_indices, val_indices, test_indices = split_indices_from_csv(config["data"]["subjects_csv"])
    result = train_fold(
        records=records,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=config,
        out_dir=config["training"]["output_dir"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
    )
    print(result)


if __name__ == "__main__":
    main()
