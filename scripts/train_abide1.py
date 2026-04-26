from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cag.data import read_subject_records
from cag.train import train_external_site_cv
from cag.utils import load_yaml, str_to_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAG with external-site ABIDE-I cross-validation.")
    parser.add_argument("--config", type=str, default="configs/cag_abide1.yaml")
    parser.add_argument("--subjects_csv", type=str, default="data/abide1/subjects.csv")
    parser.add_argument("--fc_dir", type=str, default="data/abide1/fc")
    parser.add_argument("--test_site", type=str, default="Trinity")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n_env", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--lambda_e1", type=float, default=50.0)
    parser.add_argument("--lambda_e2", type=float, default=20.0)
    parser.add_argument("--lambda_s", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mask_type", type=str, choices=["node", "edge", "node_edge"], default="node")
    parser.add_argument("--use_le1", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_le2", type=int, choices=[0, 1], default=1)
    parser.add_argument("--use_ls", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subjects_path = Path(args.subjects_csv)
    if not subjects_path.is_absolute():
        subjects_path = ROOT / subjects_path
    if not subjects_path.exists():
        raise FileNotFoundError(
            "ABIDE-I subject metadata csv is required. Provide a CSV with at least "
            "subject_id,label,site columns via --subjects_csv."
        )

    config = load_yaml(ROOT / args.config)
    config["data"]["subjects_csv"] = args.subjects_csv
    config["data"]["fc_dir"] = args.fc_dir
    config["training"]["test_site"] = args.test_site
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["lr"] = args.lr
    config["training"]["seed"] = args.seed
    config["training"]["n_splits"] = args.n_splits
    config["training"]["device"] = args.device
    config["model"]["n_env"] = args.n_env
    config["model"]["n_layers"] = args.n_layers
    config["model"]["lambda_e1"] = args.lambda_e1
    config["model"]["lambda_e2"] = args.lambda_e2
    config["model"]["lambda_s"] = args.lambda_s
    config["model"]["dropout"] = args.dropout
    config["model"]["mask_type"] = args.mask_type
    config["model"]["use_le1"] = str_to_bool(args.use_le1)
    config["model"]["use_le2"] = str_to_bool(args.use_le2)
    config["model"]["use_ls"] = str_to_bool(args.use_ls)

    records = read_subject_records(config["data"]["subjects_csv"], config["data"]["fc_dir"])
    result = train_external_site_cv(records=records, test_site=args.test_site, config=config)
    print(result["summary"])


if __name__ == "__main__":
    main()
