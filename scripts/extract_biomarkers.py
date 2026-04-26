from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cag.biomarker import extract_top_nodes, save_biomarker_outputs
from cag.data import BrainFCDataset, infer_n_nodes, read_subject_records
from cag.evaluate import load_model_checkpoint
from cag.train import make_model
from cag.utils import load_yaml, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CAG node-mask biomarkers from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subjects_csv", type=str, required=True)
    parser.add_argument("--fc_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--top_percent", type=float, default=0.03)
    parser.add_argument("--config", type=str, default="configs/cag_abide1.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(ROOT / args.config)
    device = select_device(args.device)
    records = read_subject_records(args.subjects_csv, args.fc_dir)
    dataset = BrainFCDataset(records)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = make_model(n_nodes=infer_n_nodes(records), config=config["model"]).to(device)
    checkpoint = load_model_checkpoint(model, args.checkpoint, device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        model = make_model(n_nodes=infer_n_nodes(records), config=checkpoint["config"]["model"]).to(device)
        load_model_checkpoint(model, args.checkpoint, device)
    dataframe, vector = extract_top_nodes(model, loader, top_percent=args.top_percent, asd_label=1, device=device)
    save_biomarker_outputs(dataframe, vector, args.out_dir)
    print(f"wrote biomarker outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
