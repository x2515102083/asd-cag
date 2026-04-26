from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cag.data import infer_n_nodes, read_subject_records
from cag.train import make_model
from cag.utils import load_yaml, resolve_path


PAPER_ABIDE1_COUNTS = {"ASD": 486, "TDC": 531, "total": 1017}
PAPER_FC_SHAPE = (392, 392)
PAPER_LAMBDAS = {"lambda_e1": 1.0, "lambda_e2": 1.0, "lambda_s": 10.0}
PAPER_TABLE7_PARAMS = 323_020_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local ABIDE-I CAG setup against English paper defaults.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--subjects_csv", type=str, required=True)
    parser.add_argument("--fc_dir", type=str, required=True)
    return parser.parse_args()


def _parameter_count(module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _shape_counts(records) -> tuple[dict[str, int], int, list[dict[str, str]]]:
    counts: Counter[str] = Counter()
    missing_count = 0
    invalid: list[dict[str, str]] = []
    for record in records:
        if not record.fc_path.exists():
            missing_count += 1
            continue
        try:
            shape = tuple(int(value) for value in np.load(record.fc_path, mmap_mode="r").shape)
            counts["x".join(str(value) for value in shape)] += 1
        except Exception as exc:  # pragma: no cover - diagnostics should report and continue
            invalid.append({"subject_id": record.subject_id, "path": str(record.fc_path), "error": str(exc)})
    return dict(sorted(counts.items())), missing_count, invalid


def _lambda_report(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model", {})
    actual = {key: float(model_config.get(key, float("nan"))) for key in PAPER_LAMBDAS}
    return {
        "actual": actual,
        "expected": PAPER_LAMBDAS,
        "matches_paper_default": all(abs(actual[key] - expected) < 1e-12 for key, expected in PAPER_LAMBDAS.items()),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    records = read_subject_records(args.subjects_csv, args.fc_dir)

    label_counts = Counter(record.label for record in records)
    subject_report = {
        "total": len(records),
        "ASD": int(label_counts.get(1, 0)),
        "TDC": int(label_counts.get(0, 0)),
        "paper_expected": PAPER_ABIDE1_COUNTS,
    }
    subject_report["matches_paper_abide1"] = (
        subject_report["total"] == PAPER_ABIDE1_COUNTS["total"]
        and subject_report["ASD"] == PAPER_ABIDE1_COUNTS["ASD"]
        and subject_report["TDC"] == PAPER_ABIDE1_COUNTS["TDC"]
    )

    n_nodes = infer_n_nodes(records)
    fc_shape_counts, missing_fc_count, invalid_fc = _shape_counts(records)
    inferred_shape = [n_nodes, n_nodes]
    fc_report = {
        "inferred_shape": inferred_shape,
        "expected_shape": list(PAPER_FC_SHAPE),
        "inferred_shape_matches_paper": tuple(inferred_shape) == PAPER_FC_SHAPE,
        "shape_counts": fc_shape_counts,
        "all_existing_fc_shapes_match_paper": set(fc_shape_counts) <= {"392x392"} and bool(fc_shape_counts),
        "missing_fc_count": missing_fc_count,
        "invalid_fc_count": len(invalid_fc),
        "invalid_fc_examples": invalid_fc[:5],
    }

    model = make_model(n_nodes=n_nodes, config=config.get("model", {}))
    total_params = _parameter_count(model)
    ratio = total_params / PAPER_TABLE7_PARAMS
    table7_status = "match" if abs(total_params - PAPER_TABLE7_PARAMS) / PAPER_TABLE7_PARAMS <= 0.05 else "mismatch"
    if total_params < PAPER_TABLE7_PARAMS * 0.5:
        table7_status = "mismatch_far_smaller"
    param_report = {
        "total_params": total_params,
        "trainable_params": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "module_params": {name: _parameter_count(module) for name, module in model.named_children()},
        "paper_table7_total_params": PAPER_TABLE7_PARAMS,
        "ratio_to_paper_table7": ratio,
        "paper_table7_status": table7_status,
    }

    report = {
        "config_path": str(config_path),
        "subjects": subject_report,
        "fc": fc_report,
        "config_lambdas": _lambda_report(config),
        "parameters": param_report,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
