from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cag.data import load_fc_matrix, read_subject_records
from cag.utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ABIDE-style CAG dataset metadata and FC files.")
    parser.add_argument("--subjects_csv", type=str, required=True)
    parser.add_argument("--fc_dir", type=str, required=True)
    parser.add_argument("--raw_1d_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_subject_records(args.subjects_csv, args.fc_dir)
    missing = [record for record in records if not record.fc_path.exists()]
    existing = [record for record in records if record.fc_path.exists()]
    shape = None
    if existing:
        shape = tuple(load_fc_matrix(existing[0].fc_path).shape)

    print(f"number of subjects: {len(records)}")
    print(f"labels count: {dict(Counter(record.label for record in records))}")
    print(f"sites count: {dict(Counter(record.site for record in records))}")
    print(f"missing fc count: {len(missing)}")
    print(f"fc shape: {shape}")
    if args.raw_1d_dir:
        raw_dir = resolve_path(args.raw_1d_dir)
        raw_count = len(list(raw_dir.glob("*.1D"))) if raw_dir.exists() else 0
        print(f"raw .1D count: {raw_count}")


if __name__ == "__main__":
    main()
