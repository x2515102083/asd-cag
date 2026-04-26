from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_1d(path: Path) -> np.ndarray:
    ts = np.loadtxt(path, comments="#", dtype=np.float32)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    if ts.ndim != 2:
        raise ValueError(f"Expected 2D time series at {path}, got shape={ts.shape}")
    return ts


def subject_id_from_1d(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rois_cc400"):
        stem = stem[: -len("_rois_cc400")]
    return stem


def compute_fc(time_series: np.ndarray) -> np.ndarray:
    fc = np.corrcoef(time_series.T)
    fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
    fc = 0.5 * (fc + fc.T)
    np.fill_diagonal(fc, 1.0)
    return fc.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ABIDE .1D ROI time series to FC .npy matrices.")
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.1D")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.raw_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {args.pattern!r} under {args.raw_dir}")
    for path in tqdm(paths, desc="make-fc"):
        fc = compute_fc(load_1d(path))
        np.save(args.out_dir / f"{subject_id_from_1d(path)}.npy", fc)
    print(f"converted files: {len(paths)}")


if __name__ == "__main__":
    main()
