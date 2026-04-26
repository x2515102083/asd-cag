from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cag.graph import fc_to_pyg_data
from cag.utils import resolve_path

try:
    from torch_geometric.data import Dataset as PyGDataset
except ImportError:
    PyGDataset = object


SUBJECT_ID_COLUMNS = ("subject_id", "id", "sub_id")
LABEL_COLUMNS = ("label", "dx", "diagnosis", "y")
SITE_COLUMNS = ("site", "site_id", "center")


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    fc_path: Path
    label: int
    site: str


def _find_column(columns: Iterable[str], candidates: tuple[str, ...], required: bool = True) -> str | None:
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    if required:
        raise ValueError(f"Missing required column. Expected one of: {', '.join(candidates)}")
    return None


def normalize_label(value: object) -> int:
    text = str(value).strip().upper()
    if text in {"1", "ASD", "AUTISM", "AUTISTIC", "PATIENT"}:
        return 1
    if text in {"0", "2", "TDC", "TC", "CONTROL", "CONTROLS", "HC", "TD"}:
        return 0
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number):
        if int(number) == 1:
            return 1
        if int(number) in {0, 2}:
            return 0
    raise ValueError(f"Unsupported label value: {value!r}")


def load_fc_matrix(path: str | Path) -> np.ndarray:
    matrix = np.load(path)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"FC matrix at {path} must be square, got shape={matrix.shape}")
    return matrix


def _resolve_fc_path(subject_id: str, site: str, fc_dir: Path) -> Path:
    candidates = [
        fc_dir / f"{subject_id}.npy",
        fc_dir / f"{site}_{subject_id}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(path for path in fc_dir.glob("*.npy") if subject_id in path.name)
    if matches:
        return matches[0]
    return candidates[0]


def read_subject_records(subjects_csv: str | Path, fc_dir: str | Path) -> list[SubjectRecord]:
    csv_path = resolve_path(subjects_csv)
    fc_root = resolve_path(fc_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"subjects.csv not found: {csv_path}")
    if not fc_root.exists():
        raise FileNotFoundError(f"FC directory not found: {fc_root}")

    dataframe = pd.read_csv(csv_path)
    subject_col = _find_column(dataframe.columns, SUBJECT_ID_COLUMNS)
    label_col = _find_column(dataframe.columns, LABEL_COLUMNS)
    site_col = _find_column(dataframe.columns, SITE_COLUMNS)
    fc_col = _find_column(dataframe.columns, ("fc_path",), required=False)

    records: list[SubjectRecord] = []
    for _, row in dataframe.iterrows():
        subject_id = str(row[subject_col]).strip()
        site = str(row[site_col]).strip()
        if fc_col and pd.notna(row[fc_col]) and str(row[fc_col]).strip():
            candidate = Path(str(row[fc_col]).strip())
            fc_path = candidate if candidate.is_absolute() else (csv_path.parent / candidate).resolve()
            if not fc_path.exists():
                fc_path = resolve_path(candidate)
        else:
            fc_path = _resolve_fc_path(subject_id=subject_id, site=site, fc_dir=fc_root)
        records.append(
            SubjectRecord(
                subject_id=subject_id,
                fc_path=fc_path,
                label=normalize_label(row[label_col]),
                site=site,
            )
        )
    return records


class BrainFCDataset(PyGDataset):
    def __init__(
        self,
        records: list[SubjectRecord],
        site_to_idx: dict[str, int] | None = None,
        indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.all_records = list(records)
        self.record_indices = list(range(len(records))) if indices is None else list(indices)
        if site_to_idx is None:
            site_to_idx = {site: idx for idx, site in enumerate(sorted({record.site for record in records}))}
        self.site_to_idx = site_to_idx

    def len(self) -> int:
        return len(self.record_indices)

    def get(self, idx: int):
        record_index = self.record_indices[idx]
        record = self.all_records[record_index]
        fc = load_fc_matrix(record.fc_path)
        return fc_to_pyg_data(
            fc=fc,
            label=record.label,
            site_idx=self.site_to_idx[record.site],
            subject_id=record.subject_id,
            record_index=record_index,
        )


def make_dataset(records: list[SubjectRecord], indices: list[int] | None = None) -> BrainFCDataset:
    site_to_idx = {site: idx for idx, site in enumerate(sorted({record.site for record in records}))}
    return BrainFCDataset(records=records, site_to_idx=site_to_idx, indices=indices)


def infer_n_nodes(records: list[SubjectRecord]) -> int:
    if not records:
        raise ValueError("Cannot infer node count from an empty record list.")
    return int(load_fc_matrix(records[0].fc_path).shape[0])
