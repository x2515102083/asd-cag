from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from cag.data import SubjectRecord


def normalize_site(site: str) -> str:
    return str(site).strip().lower()


def external_site_split(records: list[SubjectRecord], test_site: str) -> tuple[list[int], list[int]]:
    target = normalize_site(test_site)
    test_indices = [idx for idx, record in enumerate(records) if normalize_site(record.site) == target]
    train_indices = [idx for idx, record in enumerate(records) if normalize_site(record.site) != target]
    if not test_indices:
        available = sorted({record.site for record in records})
        raise ValueError(f"Test site {test_site!r} not found. Available sites include: {available[:20]}")
    if not train_indices:
        raise ValueError(f"No training subjects remain after holding out site {test_site!r}.")
    labels = [records[idx].label for idx in test_indices]
    if len(set(labels)) < 2:
        raise ValueError(f"External test site {test_site!r} has fewer than two label classes.")
    return train_indices, test_indices


def stratified_cv_indices(
    records: list[SubjectRecord],
    candidate_indices: list[int],
    n_splits: int = 10,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    labels = np.asarray([records[idx].label for idx in candidate_indices], dtype=int)
    class_counts = np.bincount(labels, minlength=2)
    if np.any(class_counts < n_splits):
        raise ValueError(
            f"Cannot run {n_splits}-fold CV; class counts in non-test data are {class_counts.tolist()}."
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[list[int], list[int]]] = []
    positions = np.arange(len(candidate_indices))
    for train_pos, val_pos in splitter.split(positions, labels):
        train_indices = [candidate_indices[pos] for pos in train_pos]
        val_indices = [candidate_indices[pos] for pos in val_pos]
        folds.append((train_indices, val_indices))
    return folds
