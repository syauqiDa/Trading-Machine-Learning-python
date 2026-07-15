"""Purged chronological expanding and rolling windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def walk_forward_splits(
    data: pd.DataFrame, train_size: int, validation_size: int, test_size: int,
    step_size: int, purge_bars: int = 0, embargo_bars: int = 0, mode: str = "expanding",
) -> list[Fold]:
    """Return chronological indices with overlap purge and post-boundary embargo."""
    folds: list[Fold] = []
    test_start = train_size + validation_size
    while test_start + test_size <= len(data):
        validation_start = test_start - validation_size
        train_start = 0 if mode == "expanding" else max(0, validation_start - train_size)
        train_end = validation_start
        train = np.arange(train_start, train_end)
        validation = np.arange(validation_start, test_start)
        test = np.arange(test_start, test_start + test_size)
        train = _purge(data, train, validation_start, purge_bars)
        validation = _purge(data, validation, test_start, purge_bars)
        if embargo_bars:
            validation = validation[validation < test_start - embargo_bars]
        if len(train) and len(validation) and len(test):
            folds.append(Fold(train=train, validation=validation, test=test))
        test_start += step_size
    return folds


def _purge(data: pd.DataFrame, indices: np.ndarray, boundary_position: int, purge_bars: int) -> np.ndarray:
    if not len(indices):
        return indices
    boundary_start = int(data.iloc[boundary_position]["label_start_index"])
    label_ends = data.iloc[indices]["label_end_index"].to_numpy(dtype=int)
    starts = data.iloc[indices]["label_start_index"].to_numpy(dtype=int)
    keep = (label_ends < boundary_start) & (starts < boundary_start - purge_bars)
    return indices[keep]

