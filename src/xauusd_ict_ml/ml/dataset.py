"""Candidate-level supervised dataset construction."""

from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.features.feature_pipeline import build_feature_frame
from xauusd_ict_ml.labels.target_before_stop import label_target_before_stop


def build_ml_dataset(
    frame: pd.DataFrame, setups: pd.DataFrame, max_holding_bars: int = 48,
    neutral_policy: str = "drop",
) -> pd.DataFrame:
    """Join decision-time features with separately generated forward labels."""
    features = build_feature_frame(frame, setups)
    labels = label_target_before_stop(frame, setups, max_holding_bars)
    dataset = features.merge(labels, on="setup_id", validate="one_to_one")
    if neutral_policy == "drop":
        dataset = dataset.dropna(subset=["label"]).copy()
    elif neutral_policy == "zero":
        dataset["label"] = dataset["label"].fillna(0)
    elif neutral_policy == "separate":
        dataset["label"] = dataset["label"].fillna(2)
    else:
        raise ValueError(f"Unknown neutral policy: {neutral_policy}")
    dataset["label"] = dataset["label"].astype(int)
    return dataset.sort_values("timestamp").reset_index(drop=True)

