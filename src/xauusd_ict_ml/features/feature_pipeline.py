"""Leakage-checked feature-matrix assembly."""

from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.features.ict_features import ict_setup_features
from xauusd_ict_ml.features.regime import regime_features
from xauusd_ict_ml.features.technical import technical_features

FORBIDDEN_TOKENS = ("future", "outcome", "mfe", "mae", "label", "target_hit", "stop_hit")


def build_feature_frame(frame: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    """Sample trailing market features exactly at each setup entry index."""
    market_features = technical_features(frame).merge(regime_features(frame), on="timestamp")
    sampled = market_features.iloc[setups["entry_index"].astype(int)].reset_index(drop=True)
    sampled.insert(0, "setup_id", setups["setup_id"].to_numpy())
    local = ict_setup_features(setups)
    output = sampled.merge(local, on="setup_id", how="left", suffixes=("", "_setup"))
    timestamps = pd.to_datetime(output["timestamp"], utc=True)
    decisions = pd.to_datetime(setups["entry_timestamp"], utc=True).reset_index(drop=True)
    if (timestamps.reset_index(drop=True) > decisions).any():
        raise AssertionError("A market feature timestamp occurs after its decision timestamp")
    assert_no_forbidden_features([column for column in output if column not in {"setup_id", "timestamp"}])
    local_session = pd.to_datetime(output["timestamp"], utc=True).dt.tz_convert("America/New_York")
    output["hour"] = local_session.dt.hour
    output["minute_bucket"] = local_session.dt.minute // 15
    output["weekday"] = local_session.dt.weekday
    return output


def assert_no_forbidden_features(columns: list[str]) -> None:
    """Fail closed when a feature name signals forward outcome information."""
    forbidden = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise ValueError(f"Forbidden leakage-prone feature columns: {forbidden}")

