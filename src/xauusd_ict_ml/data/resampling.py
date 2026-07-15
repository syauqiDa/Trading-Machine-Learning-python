"""Leakage-safe OHLC aggregation."""

from __future__ import annotations

import pandas as pd


def resample_ohlc(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate right-closed bars stamped only when the full interval has closed."""
    indexed = frame.set_index("timestamp").sort_index()
    aggregation: dict[str, str] = {
        "open": "first", "high": "max", "low": "min", "close": "last"
    }
    for column in ["volume", "spread", "bid", "ask"]:
        if column in indexed:
            aggregation[column] = "sum" if column == "volume" else "last"
    result = indexed.resample(frequency, label="right", closed="right").agg(aggregation)
    return result.dropna(subset=["open", "high", "low", "close"]).reset_index()


def align_completed_timeframe(base: pd.DataFrame, higher: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Backward as-of join so a base bar sees only already-closed higher bars."""
    renamed = higher.rename(columns={c: f"{prefix}_{c}" for c in higher if c != "timestamp"})
    return pd.merge_asof(
        base.sort_values("timestamp"), renamed.sort_values("timestamp"), on="timestamp", direction="backward"
    )

