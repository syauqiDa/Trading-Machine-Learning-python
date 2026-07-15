"""Data-quality checks with machine-readable summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    start: str | None
    end: str | None
    duplicate_timestamps: int
    missing_intervals: int
    invalid_ohlc: int
    nonpositive_prices: int
    negative_spreads: int
    outliers: int
    timezone: str | None
    bid_ask_available: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_market_data(
    frame: pd.DataFrame, expected_frequency: str | None = None, outlier_zscore: float = 8.0
) -> DataQualityReport:
    """Validate invariants without mutating or silently repairing the input."""
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    ts = pd.DatetimeIndex(frame["timestamp"])
    duplicate_count = int(ts.duplicated().sum())
    invalid_ohlc = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    nonpositive = int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    negative_spread = int((frame["spread"] < 0).sum()) if "spread" in frame else 0
    missing_intervals = 0
    if expected_frequency and len(ts) > 1:
        expected = pd.date_range(ts.min(), ts.max(), freq=expected_frequency)
        missing_intervals = int(len(expected.difference(ts.unique())))
    returns = frame["close"].pct_change()
    median = returns.median()
    mad = (returns - median).abs().median()
    robust_z = 0.6745 * (returns - median) / (mad if mad > 0 else np.nan)
    outliers = int((robust_z.abs() > outlier_zscore).sum())
    return DataQualityReport(
        rows=len(frame),
        start=ts.min().isoformat() if len(ts) else None,
        end=ts.max().isoformat() if len(ts) else None,
        duplicate_timestamps=duplicate_count,
        missing_intervals=missing_intervals,
        invalid_ohlc=invalid_ohlc,
        nonpositive_prices=nonpositive,
        negative_spreads=negative_spread,
        outliers=outliers,
        timezone=str(ts.tz) if ts.tz is not None else None,
        bid_ask_available={"bid", "ask"}.issubset(frame.columns),
    )


def assert_valid_market_data(report: DataQualityReport) -> None:
    """Raise for errors that make chronological research unsafe."""
    violations = {
        "duplicate timestamps": report.duplicate_timestamps,
        "invalid OHLC bars": report.invalid_ohlc,
        "nonpositive prices": report.nonpositive_prices,
        "negative spreads": report.negative_spreads,
    }
    found = {name: count for name, count in violations.items() if count}
    if found:
        raise ValueError(f"Data validation failed: {found}")

