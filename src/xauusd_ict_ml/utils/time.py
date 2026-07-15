"""Timezone-safe time helpers."""

from __future__ import annotations

import pandas as pd


def normalize_timestamp(series: pd.Series, source_timezone: str) -> pd.Series:
    """Parse timestamps, localize naive values, and normalize to UTC."""
    parsed = pd.to_datetime(series, errors="raise")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(source_timezone, ambiguous="raise", nonexistent="shift_forward")
    return parsed.dt.tz_convert("UTC")


def completed_period_start(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """Return UTC period starts for grouping completed calendar periods."""
    if index.tz is None:
        raise ValueError("DatetimeIndex must be timezone-aware")
    return index.floor(frequency)

