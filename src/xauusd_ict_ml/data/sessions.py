"""DST-safe session and completed-period liquidity helpers."""

from __future__ import annotations

from datetime import time

import pandas as pd

from xauusd_ict_ml.config import SessionsConfig


def _parse_clock(value: str) -> time:
    return time.fromisoformat(value)


def assign_sessions(timestamps: pd.Series, config: SessionsConfig) -> pd.Series:
    """Assign named local-time windows using IANA timezone conversion."""
    local = pd.to_datetime(timestamps, utc=True).dt.tz_convert(config.timezone)
    result = pd.Series("off_session", index=timestamps.index, dtype="object")
    clocks = local.dt.time
    for name, window in config.windows.items():
        if not window.enabled:
            continue
        start, end = _parse_clock(window.start), _parse_clock(window.end)
        mask = (clocks >= start) & (clocks < end) if start < end else (clocks >= start) | (clocks < end)
        result.loc[mask] = name
    return result


def previous_completed_levels(frame: pd.DataFrame, period: str = "D") -> pd.DataFrame:
    """Return prior completed day/week highs and lows, never current-period values."""
    indexed = frame.set_index("timestamp").sort_index()
    grouped = indexed.resample(period, label="left", closed="left").agg({"high": "max", "low": "min"})
    prior = grouped.shift(1).rename(columns={"high": "previous_high", "low": "previous_low"})
    # Period labels are availability timestamps; forward-fill only derived completed levels.
    return pd.merge_asof(
        frame[["timestamp"]].sort_values("timestamp"),
        prior.reset_index().sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )


def completed_session_levels(frame: pd.DataFrame, config: SessionsConfig) -> pd.DataFrame:
    """Emit session highs/lows only at the first bar after each session ends."""
    names = assign_sessions(frame["timestamp"], config)
    blocks = names.ne(names.shift()).cumsum()
    records: list[dict[str, object]] = []
    for _, positions in frame.groupby(blocks).groups.items():
        indices = list(positions)
        name = str(names.iloc[indices[0]])
        final_index = indices[-1]
        if name == "off_session" or final_index + 1 >= len(frame):
            continue
        segment = frame.iloc[indices]
        known_time = frame.iloc[final_index + 1]["timestamp"]
        records.extend(
            [
                {"known_time": known_time, "price": float(segment["high"].max()), "level_type": f"{name}_session_high", "side": "buy"},
                {"known_time": known_time, "price": float(segment["low"].min()), "level_type": f"{name}_session_low", "side": "sell"},
            ]
        )
    return pd.DataFrame(records, columns=["known_time", "price", "level_type", "side"])
