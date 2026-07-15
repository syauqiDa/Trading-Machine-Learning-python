"""Confirmed swing detection without early pivot exposure."""

from __future__ import annotations

import pandas as pd


def detect_swings(frame: pd.DataFrame, left_bars: int = 3, right_bars: int = 3) -> pd.DataFrame:
    """Return swing events stamped with occurrence and later confirmation times.

    Equality does not qualify: the candidate must be strictly beyond every
    neighbor. A pivot at row ``i`` becomes usable only at ``i + right_bars``.
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be positive")
    records: list[dict[str, object]] = []
    for index in range(left_bars, len(frame) - right_bars):
        before = frame.iloc[index - left_bars : index]
        after = frame.iloc[index + 1 : index + right_bars + 1]
        candidate = frame.iloc[index]
        if candidate["high"] > before["high"].max() and candidate["high"] > after["high"].max():
            records.append(
                {
                    "swing_type": "high",
                    "price": float(candidate["high"]),
                    "occurrence_time": candidate["timestamp"],
                    "confirmation_time": frame.iloc[index + right_bars]["timestamp"],
                    "occurrence_index": index,
                    "confirmation_index": index + right_bars,
                }
            )
        if candidate["low"] < before["low"].min() and candidate["low"] < after["low"].min():
            records.append(
                {
                    "swing_type": "low",
                    "price": float(candidate["low"]),
                    "occurrence_time": candidate["timestamp"],
                    "confirmation_time": frame.iloc[index + right_bars]["timestamp"],
                    "occurrence_index": index,
                    "confirmation_index": index + right_bars,
                }
            )
    columns = [
        "swing_type", "price", "occurrence_time", "confirmation_time",
        "occurrence_index", "confirmation_index",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        "confirmation_time", ignore_index=True
    ) if records else pd.DataFrame(columns=columns)


def confirmed_swing_series(frame: pd.DataFrame, swings: pd.DataFrame) -> pd.DataFrame:
    """Attach the most recently confirmed high and low to every bar."""
    result = frame[["timestamp"]].copy().sort_values("timestamp")
    for kind in ("high", "low"):
        events = swings.loc[swings["swing_type"] == kind, ["confirmation_time", "price"]].rename(
            columns={"confirmation_time": "timestamp", "price": f"confirmed_swing_{kind}"}
        )
        if events.empty:
            result[f"confirmed_swing_{kind}"] = float("nan")
        else:
            result = pd.merge_asof(
                result.sort_values("timestamp"), events.sort_values("timestamp"),
                on="timestamp", direction="backward",
            )
    return result

