"""Three-candle fair value gaps and lifecycle tracking."""

from __future__ import annotations

import pandas as pd


def detect_fair_value_gaps(
    frame: pd.DataFrame, atr: pd.Series | None = None, min_size_atr: float = 0.0,
    max_age_bars: int = 48, invalidate_on_close: bool = True,
) -> pd.DataFrame:
    """Return gaps available no earlier than candle three's close timestamp."""
    records: list[dict[str, object]] = []
    for third in range(2, len(frame)):
        first = third - 2
        bullish = frame.at[third, "low"] > frame.at[first, "high"]
        bearish = frame.at[third, "high"] < frame.at[first, "low"]
        if not bullish and not bearish:
            continue
        direction = "bullish" if bullish else "bearish"
        lower = float(frame.at[first, "high"] if bullish else frame.at[third, "high"])
        upper = float(frame.at[third, "low"] if bullish else frame.at[first, "low"])
        size = upper - lower
        atr_value = float(atr.iloc[third]) if atr is not None and pd.notna(atr.iloc[third]) else float("nan")
        if pd.notna(atr_value) and size / atr_value < min_size_atr:
            continue
        lifecycle = _track_gap(frame, third, direction, lower, upper, max_age_bars, invalidate_on_close)
        records.append(
            {
                "gap_id": len(records), "direction": direction,
                "formation_timestamp": frame.at[third, "timestamp"], "formation_index": third,
                "lower_bound": lower, "upper_bound": upper, "midpoint": (lower + upper) / 2,
                "size": size, "size_atr": size / atr_value if pd.notna(atr_value) and atr_value else pd.NA,
                **lifecycle,
            }
        )
    return pd.DataFrame(records)


def _track_gap(
    frame: pd.DataFrame, formed: int, direction: str, lower: float, upper: float,
    max_age: int, invalidate_on_close: bool,
) -> dict[str, object]:
    first_touch = partial_fill = complete_fill = invalidation = pd.NaT
    max_fill = 0.0
    end = min(len(frame), formed + max_age + 1)
    for index in range(formed + 1, end):
        row = frame.iloc[index]
        if direction == "bullish":
            touched = row["low"] <= upper
            fill = max(0.0, min(1.0, (upper - row["low"]) / (upper - lower)))
            complete = row["low"] <= lower
            invalid = row["close"] < lower if invalidate_on_close else complete
        else:
            touched = row["high"] >= lower
            fill = max(0.0, min(1.0, (row["high"] - lower) / (upper - lower)))
            complete = row["high"] >= upper
            invalid = row["close"] > upper if invalidate_on_close else complete
        if touched and pd.isna(first_touch):
            first_touch = row["timestamp"]
        if fill > 0 and pd.isna(partial_fill):
            partial_fill = row["timestamp"]
        max_fill = max(max_fill, fill)
        if complete and pd.isna(complete_fill):
            complete_fill = row["timestamp"]
        if invalid:
            invalidation = row["timestamp"]
            break
    return {
        "first_touch": first_touch, "partial_fill": partial_fill, "complete_fill": complete_fill,
        "fill_percentage": max_fill, "age_bars": min(max_age, len(frame) - formed - 1),
        "invalidation_timestamp": invalidation,
    }

