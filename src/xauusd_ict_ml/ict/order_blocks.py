"""Conservative displacement order blocks."""

from __future__ import annotations

import pandas as pd


def detect_order_blocks(
    frame: pd.DataFrame, displacement: pd.DataFrame, max_lookback_bars: int = 8,
    max_age_bars: int = 64, boundary: str = "full_candle",
) -> pd.DataFrame:
    """Find the last opposing candle before structure-breaking displacement."""
    records: list[dict[str, object]] = []
    for index in range(len(frame)):
        for direction, flag in [("bullish", "bullish_displacement"), ("bearish", "bearish_displacement")]:
            if not displacement.at[index, flag] or not displacement.at[index, "structure_break"]:
                continue
            for candidate in range(index - 1, max(-1, index - max_lookback_bars - 1), -1):
                opposing = frame.at[candidate, "close"] < frame.at[candidate, "open"] if direction == "bullish" else frame.at[candidate, "close"] > frame.at[candidate, "open"]
                if not opposing:
                    continue
                if boundary == "body":
                    lower = min(frame.at[candidate, "open"], frame.at[candidate, "close"])
                    upper = max(frame.at[candidate, "open"], frame.at[candidate, "close"])
                else:
                    lower, upper = frame.at[candidate, "low"], frame.at[candidate, "high"]
                invalidation = pd.NaT
                for future in range(index + 1, min(len(frame), index + max_age_bars + 1)):
                    crossed = frame.at[future, "close"] < lower if direction == "bullish" else frame.at[future, "close"] > upper
                    if crossed:
                        invalidation = frame.at[future, "timestamp"]
                        break
                records.append(
                    {
                        "direction": direction, "candle_timestamp": frame.at[candidate, "timestamp"],
                        "confirmation_timestamp": frame.at[index, "timestamp"], "lower_bound": float(lower),
                        "upper_bound": float(upper), "invalidation_timestamp": invalidation,
                        "max_age_bars": max_age_bars, "definition": f"last_opposing_{boundary}",
                    }
                )
                break
    return pd.DataFrame(records)

