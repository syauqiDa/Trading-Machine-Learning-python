"""Confirmed rolling dealing ranges and premium/discount zones."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_premium_discount(frame: pd.DataFrame, lookback_bars: int = 96) -> pd.DataFrame:
    """Use the completed trailing window ending at the current decision bar."""
    result = pd.DataFrame({"timestamp": frame["timestamp"]})
    result["range_high"] = frame["high"].rolling(lookback_bars, min_periods=2).max()
    result["range_low"] = frame["low"].rolling(lookback_bars, min_periods=2).min()
    result["equilibrium"] = (result["range_high"] + result["range_low"]) / 2
    width = (result["range_high"] - result["range_low"]).replace(0, np.nan)
    result["range_position"] = (frame["close"] - result["range_low"]) / width
    result["zone"] = np.select(
        [frame["close"] < result["equilibrium"], frame["close"] > result["equilibrium"]],
        ["discount", "premium"], default="equilibrium",
    )
    return result

