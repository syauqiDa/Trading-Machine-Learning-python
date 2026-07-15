"""Multi-condition displacement detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ict_ml.config import DisplacementConfig


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style true-range rolling mean using current and prior bars only."""
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def detect_displacement(
    frame: pd.DataFrame, config: DisplacementConfig, structure: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Score direction, range, body, close location, volume, and structure break."""
    result = pd.DataFrame({"timestamp": frame["timestamp"]})
    atr = average_true_range(frame, config.atr_period)
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    body_ratio = (frame["close"] - frame["open"]).abs() / candle_range
    close_location = (frame["close"] - frame["low"]) / candle_range
    range_atr = candle_range / atr
    volume_ratio = pd.Series(1.0, index=frame.index)
    if "volume" in frame:
        volume_ratio = frame["volume"] / frame["volume"].rolling(config.atr_period, min_periods=1).mean()
    bullish_base = (
        (frame["close"] > frame["open"]) & (range_atr >= config.min_range_atr)
        & (body_ratio >= config.min_body_ratio) & (close_location >= config.min_close_location)
        & (volume_ratio >= config.min_volume_ratio)
    )
    bearish_base = (
        (frame["close"] < frame["open"]) & (range_atr >= config.min_range_atr)
        & (body_ratio >= config.min_body_ratio) & (close_location <= 1 - config.min_close_location)
        & (volume_ratio >= config.min_volume_ratio)
    )
    if config.min_consecutive_bars > 1:
        up = frame["close"] > frame["open"]
        down = frame["close"] < frame["open"]
        bullish_base &= up.rolling(config.min_consecutive_bars).sum().eq(config.min_consecutive_bars)
        bearish_base &= down.rolling(config.min_consecutive_bars).sum().eq(config.min_consecutive_bars)
    structure_event = structure["structure_event"] if structure is not None else pd.Series("none", index=frame.index)
    if config.require_structure_break:
        bullish_base &= structure_event.str.startswith("bullish")
        bearish_base &= structure_event.str.startswith("bearish")
    result["atr"] = atr
    result["range_atr"] = range_atr
    result["body_ratio"] = body_ratio
    result["close_location"] = close_location
    result["volume_ratio"] = volume_ratio
    result["bullish_displacement"] = bullish_base.fillna(False)
    result["bearish_displacement"] = bearish_base.fillna(False)
    result["structure_break"] = structure_event.str.endswith(("bos", "mss"))
    return result

