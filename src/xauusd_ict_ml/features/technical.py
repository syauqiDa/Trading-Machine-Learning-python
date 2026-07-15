"""Past-and-current price and volatility features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_ict_ml.ict.displacement import average_true_range


def technical_features(frame: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Calculate trailing features whose window ends at each decision candle."""
    output = pd.DataFrame({"timestamp": frame["timestamp"]})
    atr = average_true_range(frame, atr_period)
    returns = frame["close"].pct_change()
    candle_range = frame["high"] - frame["low"]
    rolling_low = frame["low"].rolling(20, min_periods=2).min()
    rolling_high = frame["high"].rolling(20, min_periods=2).max()
    output["atr"] = atr
    output["realized_volatility"] = returns.rolling(20, min_periods=5).std()
    output["return_1"] = returns
    output["return_4"] = frame["close"].pct_change(4)
    output["candle_range_atr"] = candle_range / atr.replace(0, np.nan)
    output["body_ratio"] = (frame["close"] - frame["open"]).abs() / candle_range.replace(0, np.nan)
    output["distance_sma_20"] = frame["close"] / frame["close"].rolling(20, min_periods=2).mean() - 1
    output["rolling_position"] = (frame["close"] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    output["spread"] = frame["spread"] if "spread" in frame else np.nan
    volatility_percentile = output["realized_volatility"].rolling(100, min_periods=10).rank(pct=True)
    output["volatility_percentile"] = volatility_percentile
    return output

