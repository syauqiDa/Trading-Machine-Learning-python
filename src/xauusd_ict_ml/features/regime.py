"""Trailing volatility, trend, range, drawdown, and spread regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify regimes from expanding/rolling information available by each bar."""
    returns = frame["close"].pct_change()
    volatility = returns.rolling(20, min_periods=5).std()
    median_volatility = volatility.expanding(min_periods=20).median()
    sma_fast = frame["close"].rolling(10, min_periods=2).mean()
    sma_slow = frame["close"].rolling(40, min_periods=5).mean()
    drawdown = frame["close"] / frame["close"].cummax() - 1
    spread = frame["spread"] if "spread" in frame else pd.Series(np.nan, index=frame.index)
    spread_median = spread.expanding(min_periods=5).median()
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "volatility_regime": np.where(volatility > median_volatility, "high", "low"),
            "trend_regime": np.where(sma_fast > sma_slow, "up", "down"),
            "range_regime": np.where((sma_fast / sma_slow - 1).abs() < 0.001, "range", "trend"),
            "recent_drawdown_regime": np.where(drawdown < -0.02, "drawdown", "normal"),
            "spread_regime": np.where(spread > spread_median, "wide", "normal"),
        }
    )

