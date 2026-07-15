"""Deterministic synthetic XAUUSD-like bars for smoke tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_xauusd(rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Generate non-proprietary 15-minute OHLCV with injected volatility episodes."""
    if rows < 200:
        raise ValueError("At least 200 rows are required for representative tests")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2022-01-03", periods=rows, freq="15min", tz="UTC")
    cyclical = 0.08 * np.sin(np.arange(rows) / 31) + 0.04 * np.sin(np.arange(rows) / 7)
    innovations = rng.normal(0, 0.28, rows) + cyclical
    continuation = np.array([1.8, 1.45, 1.15, 0.9, 0.7, 0.5, 0.35, 0.2])
    for index in range(80, rows - len(continuation) - 1, 120):
        direction = 1 if (index // 120) % 2 else -1
        innovations[index] -= direction * 1.4
        innovations[index + 1 : index + 1 + len(continuation)] += direction * continuation
    close = 1800 + np.cumsum(innovations)
    open_price = np.r_[close[0] - innovations[0], close[:-1]]
    wick = rng.uniform(0.08, 0.42, rows)
    high = np.maximum(open_price, close) + wick
    low = np.minimum(open_price, close) - rng.uniform(0.08, 0.42, rows)
    spread = np.clip(rng.normal(0.30, 0.05, rows), 0.12, 0.55)
    mid = close
    return pd.DataFrame(
        {
            "timestamp": timestamps, "open": open_price, "high": high, "low": low, "close": close,
            "volume": rng.integers(100, 2000, rows), "bid": mid - spread / 2,
            "ask": mid + spread / 2, "spread": spread,
        }
    )
