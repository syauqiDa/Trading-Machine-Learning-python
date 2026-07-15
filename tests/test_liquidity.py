from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.ict.liquidity import detect_liquidity_sweeps


def test_sell_side_sweep_requires_close_reclaim():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "open": [101]*3, "high": [102]*3, "low": [100, 98, 99], "close": [101, 99, 101]})
    levels = pd.DataFrame([{"known_time": timestamps[0], "price": 100.0, "level_type": "swing_low", "side": "sell"}])
    sweep = detect_liquidity_sweeps(frame, levels, reclaim_bars=2).iloc[0]
    assert sweep["direction"] == "bullish"
    assert sweep["sweep_timestamp"] == timestamps[2]
    assert sweep["bars_to_reclaim"] == 2

