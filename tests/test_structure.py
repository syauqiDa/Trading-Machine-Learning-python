from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.ict.structure import detect_market_structure


def test_bos_uses_confirmed_level_only():
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "open": [9]*4, "high": [9, 10, 10, 12], "low": [8]*4, "close": [9, 9, 9, 11]})
    swings = pd.DataFrame([{"swing_type": "high", "price": 10.0, "occurrence_time": timestamps[1], "confirmation_time": timestamps[2]}])
    result = detect_market_structure(frame, swings)
    assert result.loc[:2, "structure_event"].eq("none").all()
    assert result.loc[3, "structure_event"] == "bullish_bos"

