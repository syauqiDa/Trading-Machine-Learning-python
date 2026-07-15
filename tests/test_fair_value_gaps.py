from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.ict.fair_value_gaps import detect_fair_value_gaps


def test_gap_available_at_third_close_only():
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "open": [10, 11, 14, 14], "high": [11, 14, 15, 15], "low": [9, 10, 12, 10], "close": [10, 13, 14, 11]})
    gap = detect_fair_value_gaps(frame).iloc[0]
    assert gap["formation_timestamp"] == timestamps[2]
    assert gap["lower_bound"] == 11
    assert gap["upper_bound"] == 12
    assert gap["first_touch"] == timestamps[3]

