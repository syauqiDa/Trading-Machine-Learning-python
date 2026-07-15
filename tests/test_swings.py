from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.ict.swings import confirmed_swing_series, detect_swings


def test_swing_is_unavailable_until_confirmation():
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "open": [1]*5, "high": [1, 2, 5, 2, 1], "low": [0]*5, "close": [1]*5})
    swings = detect_swings(frame, 2, 2)
    swing = swings.iloc[0]
    assert swing["occurrence_time"] == timestamps[2]
    assert swing["confirmation_time"] == timestamps[4]
    attached = confirmed_swing_series(frame, swings)
    assert pd.isna(attached.loc[3, "confirmed_swing_high"])
    assert attached.loc[4, "confirmed_swing_high"] == 5

