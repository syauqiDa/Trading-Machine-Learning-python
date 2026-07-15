from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.data.resampling import align_completed_timeframe, resample_ohlc


def test_incomplete_higher_timeframe_is_not_exposed():
    timestamps = pd.date_range("2024-01-01 00:15", periods=7, freq="15min", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "open": range(7), "high": range(1, 8), "low": range(7), "close": range(7)})
    hourly = resample_ohlc(frame, "1h")
    aligned = align_completed_timeframe(frame, hourly, "h1")
    # The 00:30 base bar cannot see the still-forming 01:00 hourly bar.
    assert pd.isna(aligned.loc[1, "h1_close"])
    assert aligned.loc[3, "h1_close"] == 3
