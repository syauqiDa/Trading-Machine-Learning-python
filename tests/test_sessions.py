from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.config import SessionWindow, SessionsConfig
from xauusd_ict_ml.data.sessions import assign_sessions, completed_session_levels, previous_completed_levels


def test_new_york_session_tracks_dst():
    timestamps = pd.Series(pd.to_datetime(["2024-01-15 12:00Z", "2024-07-15 11:00Z"]))
    config = SessionsConfig(timezone="America/New_York", windows={"ny": SessionWindow(start="07:00", end="08:00")})
    assert assign_sessions(timestamps, config).tolist() == ["ny", "ny"]


def test_previous_day_uses_completed_day_only():
    timestamp = pd.date_range("2024-01-01", periods=8, freq="12h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamp, "open": range(8), "high": range(1, 9), "low": range(8), "close": range(8)})
    levels = previous_completed_levels(frame, "1D")
    second_day = levels[levels["timestamp"] == pd.Timestamp("2024-01-02", tz="UTC")].iloc[0]
    assert second_day["previous_high"] == 2
    assert second_day["previous_low"] == 0


def test_session_level_known_after_window_ends():
    timestamps = pd.Series(pd.date_range("2024-01-01 07:00", periods=4, freq="h", tz="America/New_York").tz_convert("UTC"))
    frame = pd.DataFrame({"timestamp": timestamps, "open": [1]*4, "high": [2, 3, 4, 5], "low": [0, -1, -2, -3], "close": [1]*4})
    config = SessionsConfig(timezone="America/New_York", windows={"ny": SessionWindow(start="07:00", end="10:00")})
    levels = completed_session_levels(frame, config)
    assert set(levels["known_time"]) == {timestamps.iloc[3]}
    assert levels.loc[levels["side"] == "buy", "price"].iloc[0] == 4
