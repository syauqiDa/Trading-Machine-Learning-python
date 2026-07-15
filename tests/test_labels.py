from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.labels.target_before_stop import label_target_before_stop


def test_label_future_window_is_separate(sample_bars):
    frame = sample_bars.iloc[:5].copy()
    entry = float(frame.at[0, "close"])
    frame.loc[1, "high"] = entry + 2
    setup = pd.DataFrame([{"setup_id": 1, "entry_index": 0, "direction": "bullish", "stop_price": entry - 1, "target_price": entry + 1}])
    label = label_target_before_stop(frame, setup, 4).iloc[0]
    assert label["label"] == 1
    assert label["label_start_index"] == 0
    assert label["label_end_index"] == 1

