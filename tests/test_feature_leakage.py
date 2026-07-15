from __future__ import annotations

import pandas as pd
import pytest

from xauusd_ict_ml.features.feature_pipeline import assert_no_forbidden_features, build_feature_frame


def test_forbidden_outcome_column_rejected():
    with pytest.raises(ValueError):
        assert_no_forbidden_features(["atr", "final_trade_outcome"])


def test_future_mutation_does_not_change_past_features(sample_bars):
    setup = pd.DataFrame([{"setup_id": 0, "entry_index": 100, "entry_timestamp": sample_bars.at[100, "timestamp"], "direction": "bullish", "entry_price": 1, "stop_price": 0, "target_price": 2, "session": "ny", "market_regime": "range", "proposed_rr": 1}])
    before = build_feature_frame(sample_bars, setup)
    mutated = sample_bars.copy(); mutated.loc[101:, "close"] *= 10
    after = build_feature_frame(mutated, setup)
    pd.testing.assert_frame_equal(before, after)

