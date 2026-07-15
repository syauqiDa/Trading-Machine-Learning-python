from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.ml.splits import walk_forward_splits
from xauusd_ict_ml.config import ModelConfig, load_config
from xauusd_ict_ml.ml.walk_forward import run_walk_forward


def test_walk_forward_is_chronological_and_purged():
    data = pd.DataFrame({"label_start_index": range(100), "label_end_index": [i + 3 for i in range(100)]})
    folds = walk_forward_splits(data, 30, 10, 10, 10, purge_bars=2, embargo_bars=1)
    assert folds
    for fold in folds:
        assert fold.train.max() < fold.validation.min() < fold.test.min()
        boundary = data.iloc[fold.validation.min()]["label_start_index"]
        assert (data.iloc[fold.train]["label_end_index"] < boundary).all()


def test_final_test_block_is_reported_once():
    rows = 114
    data = pd.DataFrame({
        "setup_id": range(rows), "label_start_index": [i * 5 for i in range(rows)],
        "label_end_index": [i * 5 + 2 for i in range(rows)], "label": [0, 1] * (rows // 2),
        "direction": [1, -1] * (rows // 2), "atr": range(1, rows + 1),
        "realized_volatility": .1, "return_1": .01, "return_4": .02,
        "candle_range_atr": 1.0, "body_ratio": .5, "rolling_position": .5,
        "spread": .3, "hour": 8, "weekday": 2, "session": ["asia", "ny"] * (rows // 2),
        "volatility_regime": ["low", "high"] * (rows // 2), "sweep_penetration": .2,
        "sweep_recovery_strength": 1.0, "fvg_size_atr": .1, "proposed_rr": 2.0,
    })
    config = load_config("config/model.yaml", ModelConfig)
    config.models = ["logistic"]
    metrics, predictions = run_walk_forward(data, config)
    final_ids = set(data.iloc[int(rows * .8):].setup_id)
    reported = predictions[predictions["fold"] == "final_untouched"]
    assert set(reported.setup_id) == final_ids
