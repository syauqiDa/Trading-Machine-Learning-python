from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.backtest.engine import run_backtest
from xauusd_ict_ml.config import BacktestConfig, load_config
from xauusd_ict_ml.ml.dataset import build_ml_dataset


def test_complete_sample_pipeline_reproducible(sample_bars, tmp_path):
    entry = float(sample_bars.at[100, "ask"])
    setups = pd.DataFrame([{
        "setup_id": 0, "setup_timestamp": sample_bars.at[100, "timestamp"], "entry_timestamp": sample_bars.at[100, "timestamp"],
        "entry_index": 100, "direction": "bullish", "entry_price": entry, "stop_price": entry - 2,
        "target_price": entry + 2, "proposed_rr": 1.0, "session": "new_york", "market_regime": "range",
    }])
    config = load_config("config/backtest.yaml", BacktestConfig)
    first = run_backtest(sample_bars, setups, config)
    second = run_backtest(sample_bars, setups, config)
    pd.testing.assert_frame_equal(first.trades, second.trades)
    if not first.trades.empty:
        assert (first.trades["net_pnl"] == first.trades["gross_pnl"] - first.trades["costs"]).all()
    dataset = build_ml_dataset(sample_bars, setups, 20, "zero")
    assert len(dataset) == 1
