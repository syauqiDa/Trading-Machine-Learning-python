from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.backtest.execution import entry_fill, exit_decision
from xauusd_ict_ml.backtest.orders import OrderType, Side
from xauusd_ict_ml.backtest.engine import run_backtest
from xauusd_ict_ml.config import BacktestConfig, load_config


def test_long_uses_ask_and_short_uses_bid():
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "bid": 99.8, "ask": 100.2})
    assert entry_fill(Side.LONG, OrderType.MARKET, 100, bar, .4, 0) == 100.2
    assert entry_fill(Side.SHORT, OrderType.MARKET, 100, bar, .4, 0) == 99.8


def test_ambiguity_defaults_to_stop_first():
    bar = pd.Series({"open": 100, "high": 111, "low": 89, "close": 100, "spread": 0.0})
    decision = exit_decision(Side.LONG, 90, 110, bar, 0, 0)
    assert decision and decision.reason == "stop" and decision.ambiguous


def test_concurrent_positions_are_processed_by_bar():
    times = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "open": [100]*3, "high": [100, 103, 103], "low": [100, 100, 100], "close": [100, 102, 102], "spread": [0.0]*3})
    setups = pd.DataFrame([
        {"setup_id": i, "entry_index": 0, "entry_timestamp": times[0], "direction": "bullish", "entry_price": 100, "stop_price": 99, "target_price": 102, "proposed_rr": 2, "session": "ny", "market_regime": "trend"}
        for i in range(2)
    ])
    config = load_config("config/backtest.yaml", BacktestConfig)
    config.risk_mode = "fixed_size"; config.fixed_position_size = 1; config.max_simultaneous_positions = 2; config.max_total_exposure = 2
    config.spread = 0; config.slippage = 0; config.commission_per_unit = 0
    result = run_backtest(frame, setups, config)
    assert len(result.trades) == 2


def test_partial_take_profit_is_accounted():
    times = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "open": [100]*3, "high": [100, 101.5, 100], "low": [100, 100, 100], "close": [100, 100.5, 100], "spread": [0.0]*3})
    setups = pd.DataFrame([{"setup_id": 1, "entry_index": 0, "entry_timestamp": times[0], "direction": "bullish", "entry_price": 100, "stop_price": 99, "target_price": 103, "proposed_rr": 3, "session": "ny", "market_regime": "trend"}])
    config = load_config("config/backtest.yaml", BacktestConfig)
    config.risk_mode = "fixed_size"; config.fixed_position_size = 1; config.max_total_exposure = 1
    config.max_holding_bars = 2; config.partial_take_profit_fraction = .5; config.partial_take_profit_at_r = 1
    config.spread = 0; config.slippage = 0; config.commission_per_unit = 0
    trade = run_backtest(frame, setups, config).trades.iloc[0]
    assert trade["gross_pnl"] == .5


def test_lower_timeframe_resolves_parent_bar_ambiguity():
    times = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame({"timestamp": times, "open": [100, 100], "high": [100, 102], "low": [100, 98], "close": [100, 100], "spread": [0.0, 0.0]})
    lower_times = pd.to_datetime(["2024-01-01 00:30Z", "2024-01-01 01:00Z"])
    lower = pd.DataFrame({"timestamp": lower_times, "open": [100, 101], "high": [102, 101], "low": [100, 98], "close": [101, 99], "spread": [0.0, 0.0]})
    setups = pd.DataFrame([{"setup_id": 1, "entry_index": 0, "entry_timestamp": times[0], "direction": "bullish", "entry_price": 100, "stop_price": 99, "target_price": 101, "proposed_rr": 1, "session": "ny", "market_regime": "trend"}])
    config = load_config("config/backtest.yaml", BacktestConfig)
    config.risk_mode = "fixed_size"; config.fixed_position_size = 1; config.max_total_exposure = 1
    config.spread = 0; config.slippage = 0; config.commission_per_unit = 0
    trade = run_backtest(frame, setups, config, lower).trades.iloc[0]
    assert trade["result"] == "target"
    assert not trade["ambiguous"]
