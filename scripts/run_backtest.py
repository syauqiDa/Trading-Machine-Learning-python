#!/usr/bin/env python3
"""Run the rule-based event-driven backtest."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.backtest.engine import run_backtest
from xauusd_ict_ml.config import BacktestConfig, load_config
from xauusd_ict_ml.reporting.charts import equity_and_drawdown
from xauusd_ict_ml.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/backtest.yaml"); args = parser.parse_args()
    configure_logging("logs/backtest.log")
    config = load_config(args.config, BacktestConfig)
    lower = pd.read_parquet(config.lower_timeframe_path) if config.lower_timeframe_path else None
    result = run_backtest(
        pd.read_parquet(config.market_data_path), pd.read_parquet(config.setups_path), config, lower
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result.trades.to_csv(config.output_dir / "trades.csv", index=False)
    result.equity.to_csv(config.output_dir / "equity.csv", index=False)
    (config.output_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    for name, table in result.grouped.items():
        table.to_csv(config.output_dir / f"performance_by_{name}.csv", index=False)
    equity_and_drawdown(result.equity, config.output_dir)


if __name__ == "__main__": main()
