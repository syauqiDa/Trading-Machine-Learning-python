#!/usr/bin/env python3
"""Generate a consolidated HTML report from saved run outputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.backtest.engine import run_backtest
from xauusd_ict_ml.config import BacktestConfig, load_config
from xauusd_ict_ml.ml.robustness import (
    concentration_warnings, cost_sensitivity, label_shuffle_baseline, trade_order_monte_carlo,
)
from xauusd_ict_ml.reporting.charts import line_chart, outcome_distribution
from xauusd_ict_ml.reporting.report import generate_html_report
from xauusd_ict_ml.reporting.tables import probability_bucket_table, strategy_comparison


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--backtest-dir", default="reports/latest_backtest"); parser.add_argument("--ml-dir", default="reports/latest_ml"); parser.add_argument("--output", default="reports/final_report.html"); args = parser.parse_args()
    backtest = Path(args.backtest_dir); ml = Path(args.ml_dir)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((backtest / "metrics.json").read_text()) if (backtest / "metrics.json").exists() else {}
    tables: dict[str, pd.DataFrame] = {}
    for path in [*backtest.glob("*.csv"), ml / "walk_forward_metrics.csv", ml / "probability_buckets.csv"]:
        if path.exists(): tables[path.stem] = pd.read_csv(path)
    trades = tables.get("trades", pd.DataFrame())
    predictions = pd.read_csv(ml / "walk_forward_predictions.csv") if (ml / "walk_forward_predictions.csv").exists() else pd.DataFrame()
    if not predictions.empty:
        tables["probability_buckets"] = probability_bucket_table(predictions)
        setups = pd.read_parquet("data/processed/ml_dataset.parquet")
        tables["strategy_comparison"] = strategy_comparison(setups, predictions)
        tables["strategy_comparison"].to_csv(ml / "strategy_comparison.csv", index=False)
        shuffle_rows = []
        for model, group in predictions.groupby("model"):
            shuffled = label_shuffle_baseline(group["actual"], group["probability"], seed=42)
            shuffled.insert(0, "model", model); shuffle_rows.append(shuffled)
        tables["label_shuffle_control"] = pd.concat(shuffle_rows, ignore_index=True)
        tables["label_shuffle_control"].to_csv(ml / "label_shuffle_control.csv", index=False)
        market = pd.read_parquet("data/processed/xauusd_15m.parquet")
        setup_events = pd.read_parquet("data/processed/xauusd_setups.parquet")
        backtest_config = load_config("config/backtest.yaml", BacktestConfig)
        comparison_rows = []
        rng = np.random.default_rng(backtest_config.random_seed)
        for model, group in predictions.groupby("model"):
            evaluated = setup_events[setup_events["setup_id"].isin(group["setup_id"])].copy()
            filtered_ids = group.loc[group["take"].astype(bool), "setup_id"]
            filtered = evaluated[evaluated["setup_id"].isin(filtered_ids)]
            random_count = min(len(filtered), len(evaluated))
            random_ids = rng.choice(evaluated["setup_id"], size=random_count, replace=False) if random_count else []
            variants = {
                "rule_based": evaluated,
                f"{model}_filtered": filtered,
                f"{model}_random_same_count": evaluated[evaluated["setup_id"].isin(random_ids)],
            }
            for strategy, candidates in variants.items():
                result = run_backtest(market, candidates, backtest_config)
                comparison_rows.append(
                    {"model_group": model, "strategy": strategy,
                     **{key: result.metrics.get(key) for key in ["trade_count", "net_profit", "expectancy", "profit_factor", "maximum_drawdown", "sharpe_ratio", "gross_pnl_before_costs", "total_costs"]}}
                )
        tables["trading_strategy_comparison"] = pd.DataFrame(comparison_rows)
        tables["trading_strategy_comparison"].to_csv(ml / "trading_strategy_comparison.csv", index=False)
    if not trades.empty:
        tables["cost_sensitivity"] = cost_sensitivity(trades, [.5, 1, 1.5, 2, 3])
        tables["cost_sensitivity"].to_csv(backtest / "cost_sensitivity.csv", index=False)
        monte_carlo = trade_order_monte_carlo(trades["net_pnl"], 100000, repetitions=500)
        tables["monte_carlo_summary"] = monte_carlo[["net_profit", "maximum_drawdown"]].describe().reset_index()
        tables["monte_carlo_summary"].to_csv(backtest / "monte_carlo_summary.csv", index=False)
        outcome_distribution(trades, backtest / "trade_outcomes.png")
        line_chart(tables["cost_sensitivity"], "cost_multiplier", "net_profit", "Cost sensitivity", backtest / "cost_sensitivity.png")
        base_config = load_config("config/backtest.yaml", BacktestConfig)
        base_market = pd.read_parquet(base_config.market_data_path)
        base_setups = pd.read_parquet(base_config.setups_path)
        sensitivity_rows = []
        for spread_multiplier, slippage_multiplier in [(0.5, 1), (1, 1), (2, 1), (3, 1), (1, 2), (1, 3)]:
            stressed = base_market.drop(columns=[column for column in ["bid", "ask", "spread"] if column in base_market]).copy()
            stressed["spread"] = base_config.spread * spread_multiplier
            stressed_config = base_config.model_copy(update={"spread": base_config.spread * spread_multiplier, "slippage": base_config.slippage * slippage_multiplier})
            stressed_result = run_backtest(stressed, base_setups, stressed_config)
            sensitivity_rows.append({"spread_multiplier": spread_multiplier, "slippage_multiplier": slippage_multiplier, "net_profit": stressed_result.metrics.get("net_profit", 0), "expectancy": stressed_result.metrics.get("expectancy", 0), "maximum_drawdown": stressed_result.metrics.get("maximum_drawdown", 0)})
        tables["spread_slippage_sensitivity"] = pd.DataFrame(sensitivity_rows)
        tables["spread_slippage_sensitivity"].to_csv(backtest / "spread_slippage_sensitivity.csv", index=False)
    fold_table = tables.get("walk_forward_metrics", pd.DataFrame())
    if not fold_table.empty:
        line_chart(fold_table, "fold", "brier_score", "Walk-forward Brier score", ml / "walk_forward_folds.png")
    charts = []
    for chart in [*backtest.glob("*.png"), *ml.glob("*.png")]:
        copied = output.parent / f"{chart.parent.name}_{chart.name}"
        shutil.copy2(chart, copied); charts.append(copied)
    warnings = [
        "All bundled-run results use synthetic data and have no evidentiary value for real-market performance.",
        *concentration_warnings(trades),
    ]
    if metrics.get("total_costs", 0) and metrics.get("gross_pnl_before_costs", 0) <= metrics.get("total_costs", 0): warnings.append("Results collapse after modeled costs.")
    generate_html_report(output, "XAUUSD ICT-Style Research Report", metrics, tables, charts, warnings)


if __name__ == "__main__": main()
