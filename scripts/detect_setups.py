#!/usr/bin/env python3
"""Detect deterministic ICT-style setup candidates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.config import StrategyConfig, load_config
from xauusd_ict_ml.data.sessions import assign_sessions
from xauusd_ict_ml.ict.displacement import average_true_range, detect_displacement
from xauusd_ict_ml.ict.fair_value_gaps import detect_fair_value_gaps
from xauusd_ict_ml.ict.liquidity import (
    build_known_levels, detect_equal_levels, detect_liquidity_sweeps, tolerance_series,
)
from xauusd_ict_ml.ict.order_blocks import detect_order_blocks
from xauusd_ict_ml.ict.premium_discount import calculate_premium_discount
from xauusd_ict_ml.ict.setup_engine import generate_setups
from xauusd_ict_ml.ict.structure import detect_market_structure
from xauusd_ict_ml.ict.swings import detect_swings
from xauusd_ict_ml.utils.logging import configure_logging


def detect(data_path: str | Path, strategy_path: str | Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frame = pd.read_parquet(data_path)
    strategy = load_config(strategy_path, StrategyConfig)
    atr = average_true_range(frame, strategy.displacement.atr_period)
    swings = detect_swings(frame, strategy.swing.left_bars, strategy.swing.right_bars)
    structure = detect_market_structure(frame, swings, atr, strategy.structure.break_buffer_atr)
    displacement = detect_displacement(frame, strategy.displacement, structure)
    levels = build_known_levels(frame, swings, strategy.sessions)
    tolerance = tolerance_series(
        frame, strategy.liquidity.tolerance_mode, strategy.liquidity.tolerance_value, atr
    )
    equal_levels = detect_equal_levels(swings, tolerance)
    if not equal_levels.empty:
        equal_liquidity = equal_levels.assign(
            side=equal_levels["level_type"].map({"equal_highs": "buy", "equal_lows": "sell"})
        ).rename(columns={"known_time": "known_time"})
        levels = pd.concat(
            [levels, equal_liquidity[["known_time", "price", "level_type", "side"]]], ignore_index=True
        ).sort_values("known_time", ignore_index=True)
    sweeps = detect_liquidity_sweeps(frame, levels, strategy.liquidity.sweep_reclaim_bars)
    gaps = detect_fair_value_gaps(
        frame, atr, strategy.fair_value_gap.min_size_atr,
        strategy.fair_value_gap.max_age_bars, strategy.fair_value_gap.invalidate_on_close,
    )
    premium_discount = calculate_premium_discount(frame, strategy.premium_discount.range_lookback_bars)
    sessions = assign_sessions(frame["timestamp"], strategy.sessions)
    setups = generate_setups(frame, structure, displacement, sweeps, gaps, premium_discount, sessions, strategy)
    order_blocks = detect_order_blocks(
        frame, displacement, strategy.order_block.max_lookback_bars,
        strategy.order_block.max_age_bars, strategy.order_block.boundary,
    ) if strategy.order_block.enabled else pd.DataFrame()
    return setups, {"swings": swings, "structure": structure, "liquidity_levels": levels, "equal_levels": equal_levels, "sweeps": sweeps, "gaps": gaps, "order_blocks": order_blocks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/strategy.yaml")
    parser.add_argument("--data", default="data/processed/xauusd_15m.parquet")
    parser.add_argument("--output", default="data/processed/xauusd_setups.parquet")
    args = parser.parse_args()
    configure_logging("logs/detect_setups.log")
    setups, diagnostics = detect(args.data, args.config)
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    setups.to_parquet(destination, index=False)
    for name, table in diagnostics.items():
        table.to_parquet(destination.with_name(f"{destination.stem}_{name}.parquet"), index=False)
    logging.getLogger(__name__).info("Detected %d setup candidates", len(setups))


if __name__ == "__main__":
    main()
