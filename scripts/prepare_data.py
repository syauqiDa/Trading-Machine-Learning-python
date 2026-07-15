#!/usr/bin/env python3
"""Validate and prepare configured market data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.config import DataConfig, load_config
from xauusd_ict_ml.data.loader import save_market_data
from xauusd_ict_ml.data.preprocessing import prepare_market_data
from xauusd_ict_ml.data.resampling import resample_ohlc
from xauusd_ict_ml.data.sample import generate_synthetic_xauusd
from xauusd_ict_ml.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data.yaml")
    parser.add_argument("--generate-sample", action="store_true")
    args = parser.parse_args()
    configure_logging("logs/prepare_data.log")
    config = load_config(args.config, DataConfig)
    if args.generate_sample or not config.input_path.exists():
        sample = generate_synthetic_xauusd()
        config.input_path.parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(config.input_path, index=False)
        logging.getLogger(__name__).info("Generated synthetic sample at %s", config.input_path)
    frame, report = prepare_market_data(config)
    save_market_data(frame, config.output_path)
    for timeframe in config.timeframes:
        if timeframe == config.base_frequency:
            continue
        suffix = timeframe.replace("min", "m")
        destination = config.output_path.with_name(f"xauusd_{suffix}.parquet")
        save_market_data(resample_ohlc(frame, timeframe), destination)
    report_path = Path(config.output_path).with_suffix(".quality.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logging.getLogger(__name__).info("Prepared %d rows at %s", len(frame), config.output_path)


if __name__ == "__main__":
    main()
