#!/usr/bin/env python3
"""Run purged chronological walk-forward validation."""

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.config import ModelConfig, load_config
from xauusd_ict_ml.ml.walk_forward import run_walk_forward
from xauusd_ict_ml.reporting.charts import probability_charts
from xauusd_ict_ml.reporting.tables import probability_bucket_table
from xauusd_ict_ml.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/model.yaml"); args = parser.parse_args()
    configure_logging("logs/walk_forward.log")
    config = load_config(args.config, ModelConfig); data = pd.read_parquet(config.dataset_path)
    metrics, predictions = run_walk_forward(data, config); config.reports_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.reports_dir / "walk_forward_metrics.csv", index=False); predictions.to_csv(config.reports_dir / "walk_forward_predictions.csv", index=False)
    probability_bucket_table(predictions).to_csv(config.reports_dir / "probability_buckets.csv", index=False)
    probability_charts(predictions, config.reports_dir)


if __name__ == "__main__": main()
