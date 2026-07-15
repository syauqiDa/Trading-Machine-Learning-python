#!/usr/bin/env python3
"""Build the candidate-level ML dataset."""

from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.config import ModelConfig, load_config
from xauusd_ict_ml.ml.dataset import build_ml_dataset
from xauusd_ict_ml.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/model.yaml"); parser.add_argument("--data", default="data/processed/xauusd_15m.parquet"); args = parser.parse_args()
    configure_logging("logs/build_dataset.log")
    config = load_config(args.config, ModelConfig)
    dataset = build_ml_dataset(pd.read_parquet(args.data), pd.read_parquet(config.setups_path), config.max_holding_bars, config.neutral_policy)
    config.dataset_path.parent.mkdir(parents=True, exist_ok=True); dataset.to_parquet(config.dataset_path, index=False)


if __name__ == "__main__": main()
