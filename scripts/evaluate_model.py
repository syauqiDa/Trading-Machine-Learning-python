#!/usr/bin/env python3
"""Evaluate a saved model on an explicitly supplied chronological slice."""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.config import ModelConfig, load_config
from xauusd_ict_ml.ml.evaluation import evaluate_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("--config", default="config/model.yaml"); parser.add_argument("--output", default="reports/model_evaluation.json"); args = parser.parse_args()
    config = load_config(args.config, ModelConfig); data = pd.read_parquet(config.dataset_path); artifact = joblib.load(args.model)
    test = data.iloc[int(len(data) * .8):]; probability = artifact.pipeline.predict_proba(test[config.features])[:, 1]
    metrics = evaluate_probabilities(test["label"], probability, artifact.threshold)
    from pathlib import Path
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
