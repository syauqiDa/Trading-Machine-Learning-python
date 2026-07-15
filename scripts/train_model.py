#!/usr/bin/env python3
"""Train baseline models with chronological train/validation slices."""

from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.inspection import permutation_importance

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.config import ModelConfig, load_config
from xauusd_ict_ml.ml.training import save_trained_model, train_baselines
from xauusd_ict_ml.reporting.charts import permutation_importance_chart
from xauusd_ict_ml.utils.logging import configure_logging
from xauusd_ict_ml.utils.reproducibility import create_run_directory, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config/model.yaml"); args = parser.parse_args()
    configure_logging("logs/train_model.log")
    config = load_config(args.config, ModelConfig); data = pd.read_parquet(config.dataset_path)
    first, second = max(1, int(len(data) * .6)), max(2, int(len(data) * .8))
    run_dir = create_run_directory(config.output_dir, "models")
    artifacts = train_baselines(
        data.iloc[:first], data.iloc[first:second], config.features, config.models,
        config.probability_thresholds, config.random_seed, config.calibration_method,
    )
    summary = {}
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    validation = data.iloc[first:second]
    for name, artifact in artifacts.items():
        save_trained_model(artifact, str(run_dir / f"{name}.joblib")); summary[name] = artifact.validation_metrics
        importance = permutation_importance(
            artifact.pipeline, validation[config.features], validation["label"],
            scoring="neg_brier_score", n_repeats=10, random_state=config.random_seed,
        )
        importance_table = pd.DataFrame(
            {"feature": config.features, "importance_mean": importance.importances_mean,
             "importance_std": importance.importances_std}
        )
        importance_table.to_csv(config.reports_dir / f"{name}_permutation_importance.csv", index=False)
        permutation_importance_chart(
            importance_table, config.reports_dir / f"{name}_permutation_importance.png"
        )
    (run_dir / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(run_dir / "manifest.json", rows=len(data), feature_list=config.features, train_end=first, validation_end=second)


if __name__ == "__main__": main()
