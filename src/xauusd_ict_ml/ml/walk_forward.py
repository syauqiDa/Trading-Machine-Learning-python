"""Purged walk-forward model evaluation."""

from __future__ import annotations

import pandas as pd
import numpy as np

from xauusd_ict_ml.config import ModelConfig
from xauusd_ict_ml.ml.evaluation import evaluate_probabilities
from xauusd_ict_ml.ml.splits import walk_forward_splits
from xauusd_ict_ml.ml.training import train_baselines


def run_walk_forward(dataset: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain folds and reserve a final chronological block until the end."""
    wf = config.walk_forward
    development_end = int(len(dataset) * (1 - wf.final_test_fraction))
    development = dataset.iloc[:development_end].reset_index(drop=True)
    folds = walk_forward_splits(
        development, wf.train_size, wf.validation_size, wf.test_size, wf.step_size,
        wf.purge_bars, wf.embargo_bars, wf.mode,
    )
    metric_records: list[dict[str, object]] = []
    prediction_records: list[dict[str, object]] = []
    for fold_index, fold in enumerate(folds):
        trained = train_baselines(
            development.iloc[fold.train], development.iloc[fold.validation], config.features,
            config.models, config.probability_thresholds, config.random_seed + fold_index,
            config.calibration_method,
        )
        for name, artifact in trained.items():
            test = development.iloc[fold.test]
            probability = artifact.pipeline.predict_proba(test[config.features])[:, 1]
            metrics = evaluate_probabilities(test["label"], probability, artifact.threshold)
            metric_records.append(
                {"fold": fold_index, "model": name, "threshold": artifact.threshold,
                 "train_start": int(fold.train[0]), "train_end": int(fold.train[-1]),
                 "validation_start": int(fold.validation[0]), "validation_end": int(fold.validation[-1]),
                 "test_start": int(fold.test[0]), "test_end": int(fold.test[-1]), **metrics}
            )
            for setup_id, actual, value in zip(test["setup_id"], test["label"], probability):
                prediction_records.append(
                    {"fold": fold_index, "model": name, "setup_id": int(setup_id),
                     "actual": int(actual), "probability": float(value),
                     "threshold": artifact.threshold, "take": bool(value >= artifact.threshold)}
                )
    # The final block has not participated in fitting, calibration, threshold
    # choice, feature choice, or fold reporting. Fit once on development data,
    # with a purged trailing validation slice, then evaluate it exactly once.
    final_test = dataset.iloc[development_end:]
    validation_start = max(1, development_end - wf.validation_size)
    boundary_bar = int(dataset.iloc[validation_start]["label_start_index"])
    candidate_train = np.arange(validation_start)
    label_ends = dataset.iloc[candidate_train]["label_end_index"].to_numpy(dtype=int)
    label_starts = dataset.iloc[candidate_train]["label_start_index"].to_numpy(dtype=int)
    train_indices = candidate_train[
        (label_ends < boundary_bar) & (label_starts < boundary_bar - wf.purge_bars)
    ]
    validation_indices = np.arange(validation_start, development_end - wf.embargo_bars)
    if len(train_indices) and len(validation_indices) and len(final_test):
        final_models = train_baselines(
            dataset.iloc[train_indices], dataset.iloc[validation_indices], config.features,
            config.models, config.probability_thresholds, config.random_seed + len(folds),
            config.calibration_method,
        )
        for name, artifact in final_models.items():
            probability = artifact.pipeline.predict_proba(final_test[config.features])[:, 1]
            metrics = evaluate_probabilities(final_test["label"], probability, artifact.threshold)
            metric_records.append(
                {"fold": "final_untouched", "model": name, "threshold": artifact.threshold,
                 "train_start": int(train_indices[0]), "train_end": int(train_indices[-1]),
                 "validation_start": int(validation_indices[0]), "validation_end": int(validation_indices[-1]),
                 "test_start": development_end, "test_end": len(dataset) - 1, **metrics}
            )
            for setup_id, actual, value in zip(final_test["setup_id"], final_test["label"], probability):
                prediction_records.append(
                    {"fold": "final_untouched", "model": name, "setup_id": int(setup_id),
                     "actual": int(actual), "probability": float(value),
                     "threshold": artifact.threshold, "take": bool(value >= artifact.threshold)}
                )
    return pd.DataFrame(metric_records), pd.DataFrame(prediction_records)
