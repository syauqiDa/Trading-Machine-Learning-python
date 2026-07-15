"""Chronological baseline model fitting."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline

from xauusd_ict_ml.ml.evaluation import choose_threshold, evaluate_probabilities
from xauusd_ict_ml.ml.calibration import calibrate_prefit
from xauusd_ict_ml.ml.models import make_model
from xauusd_ict_ml.ml.preprocessing import build_preprocessor


@dataclass(frozen=True)
class TrainedModel:
    name: str
    pipeline: object
    threshold: float
    validation_metrics: dict[str, object]


def train_baselines(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str], model_names: list[str],
    thresholds: list[float], random_seed: int,
    calibration_method: str = "sigmoid",
) -> dict[str, TrainedModel]:
    """Fit preprocessing and models on train, select thresholds on validation."""
    # pandas 3 uses a dedicated string dtype, so object-only detection is unsafe.
    categorical = [name for name in features if not is_numeric_dtype(train[name])]
    numeric = [name for name in features if name not in categorical]
    output: dict[str, TrainedModel] = {}
    for name in model_names:
        estimator = make_model(name, random_seed)
        if estimator is None:
            continue
        pipeline = Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", estimator)])
        pipeline.fit(train[features], train["label"])
        calibrated = calibrate_prefit(
            pipeline, validation[features], validation["label"], calibration_method
        )
        probability = calibrated.predict_proba(validation[features])[:, 1]
        threshold = choose_threshold(validation["label"], probability, thresholds)
        output[name] = TrainedModel(name, calibrated, threshold, evaluate_probabilities(validation["label"], probability, threshold))
    return output


def save_trained_model(model: TrainedModel, path: str) -> None:
    """Persist the full train-fitted preprocessing/model pipeline."""
    joblib.dump(model, path)
