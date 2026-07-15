"""Validation-fitted probability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbabilityCalibratedEstimator(ClassifierMixin, BaseEstimator):
    """Serializable wrapper around an already train-fitted estimator."""

    estimator: object
    calibrator: object
    method: str

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])

    def fit(self, features, labels=None) -> "ProbabilityCalibratedEstimator":
        """Satisfy estimator protocols without refitting frozen components."""
        return self

    def predict_proba(self, features) -> np.ndarray:
        raw = self.estimator.predict_proba(features)[:, 1]
        if self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(raw)
        calibrated = np.clip(calibrated, 0, 1)
        return np.c_[1 - calibrated, calibrated]

    def predict(self, features) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)


def calibrate_prefit(estimator, x_validation, y_validation, method: str = "sigmoid"):
    """Fit calibration only on later validation rows, never on training or test."""
    labels = np.asarray(y_validation)
    if len(np.unique(labels)) < 2:
        return estimator
    raw = estimator.predict_proba(x_validation)[:, 1]
    if method == "sigmoid":
        calibrator = LogisticRegression().fit(raw.reshape(-1, 1), labels)
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, labels)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return ProbabilityCalibratedEstimator(estimator, calibrator, method)
