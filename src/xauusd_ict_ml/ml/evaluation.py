"""Classification, calibration, and threshold evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score,
)


def evaluate_probabilities(y_true, probabilities, threshold: float = 0.5) -> dict[str, object]:
    """Return discrimination and calibration metrics with a confusion matrix."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    unique = np.unique(y_true)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(unique) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probabilities)) if len(unique) > 1 else float("nan"),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.c_[1 - np.asarray(probabilities), probabilities], labels=[0, 1])),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def choose_threshold(y_true, probabilities, thresholds: list[float]) -> float:
    """Choose the validation threshold with best F1; never inspect test labels."""
    return max(thresholds, key=lambda value: f1_score(y_true, np.asarray(probabilities) >= value, zero_division=0))

