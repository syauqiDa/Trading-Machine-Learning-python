"""Simple classification baselines and optional gradient boosting."""

from __future__ import annotations

import importlib.util
import logging

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


def make_model(name: str, random_seed: int):
    """Create a deliberately limited-complexity classifier."""
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=random_seed, n_jobs=-1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=random_seed)
    if name == "xgboost":
        if importlib.util.find_spec("xgboost") is None:
            LOGGER.warning("xgboost is not installed; skipping optional model")
            return None
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError, ValueError) as error:
            # The macOS wheel can be present while its optional OpenMP runtime
            # is not. Treat native-loader failures like an unavailable backend.
            summary = next((line for line in str(error).splitlines() if line.strip()), type(error).__name__)
            LOGGER.warning("xgboost is installed but unavailable; skipping optional model: %s", summary)
            return None

        return XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=random_seed,
        )
    raise ValueError(f"Unknown model: {name}")
