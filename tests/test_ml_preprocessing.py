from __future__ import annotations

import pandas as pd
import builtins

from xauusd_ict_ml.ml.models import make_model
from xauusd_ict_ml.ml.training import train_baselines


def test_pandas_string_dtype_is_categorical():
    train = pd.DataFrame({"number": range(8), "session": pd.Series(["asia", "ny"] * 4, dtype="str"), "label": [0, 1] * 4})
    validation = train.copy()
    models = train_baselines(train, validation, ["number", "session"], ["logistic"], [.5], 42)
    assert "logistic" in models


def test_unloadable_optional_xgboost_is_skipped(monkeypatch):
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ValueError("native runtime unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    assert make_model("xgboost", 42) is None
