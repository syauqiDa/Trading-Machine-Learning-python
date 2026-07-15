"""Tabular robustness and strategy comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def probability_bucket_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed outcomes by predicted-probability bucket."""
    if predictions.empty:
        return pd.DataFrame()
    working = predictions.copy()
    working["probability_bucket"] = pd.cut(
        working["probability"], bins=[0, .4, .5, .6, .7, .8, 1], include_lowest=True
    )
    return working.groupby("probability_bucket", observed=True)["actual"].agg(["count", "mean"]).reset_index()


def strategy_comparison(setups: pd.DataFrame, predictions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Compare rule, ML, and same-count random filters on label expectancy."""
    if predictions.empty:
        return pd.DataFrame()
    merged = predictions.merge(setups[["setup_id"]], on="setup_id", how="inner")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model, group in merged.groupby("model"):
        taken = group[group["take"]]
        random_ids = rng.choice(group.index, size=len(taken), replace=False) if len(taken) else []
        random_group = group.loc[random_ids] if len(taken) else group.iloc[:0]
        for strategy, values in [("rule_based", group), (f"{model}_filtered", taken), (f"{model}_random_control", random_group)]:
            rows.append({"strategy": strategy, "trade_count": len(values), "observed_win_rate": values["actual"].mean() if len(values) else np.nan})
    return pd.DataFrame(rows)


def bootstrap_mean_interval(values: pd.Series, seed: int = 42, samples: int = 1000) -> tuple[float, float]:
    """Return a nonparametric 95% interval for a mean."""
    data = values.dropna().to_numpy(dtype=float)
    if not len(data):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(samples, len(data)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

