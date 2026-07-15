"""Static research chart generation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("logs/matplotlib-cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


def equity_and_drawdown(equity: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save equity and relative drawdown PNG charts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    clean = equity.dropna(subset=["timestamp"]).copy()
    if clean.empty:
        clean = pd.DataFrame({"timestamp": [pd.Timestamp.utcnow()], "equity": [equity["equity"].iloc[0]]})
    clean["drawdown"] = clean["equity"] / clean["equity"].cummax() - 1
    paths: list[Path] = []
    for column, title in [("equity", "Equity Curve"), ("drawdown", "Drawdown Curve")]:
        fig, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(clean["timestamp"], clean[column], color="#1f77b4" if column == "equity" else "#c0392b")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = destination / f"{column}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def probability_charts(predictions: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Save calibration, precision-recall, and confusion-matrix charts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if predictions.empty or predictions["actual"].nunique() < 2:
        return []
    y = predictions["actual"].astype(int)
    probability = predictions["probability"].astype(float)
    paths: list[Path] = []
    observed, forecast = calibration_curve(y, probability, n_bins=min(10, len(predictions) // 5), strategy="quantile")
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="grey")
    axis.plot(forecast, observed, marker="o")
    axis.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Probability calibration")
    path = destination / "calibration.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(path)
    fig, axis = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y, probability, ax=axis)
    path = destination / "precision_recall.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(path)
    predictions_binary = probability >= predictions["threshold"].astype(float)
    fig, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y, predictions_binary, ax=axis, colorbar=False)
    path = destination / "confusion_matrix.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig); paths.append(path)
    return paths


def feature_importance_chart(model, feature_names: list[str], output_path: str | Path) -> Path | None:
    """Save native model importance when the estimator exposes it."""
    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = np.abs(getattr(estimator, "coef_", np.array([]))).ravel()
    if len(values) != len(feature_names):
        return None
    order = np.argsort(values)[-20:]
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(np.asarray(feature_names)[order], np.asarray(values)[order])
    axis.set_title("Feature importance (model-specific)")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def line_chart(table: pd.DataFrame, x: str, y: str, title: str, output_path: str | Path) -> Path | None:
    """Save a compact line chart for sensitivity or fold summaries."""
    if table.empty or x not in table or y not in table:
        return None
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for name, group in table.groupby("model") if "model" in table else [(None, table)]:
        axis.plot(group[x].astype(str), group[y], marker="o", label=name)
    if "model" in table:
        axis.legend()
    axis.set(title=title, xlabel=x.replace("_", " ").title(), ylabel=y.replace("_", " ").title())
    axis.grid(alpha=.25)
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def outcome_distribution(trades: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Save the net-PnL outcome distribution."""
    if trades.empty:
        return None
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(trades["net_pnl"], bins=min(20, max(5, len(trades) // 2)), color="#4c78a8", edgecolor="white")
    axis.axvline(0, color="black", linewidth=1)
    axis.set(title="Trade outcome distribution", xlabel="Net PnL", ylabel="Trades")
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path


def permutation_importance_chart(table: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Save input-feature permutation importance with uncertainty bars."""
    if table.empty:
        return None
    selected = table.nlargest(min(20, len(table)), "importance_mean").sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(selected["feature"], selected["importance_mean"], xerr=selected["importance_std"], color="#4c78a8")
    axis.set_title("Validation permutation importance")
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return path
