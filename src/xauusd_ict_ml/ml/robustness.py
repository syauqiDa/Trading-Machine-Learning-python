"""Overfitting controls and distribution-free robustness summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_shuffle_baseline(labels: pd.Series, probabilities: pd.Series, seed: int = 42, repetitions: int = 500) -> pd.DataFrame:
    """Compare the observed probability/outcome correlation with shuffled labels."""
    y = labels.to_numpy(dtype=float)
    p = probabilities.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    observed = float(np.corrcoef(y, p)[0, 1]) if len(np.unique(y)) > 1 else np.nan
    shuffled = [float(np.corrcoef(rng.permutation(y), p)[0, 1]) for _ in range(repetitions)]
    return pd.DataFrame({"observed_correlation": [observed], "shuffle_mean": [np.nanmean(shuffled)], "shuffle_95_percentile": [np.nanquantile(shuffled, .95)]})


def trade_order_monte_carlo(net_pnl: pd.Series, initial_capital: float, seed: int = 42, repetitions: int = 1000) -> pd.DataFrame:
    """Reshuffle trade order to estimate path-dependent drawdown dispersion."""
    pnl = net_pnl.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    records = []
    for repetition in range(repetitions):
        equity = initial_capital + np.cumsum(rng.permutation(pnl))
        peaks = np.maximum.accumulate(equity)
        records.append({"repetition": repetition, "net_profit": float(equity[-1] - initial_capital) if len(equity) else 0.0, "maximum_drawdown": float(np.min(equity / peaks - 1)) if len(equity) else 0.0})
    return pd.DataFrame(records)


def cost_sensitivity(trades: pd.DataFrame, cost_multipliers: list[float]) -> pd.DataFrame:
    """Reprice already-recorded gross PnL under transparent cost multiples."""
    records = []
    for multiplier in cost_multipliers:
        net = trades["gross_pnl"] - trades["costs"] * multiplier
        records.append({"cost_multiplier": multiplier, "net_profit": float(net.sum()), "expectancy": float(net.mean()) if len(net) else 0.0, "win_rate": float((net > 0).mean()) if len(net) else 0.0})
    return pd.DataFrame(records)


def concentration_warnings(trades: pd.DataFrame) -> list[str]:
    """Flag narrow profit concentration and statistically fragile samples."""
    warnings: list[str] = []
    if len(trades) < 30:
        warnings.append("Fewer than 30 trades; estimates are statistically fragile.")
    if not trades.empty and trades["net_pnl"].sum() > 0:
        top_share = trades.nlargest(max(1, len(trades) // 20), "net_pnl")["net_pnl"].sum() / trades["net_pnl"].sum()
        if top_share > .5:
            warnings.append("More than half of net profit comes from the top 5% of trades.")
    return warnings

