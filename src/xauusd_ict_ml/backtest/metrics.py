"""Trading performance metrics and grouped summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_metrics(trades: pd.DataFrame, initial_capital: float) -> dict[str, float | int]:
    """Calculate cost-aware descriptive metrics without profitability claims."""
    if trades.empty:
        return {"trade_count": 0, "net_profit": 0.0, "total_return": 0.0, "ambiguous_trades": 0}
    pnl = trades["net_pnl"].astype(float)
    gross = trades["gross_pnl"].astype(float)
    equity = initial_capital + pnl.cumsum()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    returns = pnl / initial_capital
    downside = returns[returns < 0]
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(pnl[pnl < 0].sum())
    standard_deviation = returns.std(ddof=1)
    sharpe = float(returns.mean() / standard_deviation * math.sqrt(252)) if standard_deviation and not np.isnan(standard_deviation) else 0.0
    downside_std = downside.std(ddof=1)
    sortino = float(returns.mean() / downside_std * math.sqrt(252)) if downside_std and not np.isnan(downside_std) else 0.0
    total_return = float(pnl.sum() / initial_capital)
    max_drawdown = float(drawdown.min())
    return {
        "total_return": total_return, "net_profit": float(pnl.sum()),
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": gross_profit / abs(gross_loss) if gross_loss else float("inf"),
        "maximum_drawdown": max_drawdown, "drawdown_duration": _drawdown_duration(drawdown),
        "sharpe_ratio": sharpe, "sortino_ratio": sortino,
        "calmar_ratio": total_return / abs(max_drawdown) if max_drawdown else 0.0,
        "expectancy": float(pnl.mean()), "win_rate": float((pnl > 0).mean()),
        "average_win": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else 0.0,
        "average_loss": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0,
        "average_risk_reward": float(trades["proposed_rr"].mean()),
        "trade_count": int(len(trades)), "exposure": float(trades["bars_held"].sum()),
        "consecutive_wins": _max_streak(pnl > 0), "consecutive_losses": _max_streak(pnl < 0),
        "value_at_risk_95": float(np.quantile(pnl, 0.05)),
        "gross_pnl_before_costs": float(gross.sum()), "total_costs": float(trades["costs"].sum()),
        "ambiguous_trades": int(trades["ambiguous"].sum()),
    }


def grouped_performance(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return year/month/session/weekday/direction/regime performance tables."""
    if trades.empty:
        return {}
    working = trades.copy()
    working["exit_timestamp"] = pd.to_datetime(working["exit_timestamp"], utc=True)
    working["year"] = working["exit_timestamp"].dt.year
    working["month"] = working["exit_timestamp"].dt.strftime("%Y-%m")
    working["weekday"] = working["exit_timestamp"].dt.day_name()
    output: dict[str, pd.DataFrame] = {}
    for column in ["year", "month", "session", "weekday", "direction", "market_regime"]:
        output[column] = working.groupby(column, dropna=False)["net_pnl"].agg(["count", "sum", "mean"]).reset_index()
    return output


def _max_streak(mask: pd.Series) -> int:
    groups = (~mask).cumsum()
    return int(mask.groupby(groups).sum().max()) if len(mask) else 0


def _drawdown_duration(drawdown: pd.Series) -> int:
    return _max_streak(drawdown < 0)
