"""Minimal realized-equity ledger."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from xauusd_ict_ml.backtest.orders import Trade


@dataclass
class Portfolio:
    initial_capital: float
    cash: float = field(init=False)
    trades: list[Trade] = field(default_factory=list)
    equity_points: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital

    def book(self, trade: Trade) -> None:
        self.cash += trade.net_pnl
        self.trades.append(trade)
        self.equity_points.append({"timestamp": trade.exit_timestamp, "equity": self.cash})

    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame([trade.__dict__ for trade in self.trades])

    def equity_frame(self) -> pd.DataFrame:
        opening = pd.DataFrame([{"timestamp": pd.NaT, "equity": self.initial_capital}])
        return pd.concat([opening, pd.DataFrame(self.equity_points)], ignore_index=True)

