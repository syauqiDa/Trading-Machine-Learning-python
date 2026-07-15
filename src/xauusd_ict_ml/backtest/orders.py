"""Typed order and trade records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass(frozen=True)
class Order:
    setup_id: int
    timestamp: pd.Timestamp
    side: Side
    order_type: OrderType
    requested_price: float
    stop_price: float
    target_price: float
    quantity: float


@dataclass
class Position:
    order: Order
    entry_timestamp: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    quantity: float
    bars_held: int = 0
    partial_realized: float = 0.0


@dataclass(frozen=True)
class Trade:
    setup_id: int
    direction: str
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    quantity: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_fraction: float
    result: str
    bars_held: int
    ambiguous: bool
    session: str
    market_regime: str
    proposed_rr: float

