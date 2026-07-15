"""Bid/ask-aware fills and explicit intrabar ambiguity handling."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xauusd_ict_ml.backtest.orders import OrderType, Side


@dataclass(frozen=True)
class ExitDecision:
    reason: str
    price: float
    ambiguous: bool = False


def quote_prices(bar: pd.Series, configured_spread: float) -> tuple[float, float]:
    """Return bid/ask, synthesizing them symmetrically only when absent."""
    if "bid" in bar and "ask" in bar and pd.notna(bar["bid"]) and pd.notna(bar["ask"]):
        return float(bar["bid"]), float(bar["ask"])
    mid = float(bar["close"])
    spread = float(bar.get("spread", configured_spread))
    return mid - spread / 2, mid + spread / 2


def entry_fill(
    side: Side, order_type: OrderType, requested_price: float, bar: pd.Series,
    configured_spread: float, slippage: float,
) -> float | None:
    """Fill market, touched limit, or triggered stop orders on the correct quote side."""
    bid, ask = quote_prices(bar, configured_spread)
    if order_type is OrderType.MARKET:
        return ask + slippage if side is Side.LONG else bid - slippage
    if side is Side.LONG:
        touched = bar["low"] <= requested_price if order_type is OrderType.LIMIT else bar["high"] >= requested_price
        return requested_price + configured_spread / 2 + slippage if touched else None
    touched = bar["high"] >= requested_price if order_type is OrderType.LIMIT else bar["low"] <= requested_price
    return requested_price - configured_spread / 2 - slippage if touched else None


def exit_decision(
    side: Side, stop: float, target: float, bar: pd.Series, configured_spread: float,
    slippage: float, intrabar_rule: str = "stop_first",
) -> ExitDecision | None:
    """Resolve stop/target touches; stop-first is the conservative default."""
    spread = float(bar.get("spread", configured_spread))
    if side is Side.LONG:
        bid_low = float(bar["low"]) - spread / 2
        bid_high = float(bar["high"]) - spread / 2
        stop_hit, target_hit = bid_low <= stop, bid_high >= target
    else:
        ask_low = float(bar["low"]) + spread / 2
        ask_high = float(bar["high"]) + spread / 2
        stop_hit, target_hit = ask_high >= stop, ask_low <= target
    if stop_hit and target_hit:
        if intrabar_rule == "target_first":
            return ExitDecision("target", target - slippage if side is Side.LONG else target + slippage, True)
        if intrabar_rule == "nearest_open":
            chosen = "stop" if abs(float(bar["open"]) - stop) <= abs(float(bar["open"]) - target) else "target"
            price = stop if chosen == "stop" else target
            return ExitDecision(chosen, price - slippage if side is Side.LONG else price + slippage, True)
        return ExitDecision("stop", stop - slippage if side is Side.LONG else stop + slippage, True)
    if stop_hit:
        return ExitDecision("stop", stop - slippage if side is Side.LONG else stop + slippage)
    if target_hit:
        return ExitDecision("target", target - slippage if side is Side.LONG else target + slippage)
    return None


def mark_exit(side: Side, bar: pd.Series, configured_spread: float, slippage: float) -> float:
    """Close long at bid or short at ask."""
    bid, ask = quote_prices(bar, configured_spread)
    return bid - slippage if side is Side.LONG else ask + slippage

