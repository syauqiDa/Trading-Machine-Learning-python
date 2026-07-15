"""Position sizing and portfolio risk limits."""

from __future__ import annotations

import math


def position_size(
    equity: float, entry: float, stop: float, mode: str, risk_fraction: float,
    fixed_size: float, atr: float | None = None,
) -> float:
    """Size positions by fixed fractional loss, fixed units, or ATR-scaled risk."""
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        raise ValueError("Entry and stop must differ")
    if mode == "fixed_size":
        return fixed_size
    base = equity * risk_fraction / risk_per_unit
    if mode == "volatility_adjusted":
        if atr is None or atr <= 0:
            raise ValueError("Positive ATR required for volatility-adjusted sizing")
        base *= min(2.0, max(0.25, risk_per_unit / atr))
    return max(0.0, math.floor(base * 1000) / 1000)


def daily_loss_limit_reached(day_start_equity: float, equity: float, fraction: float) -> bool:
    """Block new risk once realized equity breaches the configured daily loss."""
    return equity <= day_start_equity * (1 - fraction)


def within_exposure_limit(quantity: float, current_exposure: float, maximum: float) -> bool:
    """Apply a simple absolute-unit exposure cap."""
    return current_exposure + abs(quantity) <= maximum

