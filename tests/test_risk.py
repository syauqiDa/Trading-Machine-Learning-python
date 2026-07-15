from __future__ import annotations

from xauusd_ict_ml.backtest.risk import daily_loss_limit_reached, position_size


def test_fixed_fractional_size_and_daily_limit():
    assert position_size(100_000, 100, 90, "fixed_fractional", .01, 1) == 100
    assert daily_loss_limit_reached(100_000, 97_999, .02)

