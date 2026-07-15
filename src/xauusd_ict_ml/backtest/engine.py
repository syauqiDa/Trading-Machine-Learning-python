"""Sequential event-driven setup backtester."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from xauusd_ict_ml.backtest.execution import entry_fill, exit_decision, mark_exit
from xauusd_ict_ml.backtest.metrics import calculate_metrics, grouped_performance
from xauusd_ict_ml.backtest.orders import OrderType, Side, Trade
from xauusd_ict_ml.backtest.portfolio import Portfolio
from xauusd_ict_ml.backtest.risk import daily_loss_limit_reached, position_size, within_exposure_limit
from xauusd_ict_ml.config import BacktestConfig


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict[str, float | int]
    grouped: dict[str, pd.DataFrame]


@dataclass
class _ActiveTrade:
    setup: pd.Series
    side: Side
    entry_index: int
    entry_timestamp: pd.Timestamp
    entry_price: float
    stop_price: float
    active_stop: float
    target_price: float
    original_quantity: float
    remaining_quantity: float
    partial_realized: float = 0.0
    partial_done: bool = False


@dataclass
class _PendingOrder:
    setup: pd.Series
    created_index: int
    expires_index: int


def run_backtest(
    frame: pd.DataFrame, setups: pd.DataFrame, config: BacktestConfig,
    lower_timeframe: pd.DataFrame | None = None,
) -> BacktestResult:
    """Process pending orders, positions, and exits sequentially by market bar."""
    market = frame.sort_values("timestamp").reset_index(drop=True)
    candidates = setups.sort_values("entry_timestamp").reset_index(drop=True)
    portfolio = Portfolio(config.initial_capital)
    scheduled: dict[int, list[pd.Series]] = {}
    for _, setup in candidates.iterrows():
        scheduled.setdefault(int(setup["entry_index"]), []).append(setup)
    pending: list[_PendingOrder] = []
    active: list[_ActiveTrade] = []
    current_day = None
    day_start_equity = config.initial_capital
    order_type = OrderType(config.order_type)
    for index, bar in market.iterrows():
        day = pd.Timestamp(bar["timestamp"]).date()
        if day != current_day:
            current_day, day_start_equity = day, portfolio.cash
        survivors: list[_ActiveTrade] = []
        for position in active:
            decision = exit_decision(
                position.side, position.active_stop, position.target_price, bar,
                config.spread, config.slippage, config.intrabar_rule,
            )
            if decision is not None and decision.ambiguous and lower_timeframe is not None:
                decision = _resolve_with_lower_timeframe(
                    lower_timeframe, market, index, position, config
                ) or decision
            time_exit = index - position.entry_index >= config.max_holding_bars
            session_exit = bool(
                config.session_exit and "session" in bar
                and str(bar["session"]) != str(position.setup.get("session", "unknown"))
            )
            if decision is not None:
                _book_close(portfolio, position, bar, index, decision.price, decision.reason, decision.ambiguous, config)
                continue
            if time_exit or session_exit:
                reason = "session_exit" if session_exit else "time_exit"
                _book_close(portfolio, position, bar, index, mark_exit(position.side, bar, config.spread, config.slippage), reason, False, config)
                continue
            initial_risk = abs(position.entry_price - position.stop_price)
            favorable = float(bar["high"] - position.entry_price) if position.side is Side.LONG else float(position.entry_price - bar["low"])
            if config.partial_take_profit_fraction and not position.partial_done and favorable >= config.partial_take_profit_at_r * initial_risk:
                partial_quantity = position.original_quantity * config.partial_take_profit_fraction
                partial_level = position.entry_price + (1 if position.side is Side.LONG else -1) * config.partial_take_profit_at_r * initial_risk
                partial_price = partial_level - config.slippage if position.side is Side.LONG else partial_level + config.slippage
                multiplier = 1 if position.side is Side.LONG else -1
                position.partial_realized += multiplier * (partial_price - position.entry_price) * partial_quantity
                position.remaining_quantity -= partial_quantity
                position.partial_done = True
            # New break-even/trailing levels apply from the next bar because the
            # path within this candle is unknown.
            if config.break_even_at_r and favorable >= config.break_even_at_r * initial_risk:
                position.active_stop = max(position.active_stop, position.entry_price) if position.side is Side.LONG else min(position.active_stop, position.entry_price)
            if config.trailing_stop_atr and "atr" in bar and pd.notna(bar["atr"]):
                trail = float(bar["atr"]) * config.trailing_stop_atr
                position.active_stop = max(position.active_stop, float(bar["high"]) - trail) if position.side is Side.LONG else min(position.active_stop, float(bar["low"]) + trail)
            survivors.append(position)
        active = survivors

        for setup in scheduled.get(index, []):
            pending.append(_PendingOrder(setup, index, index + config.order_expiry_bars))
        retained: list[_PendingOrder] = []
        for order in pending:
            if index > order.expires_index:
                continue
            setup = order.setup
            side = Side.LONG if setup["direction"] == "bullish" else Side.SHORT
            fill = entry_fill(side, order_type, float(setup["entry_price"]), bar, config.spread, config.slippage)
            if fill is None:
                retained.append(order)
                continue
            if len(active) >= config.max_simultaneous_positions or daily_loss_limit_reached(day_start_equity, portfolio.cash, config.daily_loss_limit_fraction):
                continue
            quantity = position_size(
                portfolio.cash, fill, float(setup["stop_price"]), config.risk_mode,
                config.risk_per_trade, config.fixed_position_size,
            )
            exposure = sum(position.remaining_quantity for position in active)
            if not within_exposure_limit(quantity, exposure, config.max_total_exposure):
                quantity = max(0.0, config.max_total_exposure - exposure)
            if quantity <= 0:
                continue
            stop, target = float(setup["stop_price"]), float(setup["target_price"])
            active.append(
                _ActiveTrade(
                    setup=setup, side=side, entry_index=index, entry_timestamp=bar["timestamp"],
                    entry_price=fill, stop_price=stop, active_stop=stop, target_price=target,
                    original_quantity=quantity, remaining_quantity=quantity,
                )
            )
        pending = retained
    if len(market):
        final_index = len(market) - 1
        final_bar = market.iloc[final_index]
        for position in active:
            _book_close(
                portfolio, position, final_bar, final_index,
                mark_exit(position.side, final_bar, config.spread, config.slippage),
                "end_of_data", False, config,
            )
    trades = portfolio.trade_frame()
    return BacktestResult(
        trades=trades, equity=portfolio.equity_frame(),
        metrics=calculate_metrics(trades, config.initial_capital), grouped=grouped_performance(trades),
    )


def _book_close(
    portfolio: Portfolio, position: _ActiveTrade, bar: pd.Series, exit_index: int,
    exit_price: float, result: str, ambiguous: bool, config: BacktestConfig,
) -> None:
    multiplier = 1 if position.side is Side.LONG else -1
    gross_pnl = position.partial_realized + multiplier * (exit_price - position.entry_price) * position.remaining_quantity
    holding_days = max(0, (pd.Timestamp(bar["timestamp"]) - pd.Timestamp(position.entry_timestamp)).days)
    costs = 2 * config.commission_per_unit * position.original_quantity + holding_days * config.overnight_financing_per_unit * position.original_quantity
    net_pnl = gross_pnl - costs
    trade = Trade(
        setup_id=int(position.setup["setup_id"]), direction=position.side.value,
        entry_timestamp=position.entry_timestamp, exit_timestamp=bar["timestamp"],
        entry_price=position.entry_price, exit_price=exit_price, stop_price=position.stop_price,
        target_price=position.target_price, quantity=position.original_quantity,
        gross_pnl=gross_pnl, costs=costs, net_pnl=net_pnl,
        return_fraction=net_pnl / portfolio.cash if portfolio.cash else 0.0,
        result=result, bars_held=exit_index - position.entry_index, ambiguous=ambiguous,
        session=str(position.setup.get("session", "unknown")),
        market_regime=str(position.setup.get("market_regime", "unknown")),
        proposed_rr=float(position.setup.get("proposed_rr", 0.0)),
    )
    portfolio.book(trade)


def _resolve_with_lower_timeframe(
    lower: pd.DataFrame, market: pd.DataFrame, bar_index: int,
    position: _ActiveTrade, config: BacktestConfig,
):
    """Use ordered sub-bars inside an ambiguous parent bar when supplied."""
    current_time = pd.Timestamp(market.at[bar_index, "timestamp"])
    previous_time = pd.Timestamp(market.at[bar_index - 1, "timestamp"]) if bar_index else position.entry_timestamp
    sub_bars = lower[
        (pd.to_datetime(lower["timestamp"], utc=True) > previous_time)
        & (pd.to_datetime(lower["timestamp"], utc=True) <= current_time)
    ].sort_values("timestamp")
    for _, sub_bar in sub_bars.iterrows():
        decision = exit_decision(
            position.side, position.active_stop, position.target_price, sub_bar,
            config.spread, config.slippage, config.intrabar_rule,
        )
        if decision is not None:
            return decision
    return None
