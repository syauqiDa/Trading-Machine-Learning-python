"""Validated YAML configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ColumnConfig(StrictModel):
    timestamp: str = "timestamp"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str | None = "volume"
    bid: str | None = "bid"
    ask: str | None = "ask"
    spread: str | None = "spread"


class DataConfig(StrictModel):
    input_path: Path
    output_path: Path
    format: Literal["csv", "parquet"] = "csv"
    columns: ColumnConfig = Field(default_factory=ColumnConfig)
    source_timezone: str = "UTC"
    broker_timezone: str = "UTC"
    trading_timezone: str = "America/New_York"
    base_frequency: str = "15min"
    timeframes: list[str] = Field(default_factory=lambda: ["15min", "1h", "4h"])
    allow_ohlc_forward_fill: bool = False
    expected_frequency: str | None = None
    outlier_zscore: float = Field(default=8.0, gt=0)


class SwingConfig(StrictModel):
    left_bars: int = Field(default=3, ge=1)
    right_bars: int = Field(default=3, ge=1)


class StructureConfig(StrictModel):
    break_buffer_atr: float = Field(default=0.05, ge=0)


class LiquidityConfig(StrictModel):
    tolerance_mode: Literal["ticks", "percentage", "atr"] = "atr"
    tolerance_value: float = Field(default=0.1, gt=0)
    sweep_reclaim_bars: int = Field(default=3, ge=1)
    lookback_bars: int = Field(default=48, ge=2)


class DisplacementConfig(StrictModel):
    atr_period: int = Field(default=14, ge=2)
    min_range_atr: float = Field(default=1.2, gt=0)
    min_body_ratio: float = Field(default=0.6, ge=0, le=1)
    min_close_location: float = Field(default=0.7, ge=0, le=1)
    require_structure_break: bool = False
    min_consecutive_bars: int = Field(default=1, ge=1)
    min_volume_ratio: float = Field(default=1.0, ge=0)


class FairValueGapConfig(StrictModel):
    min_size_atr: float = Field(default=0.0, ge=0)
    max_age_bars: int = Field(default=48, ge=1)
    invalidate_on_close: bool = True


class PremiumDiscountConfig(StrictModel):
    range_lookback_bars: int = Field(default=96, ge=2)


class OrderBlockConfig(StrictModel):
    enabled: bool = True
    max_lookback_bars: int = Field(default=8, ge=1)
    max_age_bars: int = Field(default=64, ge=1)
    boundary: Literal["full_candle", "body"] = "full_candle"


class SessionWindow(StrictModel):
    start: str
    end: str
    enabled: bool = True


class SessionsConfig(StrictModel):
    timezone: str = "America/New_York"
    windows: dict[str, SessionWindow]


class SetupConfig(StrictModel):
    enabled_conditions: dict[str, bool]
    event_lookback_bars: int = Field(default=12, ge=1)
    max_spread: float = Field(default=1.0, ge=0)
    stop_buffer_atr: float = Field(default=0.1, ge=0)
    target_mode: Literal["risk_reward", "opposing_liquidity"] = "risk_reward"
    target_risk_reward: float = Field(default=2.0, gt=0)
    min_risk_reward: float = Field(default=1.0, gt=0)


class StrategyConfig(StrictModel):
    swing: SwingConfig
    structure: StructureConfig
    liquidity: LiquidityConfig
    displacement: DisplacementConfig
    fair_value_gap: FairValueGapConfig
    premium_discount: PremiumDiscountConfig
    order_block: OrderBlockConfig
    sessions: SessionsConfig
    setup: SetupConfig


class BacktestConfig(StrictModel):
    market_data_path: Path
    setups_path: Path
    lower_timeframe_path: Path | None = None
    initial_capital: float = Field(gt=0)
    order_type: Literal["market", "limit", "stop"] = "market"
    order_expiry_bars: int = Field(default=4, ge=0)
    risk_mode: Literal["fixed_fractional", "fixed_size", "volatility_adjusted"]
    risk_per_trade: float = Field(default=0.005, gt=0, le=1)
    fixed_position_size: float = Field(default=1.0, gt=0)
    spread: float = Field(default=0.0, ge=0)
    commission_per_unit: float = Field(default=0.0, ge=0)
    slippage: float = Field(default=0.0, ge=0)
    overnight_financing_per_unit: float = Field(default=0.0, ge=0)
    max_holding_bars: int = Field(default=48, ge=1)
    intrabar_rule: Literal["stop_first", "target_first", "nearest_open"] = "stop_first"
    max_simultaneous_positions: int = Field(default=1, ge=1)
    max_total_exposure: float = Field(default=5.0, gt=0)
    daily_loss_limit_fraction: float = Field(default=0.02, gt=0, le=1)
    partial_take_profit_fraction: float = Field(default=0.0, ge=0, lt=1)
    partial_take_profit_at_r: float = Field(default=1.0, gt=0)
    break_even_at_r: float = Field(default=0.0, ge=0)
    trailing_stop_atr: float = Field(default=0.0, ge=0)
    session_exit: bool = False
    random_seed: int = 42
    output_dir: Path


class WalkForwardConfig(StrictModel):
    mode: Literal["expanding", "rolling"] = "expanding"
    train_size: int = Field(ge=20)
    validation_size: int = Field(ge=1)
    test_size: int = Field(ge=1)
    step_size: int = Field(ge=1)
    purge_bars: int = Field(default=0, ge=0)
    embargo_bars: int = Field(default=0, ge=0)
    final_test_fraction: float = Field(default=0.2, gt=0, lt=0.5)


class ModelConfig(StrictModel):
    dataset_path: Path
    setups_path: Path
    output_dir: Path
    reports_dir: Path
    random_seed: int = 42
    neutral_policy: Literal["drop", "zero", "separate"] = "drop"
    max_holding_bars: int = Field(default=48, ge=1)
    features: list[str]
    models: list[str]
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid"
    probability_thresholds: list[float]
    walk_forward: WalkForwardConfig

    @model_validator(mode="after")
    def thresholds_are_probabilities(self) -> "ModelConfig":
        if not all(0 < value < 1 for value in self.probability_thresholds):
            raise ValueError("probability thresholds must be between zero and one")
        return self


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a mapping from YAML with explicit errors."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def load_config(path: str | Path, model: type[StrictModel]) -> StrictModel:
    """Load and validate a typed configuration model."""
    return model.model_validate(load_yaml(path))
