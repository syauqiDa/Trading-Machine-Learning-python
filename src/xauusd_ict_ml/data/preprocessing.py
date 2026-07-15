"""Market-data preprocessing orchestration."""

from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.config import DataConfig
from xauusd_ict_ml.data.loader import load_market_data
from xauusd_ict_ml.data.validation import DataQualityReport, assert_valid_market_data, validate_market_data


def prepare_market_data(config: DataConfig) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load and validate data; missing OHLC intervals remain missing by default."""
    frame = load_market_data(config)
    report = validate_market_data(frame, config.expected_frequency, config.outlier_zscore)
    assert_valid_market_data(report)
    if config.allow_ohlc_forward_fill:
        frame[["open", "high", "low", "close"]] = frame[["open", "high", "low", "close"]].ffill()
    return frame, report
