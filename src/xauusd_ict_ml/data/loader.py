"""CSV and Parquet market-data loaders."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from xauusd_ict_ml.config import DataConfig
from xauusd_ict_ml.utils.time import normalize_timestamp

LOGGER = logging.getLogger(__name__)


def load_market_data(config: DataConfig) -> pd.DataFrame:
    """Load, standardize column names, normalize UTC, and sort market bars."""
    path = Path(config.input_path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path) if config.format == "csv" else pd.read_parquet(path)
    mapping = {
        source: canonical
        for canonical, source in config.columns.model_dump().items()
        if source is not None and source in frame.columns
    }
    frame = frame.rename(columns=mapping)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["timestamp"] = normalize_timestamp(frame["timestamp"], config.source_timezone)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    numeric = [c for c in ["open", "high", "low", "close", "volume", "bid", "ask", "spread"] if c in frame]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if "spread" not in frame and {"bid", "ask"}.issubset(frame.columns):
        frame["spread"] = frame["ask"] - frame["bid"]
    if not {"bid", "ask"}.issubset(frame.columns):
        LOGGER.warning("Bid/ask unavailable; execution will use configured synthetic spread")
    return frame


def save_market_data(frame: pd.DataFrame, path: str | Path) -> None:
    """Save by extension without altering temporal columns."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        frame.to_csv(destination, index=False)
    elif destination.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(destination, index=False)
    else:
        raise ValueError("Output extension must be .csv or .parquet")

