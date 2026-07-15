"""Flatten setup-engine metadata into numerical model features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ict_setup_features(setups: pd.DataFrame) -> pd.DataFrame:
    """Return only values computed and recorded at the setup timestamp."""
    output = pd.DataFrame({"setup_id": setups["setup_id"]})
    output["direction"] = setups["direction"].map({"bullish": 1, "bearish": -1})
    for column in [
        "sweep_penetration", "sweep_recovery_strength", "fvg_size_atr", "proposed_rr",
        "liquidity_level", "fvg_lower", "fvg_upper",
    ]:
        output[column] = pd.to_numeric(setups.get(column, np.nan), errors="coerce")
    output["distance_to_stop"] = (setups["entry_price"] - setups["stop_price"]).abs()
    output["distance_to_target"] = (setups["target_price"] - setups["entry_price"]).abs()
    output["session"] = setups["session"].astype(str)
    output["market_regime"] = setups["market_regime"].astype(str)
    return output

