"""Ablatable rule engine with auditable condition explanations."""

from __future__ import annotations

import json

import pandas as pd

from xauusd_ict_ml.config import StrategyConfig


def generate_setups(
    frame: pd.DataFrame, structure: pd.DataFrame, displacement: pd.DataFrame,
    sweeps: pd.DataFrame, gaps: pd.DataFrame, premium_discount: pd.DataFrame,
    sessions: pd.Series, config: StrategyConfig,
) -> pd.DataFrame:
    """Generate mirrored bullish/bearish candidates using only known events."""
    records: list[dict[str, object]] = []
    lookback = config.setup.event_lookback_bars
    for index, bar in frame.iterrows():
        timestamp = bar["timestamp"]
        for direction, sign in [("bullish", 1), ("bearish", -1)]:
            recent_sweeps = sweeps[
                (sweeps["direction"] == direction)
                & (sweeps["sweep_timestamp"] <= timestamp)
                & (sweeps["sweep_timestamp"] >= frame.at[max(0, index - lookback), "timestamp"])
            ] if not sweeps.empty else sweeps
            recent_gaps = gaps[
                (gaps["direction"] == direction)
                & (gaps["formation_timestamp"] <= timestamp)
                & (gaps["formation_index"] >= index - lookback)
            ] if not gaps.empty else gaps
            latest_sweep = recent_sweeps.iloc[-1] if not recent_sweeps.empty else None
            latest_gap = recent_gaps.iloc[-1] if not recent_gaps.empty else None
            disp_flag = "bullish_displacement" if direction == "bullish" else "bearish_displacement"
            recent_disp = bool(displacement.loc[max(0, index - lookback):index, disp_flag].any())
            bias = int(structure.at[index, "structure_direction"]) in (0, sign)
            zone = premium_discount.at[index, "zone"]
            pd_ok = zone == ("discount" if direction == "bullish" else "premium")
            structure_ok = str(structure.at[index, "structure_event"]).startswith(direction)
            spread = float(bar.get("spread", 0.0))
            session = str(sessions.iloc[index])
            retracement = _inside_gap(float(bar["low"]), float(bar["high"]), latest_gap)
            conditions = {
                "bias": bias, "premium_discount": pd_ok,
                "liquidity_sweep": latest_sweep is not None, "structure_shift": structure_ok,
                "displacement": recent_disp, "fair_value_gap": latest_gap is not None,
                "retracement": retracement, "session": session != "off_session",
                "spread": spread <= config.setup.max_spread,
            }
            passes = all(value for name, value in conditions.items() if config.setup.enabled_conditions.get(name, False))
            if not passes:
                continue
            atr = float(displacement.at[index, "atr"]) if pd.notna(displacement.at[index, "atr"]) else float(bar["high"] - bar["low"])
            entry = float(bar.get("ask", bar["close"] + spread / 2)) if direction == "bullish" else float(bar.get("bid", bar["close"] - spread / 2))
            extreme = float(latest_sweep["extreme"]) if latest_sweep is not None else (float(bar["low"]) if direction == "bullish" else float(bar["high"]))
            stop = extreme - config.setup.stop_buffer_atr * atr if direction == "bullish" else extreme + config.setup.stop_buffer_atr * atr
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            target = entry + sign * config.setup.target_risk_reward * risk
            explanation = {
                "passed": [name for name, value in conditions.items() if value],
                "failed": [name for name, value in conditions.items() if not value],
                "enabled": config.setup.enabled_conditions,
                "values": {"atr": atr, "spread": spread, "zone": zone, "structure": structure.at[index, "structure_event"]},
            }
            records.append(
                {
                    "setup_id": len(records), "setup_timestamp": timestamp, "entry_timestamp": timestamp,
                    "entry_index": index, "direction": direction, "entry_price": entry,
                    "stop_price": stop, "target_price": target,
                    "proposed_rr": abs(target - entry) / risk, "session": session,
                    "market_regime": "trend" if structure.at[index, "structure_direction"] else "range",
                    "liquidity_level": float(latest_sweep["swept_level"]) if latest_sweep is not None else pd.NA,
                    "sweep_penetration": float(latest_sweep["penetration_distance"]) if latest_sweep is not None else 0.0,
                    "sweep_recovery_strength": float(latest_sweep["recovery_strength"]) if latest_sweep is not None else 0.0,
                    "fvg_id": int(latest_gap["gap_id"]) if latest_gap is not None else pd.NA,
                    "fvg_lower": float(latest_gap["lower_bound"]) if latest_gap is not None else pd.NA,
                    "fvg_upper": float(latest_gap["upper_bound"]) if latest_gap is not None else pd.NA,
                    "fvg_size_atr": float(latest_gap["size_atr"]) if latest_gap is not None and pd.notna(latest_gap["size_atr"]) else 0.0,
                    "explanation": json.dumps(explanation, default=str),
                }
            )
    return pd.DataFrame(records)


def _inside_gap(low: float, high: float, gap: pd.Series | None) -> bool:
    if gap is None:
        return False
    return low <= float(gap["upper_bound"]) and high >= float(gap["lower_bound"])
