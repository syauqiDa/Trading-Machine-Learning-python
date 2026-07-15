"""Structure breaks based only on confirmed swings."""

from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.ict.swings import confirmed_swing_series


def detect_market_structure(
    frame: pd.DataFrame, swings: pd.DataFrame, atr: pd.Series | None = None, break_buffer_atr: float = 0.0
) -> pd.DataFrame:
    """Classify BOS/MSS/CHOCH when closes cross confirmed swing levels.

    A same-direction break is BOS. A break opposite the active state is marked
    MSS and CHOCH; these are synonymous event labels in many discretionary
    descriptions, so both boolean columns are retained explicitly.
    """
    attached = frame.copy().merge(confirmed_swing_series(frame, swings), on="timestamp", how="left")
    atr_values = atr.reindex(frame.index).fillna(0.0) if atr is not None else pd.Series(0.0, index=frame.index)
    direction = 0
    last_broken_high: float | None = None
    last_broken_low: float | None = None
    records: list[dict[str, object]] = []
    for position, row in attached.iterrows():
        buffer = float(atr_values.iloc[position]) * break_buffer_atr
        high_level = row["confirmed_swing_high"]
        low_level = row["confirmed_swing_low"]
        event = "none"
        bos = mss = choch = False
        if pd.notna(high_level) and row["close"] > high_level + buffer and high_level != last_broken_high:
            mss = direction < 0
            choch = mss
            bos = not mss
            event = "bullish_mss" if mss else "bullish_bos"
            direction = 1
            last_broken_high = float(high_level)
        elif pd.notna(low_level) and row["close"] < low_level - buffer and low_level != last_broken_low:
            mss = direction > 0
            choch = mss
            bos = not mss
            event = "bearish_mss" if mss else "bearish_bos"
            direction = -1
            last_broken_low = float(low_level)
        records.append(
            {
                "timestamp": row["timestamp"], "structure_direction": direction,
                "structure_event": event, "bos": bos, "mss": mss, "choch": choch,
                "protected_high": high_level if direction < 0 else pd.NA,
                "protected_low": low_level if direction > 0 else pd.NA,
            }
        )
    return pd.DataFrame(records)

