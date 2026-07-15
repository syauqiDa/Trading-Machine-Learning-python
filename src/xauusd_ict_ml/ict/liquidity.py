"""Known liquidity levels, equal highs/lows, and reclaim sweeps."""

from __future__ import annotations

import pandas as pd

from xauusd_ict_ml.config import SessionsConfig
from xauusd_ict_ml.data.sessions import completed_session_levels, previous_completed_levels


def tolerance_series(
    frame: pd.DataFrame, mode: str, value: float, atr: pd.Series | None = None, tick_size: float = 0.01
) -> pd.Series:
    """Calculate level-comparison tolerance from ticks, price percentage, or ATR."""
    if mode == "ticks":
        return pd.Series(value * tick_size, index=frame.index)
    if mode == "percentage":
        return frame["close"] * value
    if mode == "atr":
        if atr is None:
            raise ValueError("ATR is required for ATR-based liquidity tolerance")
        return atr * value
    raise ValueError(f"Unknown tolerance mode: {mode}")


def previous_day_week_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach prior completed UTC day and ISO-week levels."""
    daily = previous_completed_levels(frame, "1D").rename(
        columns={"previous_high": "previous_day_high", "previous_low": "previous_day_low"}
    )
    weekly = previous_completed_levels(frame, "W-MON").rename(
        columns={"previous_high": "previous_week_high", "previous_low": "previous_week_low"}
    )
    return daily.merge(weekly, on="timestamp", how="left")


def detect_equal_levels(swings: pd.DataFrame, tolerance: float | pd.Series) -> pd.DataFrame:
    """Create an equal-high/low level only when the second pivot is confirmed."""
    records: list[dict[str, object]] = []
    for kind in ("high", "low"):
        subset = swings.loc[swings["swing_type"] == kind].sort_values("confirmation_time")
        for (_, previous), (_, current) in zip(subset.iloc[:-1].iterrows(), subset.iloc[1:].iterrows()):
            current_tolerance = float(tolerance.iloc[int(current["confirmation_index"])]) if isinstance(tolerance, pd.Series) else tolerance
            if pd.notna(current_tolerance) and abs(float(current["price"]) - float(previous["price"])) <= current_tolerance:
                records.append(
                    {
                        "level_type": f"equal_{kind}s",
                        "price": (float(current["price"]) + float(previous["price"])) / 2,
                        "known_time": current["confirmation_time"],
                    }
                )
    return pd.DataFrame(records, columns=["level_type", "price", "known_time"])


def detect_liquidity_sweeps(
    frame: pd.DataFrame, levels: pd.DataFrame, reclaim_bars: int = 3
) -> pd.DataFrame:
    """Detect excursions through levels followed by a close back across them.

    ``levels`` needs ``known_time``, ``price``, and ``side`` where side is
    ``sell`` for levels below price and ``buy`` for levels above price.
    """
    records: list[dict[str, object]] = []
    ordered = frame.reset_index(drop=True)
    for _, level in levels.iterrows():
        eligible = ordered.index[ordered["timestamp"] >= level["known_time"]]
        for index in eligible:
            price = float(level["price"])
            if level["side"] == "sell" and ordered.at[index, "low"] < price:
                for reclaim in range(index, min(index + reclaim_bars, len(ordered) - 1) + 1):
                    if ordered.at[reclaim, "close"] > price:
                        penetration = price - float(ordered.loc[index:reclaim, "low"].min())
                        records.append(_sweep_record(ordered, index, reclaim, level, "bullish", penetration))
                        break
                break
            if level["side"] == "buy" and ordered.at[index, "high"] > price:
                for reclaim in range(index, min(index + reclaim_bars, len(ordered) - 1) + 1):
                    if ordered.at[reclaim, "close"] < price:
                        penetration = float(ordered.loc[index:reclaim, "high"].max()) - price
                        records.append(_sweep_record(ordered, index, reclaim, level, "bearish", penetration))
                        break
                break
    columns = [
        "sweep_timestamp", "direction", "swept_level", "level_type",
        "penetration_distance", "recovery_strength", "bars_to_reclaim", "extreme",
    ]
    return pd.DataFrame(records, columns=columns).sort_values(
        "sweep_timestamp", ignore_index=True
    ) if records else pd.DataFrame(columns=columns)


def _sweep_record(
    frame: pd.DataFrame, breach: int, reclaim: int, level: pd.Series, direction: str, penetration: float
) -> dict[str, object]:
    price = float(level["price"])
    close = float(frame.at[reclaim, "close"])
    recovery = abs(close - price) / penetration if penetration > 0 else 0.0
    extreme = float(frame.loc[breach:reclaim, "low"].min()) if direction == "bullish" else float(frame.loc[breach:reclaim, "high"].max())
    return {
        "sweep_timestamp": frame.at[reclaim, "timestamp"], "direction": direction,
        "swept_level": price, "level_type": level["level_type"],
        "penetration_distance": penetration, "recovery_strength": recovery,
        "bars_to_reclaim": reclaim - breach + 1, "extreme": extreme,
    }


def build_known_levels(
    frame: pd.DataFrame, swings: pd.DataFrame, sessions_config: SessionsConfig | None = None,
) -> pd.DataFrame:
    """Build completed day/week and confirmed swing liquidity events."""
    calendar = previous_day_week_levels(frame)
    records: list[dict[str, object]] = []
    for _, row in calendar.drop_duplicates(
        ["previous_day_high", "previous_day_low", "previous_week_high", "previous_week_low"]
    ).iterrows():
        for name, side in [
            ("previous_day_high", "buy"), ("previous_day_low", "sell"),
            ("previous_week_high", "buy"), ("previous_week_low", "sell"),
        ]:
            if pd.notna(row[name]):
                records.append({"known_time": row["timestamp"], "price": row[name], "level_type": name, "side": side})
    for _, row in swings.iterrows():
        records.append(
            {
                "known_time": row["confirmation_time"], "price": row["price"],
                "level_type": f"confirmed_swing_{row['swing_type']}",
                "side": "buy" if row["swing_type"] == "high" else "sell",
            }
        )
    output = pd.DataFrame(records, columns=["known_time", "price", "level_type", "side"])
    if sessions_config is not None:
        output = pd.concat([output, completed_session_levels(frame, sessions_config)], ignore_index=True)
    return output.sort_values("known_time", ignore_index=True)
