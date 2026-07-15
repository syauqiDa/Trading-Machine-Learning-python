"""Generic price-scaled triple-barrier labels."""

from __future__ import annotations

import pandas as pd


def triple_barrier_labels(
    frame: pd.DataFrame, decision_indices: list[int], upper_fraction: float,
    lower_fraction: float, max_holding_bars: int, side: int = 1,
) -> pd.DataFrame:
    """Apply upper, lower, and time barriers from each decision close."""
    if upper_fraction <= 0 or lower_fraction <= 0:
        raise ValueError("Barrier fractions must be positive")
    records: list[dict[str, object]] = []
    for start in decision_indices:
        entry = float(frame.at[start, "close"])
        upper = entry * (1 + upper_fraction)
        lower = entry * (1 - lower_fraction)
        label = 0
        end = min(len(frame) - 1, start + max_holding_bars)
        for index in range(start + 1, end + 1):
            up = frame.at[index, "high"] >= upper
            down = frame.at[index, "low"] <= lower
            if up or down:
                label = (1 if up and not down else -1) * side
                end = index
                break
        records.append({"decision_index": start, "label": label, "label_end_index": end})
    return pd.DataFrame(records)

