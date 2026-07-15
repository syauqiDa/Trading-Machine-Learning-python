"""Target-before-stop setup labeling."""

from __future__ import annotations

import pandas as pd


def label_target_before_stop(
    frame: pd.DataFrame, setups: pd.DataFrame, max_holding_bars: int = 48,
    intrabar_rule: str = "stop_first",
) -> pd.DataFrame:
    """Label each setup from future bars; features must never call this function."""
    records: list[dict[str, object]] = []
    for _, setup in setups.iterrows():
        start = int(setup["entry_index"])
        stop, target = float(setup["stop_price"]), float(setup["target_price"])
        bullish = setup["direction"] == "bullish"
        label: int | None = None
        outcome = "neutral"
        end = min(len(frame) - 1, start + max_holding_bars)
        ambiguous = False
        for index in range(start + 1, end + 1):
            bar = frame.iloc[index]
            stop_hit = bar["low"] <= stop if bullish else bar["high"] >= stop
            target_hit = bar["high"] >= target if bullish else bar["low"] <= target
            if stop_hit and target_hit:
                ambiguous = True
                if intrabar_rule == "target_first":
                    label, outcome = 1, "target"
                else:
                    label, outcome = 0, "stop"
                end = index
                break
            if stop_hit:
                label, outcome, end = 0, "stop", index
                break
            if target_hit:
                label, outcome, end = 1, "target", index
                break
        records.append(
            {
                "setup_id": int(setup["setup_id"]), "label": label, "label_outcome": outcome,
                "label_start_index": start, "label_end_index": end,
                "label_start_time": frame.at[start, "timestamp"], "label_end_time": frame.at[end, "timestamp"],
                "label_ambiguous": ambiguous,
            }
        )
    return pd.DataFrame(records)

