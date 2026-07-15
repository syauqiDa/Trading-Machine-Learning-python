"""Offline, validation-gated paper replay with no broker connectivity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ValidationGate:
    tests_passed: bool
    data_quality_passed: bool
    walk_forward_completed: bool
    user_approved: bool

    @property
    def passed(self) -> bool:
        return all((self.tests_passed, self.data_quality_passed, self.walk_forward_completed, self.user_approved))

    @classmethod
    def from_json(cls, path: str | Path) -> "ValidationGate":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def run_offline_paper_replay(frame: pd.DataFrame, setups: pd.DataFrame, gate: ValidationGate) -> pd.DataFrame:
    """Replay closed-bar setup alerts only after the explicit gate passes.

    This function has no networking, broker API, order submission, or live data
    code. Its output is an immutable-style alert log for educational review.
    """
    if not gate.passed:
        raise PermissionError("Paper replay is disabled until every validation gate is true")
    available = set(pd.to_datetime(frame["timestamp"], utc=True))
    alerts = setups[pd.to_datetime(setups["entry_timestamp"], utc=True).isin(available)].copy()
    alerts["paper_status"] = "candidate_only_no_order_sent"
    alerts["processed_on_closed_bar"] = True
    return alerts

