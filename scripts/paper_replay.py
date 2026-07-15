#!/usr/bin/env python3
"""Run explicit-gate, offline paper replay; never connects to a broker."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import bootstrap

bootstrap()

from xauusd_ict_ml.paper import ValidationGate, run_offline_paper_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, help="User-created JSON with four explicit boolean approvals")
    parser.add_argument("--data", default="data/processed/xauusd_15m.parquet")
    parser.add_argument("--setups", default="data/processed/xauusd_setups.parquet")
    parser.add_argument("--output", default="reports/paper_replay_alerts.csv")
    args = parser.parse_args()
    alerts = run_offline_paper_replay(pd.read_parquet(args.data), pd.read_parquet(args.setups), ValidationGate.from_json(args.gate))
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True); alerts.to_csv(destination, index=False)


if __name__ == "__main__": main()
