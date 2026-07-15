"""Small dispatcher for the runnable research stages."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="xauusd-ict-ml")
    parser.add_argument("stage", choices=["prepare", "detect", "backtest", "dataset", "train", "walk-forward", "report"])
    parser.add_argument("--config")
    args = parser.parse_args()
    scripts = {"prepare": "prepare_data.py", "detect": "detect_setups.py", "backtest": "run_backtest.py", "dataset": "build_dataset.py", "train": "train_model.py", "walk-forward": "run_walk_forward.py", "report": "generate_report.py"}
    command = [sys.executable, f"scripts/{scripts[args.stage]}"]
    if args.config: command.extend(["--config", args.config])
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__": main()
