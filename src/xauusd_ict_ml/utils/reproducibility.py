"""Experiment provenance and deterministic seeds."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy; estimators receive the same seed explicitly."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 of a dataset or configuration file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Return HEAD when the working directory is inside a committed repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def create_run_directory(base: str | Path, prefix: str = "run") -> Path:
    """Create a unique UTC-timestamped experiment directory."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(base) / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(path: str | Path, **metadata: Any) -> None:
    """Persist environment and run metadata as JSON."""
    payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        **metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

