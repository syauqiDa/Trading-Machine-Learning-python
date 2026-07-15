"""Make direct ``python scripts/...`` execution independent of installation mode."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap() -> None:
    source = str(Path(__file__).resolve().parents[1] / "src")
    if source not in sys.path:
        sys.path.insert(0, source)

