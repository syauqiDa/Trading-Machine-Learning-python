"""Consistent application logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Configure console and optional file logging without duplicate handlers."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

