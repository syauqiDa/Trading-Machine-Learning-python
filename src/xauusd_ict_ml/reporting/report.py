"""Self-contained HTML research report."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def generate_html_report(
    output_path: str | Path, title: str, metrics: dict[str, object],
    tables: dict[str, pd.DataFrame], chart_paths: list[str | Path], warnings: list[str] | None = None,
) -> Path:
    """Write an answer-first HTML report referencing colocated PNG assets."""
    warnings = warnings or []
    sections = [f"<h1>{html.escape(title)}</h1>", "<p><strong>Research and paper trading only.</strong> Historical results are not a profitability claim.</p>"]
    if warnings:
        sections.append("<h2>Validation warnings</h2><ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in warnings) + "</ul>")
    sections.append(f"<h2>Metrics</h2><pre>{html.escape(json.dumps(metrics, indent=2, default=str))}</pre>")
    for name, table in tables.items():
        sections.append(f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>{table.to_html(index=False, border=0)}")
    for chart in chart_paths:
        path = Path(chart)
        sections.append(f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem.replace("_", " ").title())}</figcaption></figure>')
    styles = "body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}table{border-collapse:collapse}td,th{padding:.4rem;border-bottom:1px solid #ddd}img{max-width:100%}pre{background:#f5f6f7;padding:1rem;overflow:auto}"
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"<!doctype html><meta charset='utf-8'><style>{styles}</style>" + "".join(sections), encoding="utf-8")
    return destination

