"""Build a local, offline audit pack (JSON) summarizing one analysis run.

Pure functions only: no network calls, no Streamlit dependency. The pack is
meant to be downloaded directly from the demo UI as a self-contained record
of what the ReAct agent did, what it produced, and how it was judged.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.analysis.trace_formatter import renumber_trace_steps_for_display

_QUALITY_CONTROL_NOTE = (
    "Domain accept/reject decisions come from the in-app judge and "
    "validator chains (section_label_single_chains.py), not from any "
    "external service."
)


def _labeled_rows_summary(
    labeled_rows: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for row in labeled_rows or []:
        summary.append(
            {
                "display_title": str(row.get("display_title") or ""),
                "category": str(row.get("category") or ""),
            }
        )
    return summary


def build_audit_pack(
    *,
    source_filename: str | None,
    analysis_mode: str | None,
    run_profile: str | None,
    model_name: str | None,
    timings: dict[str, Any] | None,
    labeled_rows: list[dict[str, Any]] | None,
    dtc_results: dict[str, Any] | None,
    ip_results: dict[str, Any] | None,
    strategy_results: list[dict[str, Any]] | None,
    react_trace: list[dict[str, Any]] | None,
    react_trace_plain_english: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a JSON-serializable audit pack for the current run."""
    timestamp = generated_at or datetime.now(timezone.utc)
    return {
        "schema_version": "1.0",
        "generated_at": timestamp.isoformat(),
        "app": {
            "name": "music_analytics_app",
            "surface": "streamlit_demo",
        },
        "run": {
            "source_filename": source_filename,
            "analysis_mode": analysis_mode,
            "run_profile": run_profile,
            "model": model_name,
            "timings": dict(timings or {}),
        },
        "quality_control": {
            "note": _QUALITY_CONTROL_NOTE,
            "labeled_rows_summary": _labeled_rows_summary(labeled_rows),
        },
        "outputs": {
            "dtc_results": dtc_results,
            "ip_results": ip_results,
            "strategy_results": strategy_results,
        },
        "react_trace": renumber_trace_steps_for_display(list(react_trace or [])),
        "react_trace_plain_english": react_trace_plain_english,
    }


def build_audit_filename(
    *,
    source_filename: str | None,
    analysis_mode: str | None,
    generated_at: datetime | None = None,
) -> str:
    """Build a stable, filesystem-safe filename for the audit pack download."""
    timestamp = generated_at or datetime.now(timezone.utc)
    stem = Path(source_filename).stem if source_filename else "report"
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "report"
    safe_mode = re.sub(
        r"[^A-Za-z0-9_-]+", "_", (analysis_mode or "run").strip()
    ).strip("_") or "run"
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return f"audit_{safe_stem}_{safe_mode}_{stamp}.json"
