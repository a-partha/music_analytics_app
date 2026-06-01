from __future__ import annotations

from typing import Any

from src.config.run_profiles import RunProfile
from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
)


def _manifest_index(row: dict[str, Any]) -> int:
    raw = row.get("__manifest_index")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 10**9


def limit_labeled_rows_one_per_category(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep one labeled row per DTC / IP / OTHER, in manifest order.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=_manifest_index):
        cat = str(row.get("category") or CATEGORY_OTHER).upper()
        if cat not in {CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER}:
            cat = CATEGORY_OTHER
        if cat in seen:
            continue
        seen.add(cat)
        out.append(row)
        if len(seen) == 3:
            break
    return out


def apply_run_profile_to_manifest(
    manifest: list[dict[str, str]],
    profile: RunProfile,
) -> list[dict[str, str]]:
    """
    Keep manifest rows untouched for all profiles.

    Dev mode reduction happens after neutral summaries and LLM labeling.
    """
    _ = profile
    return list(manifest)
