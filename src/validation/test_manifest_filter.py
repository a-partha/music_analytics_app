"""Unit tests for manifest filtering (no network)."""

from __future__ import annotations

from src.config.run_profiles import RunProfile
from src.services.manifest_filter import (
    apply_run_profile_to_manifest,
    limit_labeled_rows_one_per_category,
)


def test_apply_dev_profile() -> None:
    manifest = [
        {"subsection_key": "a", "display_title": "Engagement Horizon"},
        {"subsection_key": "b", "display_title": "Streaming Atlas"},
        {"subsection_key": "c", "display_title": "Midyear Charts"},
    ]
    full = apply_run_profile_to_manifest(manifest, RunProfile.FULL)
    assert len(full) == 3
    dev = apply_run_profile_to_manifest(manifest, RunProfile.DEV_ONE_PER_CATEGORY)
    assert len(dev) == 3
    assert dev == manifest


def test_limit_labeled_rows_post_label_uses_manifest_order() -> None:
    labeled = [
        {"display_title": "Late OTHER", "category": "OTHER", "__manifest_index": 4},
        {"display_title": "Early DTC", "category": "DTC", "__manifest_index": 0},
        {"display_title": "Early OTHER", "category": "OTHER", "__manifest_index": 1},
        {"display_title": "Early IP", "category": "IP", "__manifest_index": 2},
        {"display_title": "Late IP", "category": "IP", "__manifest_index": 3},
    ]
    out = limit_labeled_rows_one_per_category(labeled)
    assert [r["display_title"] for r in out] == [
        "Early DTC",
        "Early OTHER",
        "Early IP",
    ]


def test_limit_labeled_rows_allows_missing_categories() -> None:
    labeled = [
        {"display_title": "Only Other 1", "category": "OTHER", "__manifest_index": 0},
        {"display_title": "Only Other 2", "category": "OTHER", "__manifest_index": 1},
    ]
    out = limit_labeled_rows_one_per_category(labeled)
    assert len(out) == 1
    assert out[0]["display_title"] == "Only Other 1"
