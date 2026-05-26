"""Unit tests for fuzzy category assignment (no network)."""

from __future__ import annotations

from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
    assign_category,
)


def test_assign_streaming_atlas_ip() -> None:
    assert assign_category("Streaming Atlas") == CATEGORY_IP


def test_assign_import_export_ip() -> None:
    assert assign_category("Import / Export") == CATEGORY_IP


def test_assign_engagement_dtc() -> None:
    assert assign_category("Engagement Horizon") == CATEGORY_DTC


def test_assign_midyear_other() -> None:
    assert assign_category("Midyear Metrics") == CATEGORY_OTHER


def test_assign_gibberish_other() -> None:
    assert assign_category("Qwerty Completely Unknown 12345") == CATEGORY_OTHER
