"""Tests for strategy bundle shaping."""

from __future__ import annotations

from src.services.strategy_bundle import build_strategy_bundle, strategy_bundle_has_content


def test_bundle_contains_only_dtc_ip_insights() -> None:
    dtc_results = {
        "insights": "### Engagement Horizon\nDTC summary text.",
        "evidence_by_section": {"Engagement Horizon": "Raw DTC evidence"},
    }
    ip_results = {
        "insights": "### Import / Export\nIP summary text.",
        "evidence_by_section": {"Import / Export": "Raw IP evidence"},
    }
    section_results = {
        "Midyear Metrics": {"summary": "Other summary", "evidence": "Other evidence"}
    }

    bundle = build_strategy_bundle(dtc_results, ip_results, section_results)

    assert "=== DTC (Fan / Engagement) ===" in bundle
    assert "=== IP (Catalog / Export) ===" in bundle
    assert "DTC summary text." in bundle
    assert "IP summary text." in bundle
    assert "Raw DTC evidence" not in bundle
    assert "Raw IP evidence" not in bundle
    assert "=== Other section summaries ===" not in bundle
    assert "Other summary" not in bundle
    assert "Other evidence" not in bundle


def test_bundle_has_content_only_for_dtc_or_ip_insights() -> None:
    empty_bundle = build_strategy_bundle(
        {"insights": "", "evidence_by_section": {"A": "ignored"}},
        {"insights": "", "evidence_by_section": {"B": "ignored"}},
        {"Other": {"summary": "ignored", "evidence": "ignored"}},
    )
    assert strategy_bundle_has_content(empty_bundle) is False
