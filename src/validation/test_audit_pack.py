"""Unit tests for the local audit pack builder (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.services.audit_pack import build_audit_filename, build_audit_pack


def test_build_audit_pack_is_json_serializable_and_has_expected_shape() -> None:
    pack = build_audit_pack(
        source_filename="IFPI_Global_Music_Report_2025.pdf",
        analysis_mode="dtc_only",
        run_profile="dev_one_per_category",
        model_name="gemini-3.1-flash-lite",
        timings={"react_dtc_only_s": 12.3},
        labeled_rows=[
            {"display_title": "Fan engagement", "category": "DTC"},
            {"display_title": "Charts", "category": "OTHER"},
        ],
        dtc_results={"insights": "Fans stream more.", "evidence_by_section": {}},
        ip_results=None,
        strategy_results=None,
        react_trace=[{"step": 1, "kind": "thought", "text": "Planned next step."}],
        react_trace_plain_english="Step 1: Planned the next action.",
        generated_at=datetime(2026, 7, 12, 20, 41, 3, tzinfo=timezone.utc),
    )

    serialized = json.dumps(pack)
    reloaded = json.loads(serialized)

    assert reloaded["schema_version"] == "1.0"
    assert reloaded["generated_at"] == "2026-07-12T20:41:03+00:00"
    assert reloaded["app"] == {
        "name": "music_analytics_app",
        "surface": "streamlit_demo",
    }
    assert reloaded["run"]["source_filename"] == "IFPI_Global_Music_Report_2025.pdf"
    assert reloaded["run"]["analysis_mode"] == "dtc_only"
    assert reloaded["run"]["timings"] == {"react_dtc_only_s": 12.3}
    assert "judge and validator" in reloaded["quality_control"]["note"]
    assert reloaded["quality_control"]["labeled_rows_summary"] == [
        {"display_title": "Fan engagement", "category": "DTC"},
        {"display_title": "Charts", "category": "OTHER"},
    ]
    assert reloaded["outputs"]["dtc_results"]["insights"] == "Fans stream more."
    assert reloaded["outputs"]["ip_results"] is None
    assert reloaded["react_trace"] == [
        {"step": 1, "kind": "thought", "text": "Planned next step."}
    ]
    assert reloaded["react_trace_plain_english"] == "Step 1: Planned the next action."


def test_build_audit_pack_defaults_missing_inputs_to_empty() -> None:
    pack = build_audit_pack(
        source_filename=None,
        analysis_mode=None,
        run_profile=None,
        model_name=None,
        timings=None,
        labeled_rows=None,
        dtc_results=None,
        ip_results=None,
        strategy_results=None,
        react_trace=None,
        react_trace_plain_english="Waiting for agent activity...",
    )

    assert pack["run"]["timings"] == {}
    assert pack["quality_control"]["labeled_rows_summary"] == []
    assert pack["react_trace"] == []


def test_build_audit_pack_renumbers_step_one_spam() -> None:
    pack = build_audit_pack(
        source_filename="report.pdf",
        analysis_mode="dtc_only",
        run_profile="dev_one_per_category",
        model_name="test",
        timings={},
        labeled_rows=None,
        dtc_results=None,
        ip_results=None,
        strategy_results=None,
        react_trace=[
            {"step": 1, "kind": "tool", "tool": "list_pending_subsections", "args": {}},
            {
                "step": 1,
                "kind": "observation",
                "tool": "list_pending_subsections",
                "text": "[]",
            },
            {
                "step": 1,
                "kind": "tool",
                "tool": "finish",
                "args": {},
            },
            {
                "step": 1,
                "kind": "observation",
                "tool": "finish",
                "text": "FINISH_OK",
            },
        ],
        react_trace_plain_english="ignored",
    )
    assert [row["step"] for row in pack["react_trace"]] == [1, 1, 2, 2]


def test_build_audit_filename_is_safe_and_stable() -> None:
    name = build_audit_filename(
        source_filename="My Report (Final) v2.pdf",
        analysis_mode="ip_only",
        generated_at=datetime(2026, 7, 12, 20, 41, 3, tzinfo=timezone.utc),
    )
    assert name == "audit_My_Report_Final_v2_ip_only_20260712T204103Z.json"


def test_build_audit_filename_falls_back_when_missing_source_filename() -> None:
    name = build_audit_filename(
        source_filename=None,
        analysis_mode=None,
        generated_at=datetime(2026, 7, 12, 20, 41, 3, tzinfo=timezone.utc),
    )
    assert name == "audit_report_run_20260712T204103Z.json"
