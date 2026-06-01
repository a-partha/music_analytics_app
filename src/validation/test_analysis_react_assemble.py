from __future__ import annotations

import pytest

from src.agents.analysis.nodes import assemble_outputs_react_node
from src.services.file_search_retrieval import RetrievalError


def test_assemble_outputs_react_enforces_mode_specific_cap() -> None:
    state = {
        "manifest": [
            {"subsection_key": "a", "display_title": "A"},
            {"subsection_key": "b", "display_title": "B"},
            {"subsection_key": "c", "display_title": "C"},
            {"subsection_key": "d", "display_title": "D"},
        ],
        "analysis_mode": "both",
        "judged_rows": [
            {
                "subsection_key": "a",
                "display_title": "A",
                "summary": "DTC summary",
                "evidence": "a-evidence",
                "category": "DTC",
                "accepted_for_target": True,
            },
            {
                "subsection_key": "b",
                "display_title": "B",
                "summary": "Second DTC summary",
                "evidence": "b-evidence",
                "category": "DTC",
                "accepted_for_target": True,
            },
            {
                "subsection_key": "c",
                "display_title": "C",
                "summary": "IP summary",
                "evidence": "c-evidence",
                "category": "IP",
                "accepted_for_target": True,
            },
            {
                "subsection_key": "d",
                "display_title": "D",
                "summary": "OTHER summary",
                "evidence": "d-evidence",
                "category": "OTHER",
                "accepted_for_target": False,
            },
        ],
        "timings": {},
    }

    out = assemble_outputs_react_node(state)

    labeled = out["labeled_rows"]
    assert [row["subsection_key"] for row in labeled] == ["a", "c"]
    assert "### B" not in out["dtc_results"].get("insights", "")
    assert out["section_results"] == {}
    assert out["timings"]["react_rows_after_mode_cap"] == 2.0


def test_assemble_outputs_react_fails_when_dtc_target_missing() -> None:
    state = {
        "manifest": [{"subsection_key": "a", "display_title": "A"}],
        "analysis_mode": "dtc_only",
        "judged_rows": [
            {
                "subsection_key": "a",
                "display_title": "A",
                "summary": "Looks somewhat DTC but validator rejected it.",
                "evidence": "a-evidence",
                "category": "DTC",
                "accepted_for_target": False,
            }
        ],
        "timings": {},
    }

    with pytest.raises(
        RetrievalError,
        match=r"required target label\(s\): DTC",
    ):
        assemble_outputs_react_node(state)


def test_assemble_outputs_react_allows_partial_both_when_one_target_found() -> None:
    state = {
        "manifest": [
            {"subsection_key": "a", "display_title": "A"},
            {"subsection_key": "b", "display_title": "B"},
        ],
        "analysis_mode": "both",
        "judged_rows": [
            {
                "subsection_key": "a",
                "display_title": "A",
                "summary": "DTC summary",
                "evidence": "a-evidence",
                "category": "DTC",
                "accepted_for_target": True,
            },
            {
                "subsection_key": "b",
                "display_title": "B",
                "summary": "IP-flavored text but rejected",
                "evidence": "b-evidence",
                "category": "IP",
                "accepted_for_target": False,
            },
        ],
        "timings": {},
    }

    out = assemble_outputs_react_node(state)

    assert [row["subsection_key"] for row in out["labeled_rows"]] == ["a"]
    assert "DTC summary" in out["dtc_results"].get("insights", "")
    assert out["ip_results"].get("insights", "") == ""
    assert any("partial success" in err for err in out.get("errors", []))


def test_assemble_outputs_react_fails_when_both_mode_missing_all_targets() -> None:
    state = {
        "manifest": [
            {"subsection_key": "a", "display_title": "A"},
            {"subsection_key": "b", "display_title": "B"},
        ],
        "analysis_mode": "both",
        "judged_rows": [
            {
                "subsection_key": "a",
                "display_title": "A",
                "summary": "DTC-flavored text but rejected",
                "evidence": "a-evidence",
                "category": "DTC",
                "accepted_for_target": False,
            },
            {
                "subsection_key": "b",
                "display_title": "B",
                "summary": "IP-flavored text but rejected",
                "evidence": "b-evidence",
                "category": "IP",
                "accepted_for_target": False,
            },
        ],
        "timings": {},
    }

    with pytest.raises(
        RetrievalError,
        match=r"required target label\(s\): DTC, IP",
    ):
        assemble_outputs_react_node(state)
