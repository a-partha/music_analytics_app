from __future__ import annotations

from src.agents.analysis import react_tools as rt
from src.config.analysis_mode import AnalysisMode


class _DummyRetrieveTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def invoke(self, payload: dict[str, str]) -> str:
        self.calls.append(payload)
        return f"EVIDENCE::{payload['subsection_key']}"


def test_react_tools_finish_uses_mode_satisfaction(monkeypatch) -> None:
    dummy = _DummyRetrieveTool()
    monkeypatch.setattr(rt, "retrieve_subsection_evidence_tool", dummy)
    monkeypatch.setattr(
        rt,
        "summarize_section_from_context",
        lambda section_name, grounded_context, model_name=None: (
            f"SUMMARY::{section_name}::{grounded_context[:12]}"
        ),
    )
    monkeypatch.setattr(
        rt,
        "label_one_dtc",
        lambda *, title, summary, model_name=None: "DTC",
    )
    monkeypatch.setattr(
        rt,
        "validate_label_dtc",
        lambda *, title, summary, predicted_label, model_name=None: {
            "decision": "ACCEPT",
            "reason": "",
        },
    )

    ctx = rt.build_react_run_context(
        analysis_mode=AnalysisMode.DTC_ONLY,
        file_search_store_name="store",
        source_filename="report.pdf",
        gemini_model_name="gem-model",
        synthesis_model_name="syn-model",
        pdf_hash=None,
        manifest=[
            {"subsection_key": "a", "display_title": "A"},
            {"subsection_key": "b", "display_title": "B"},
        ],
    )

    with rt.activate_react_tool_context(ctx):
        pending = rt.list_pending_subsections.invoke({})
        assert [row["subsection_key"] for row in pending] == ["a", "b"]

        first = rt.judge_section_dtc.invoke({"subsection_key": "a"})
        assert first["label"] == "DTC"
        assert first["accepted_for_target"] is True

        assert rt.finish.invoke({}) == "FINISH_OK"
        pending_after = rt.list_pending_subsections.invoke({})
        assert [row["subsection_key"] for row in pending_after] == ["b"]

    rows = rt.materialize_judged_rows(ctx)
    assert [row["subsection_key"] for row in rows] == ["a"]
    assert len(dummy.calls) == 1


def test_react_tools_validator_reject_then_next_section(monkeypatch) -> None:
    dummy = _DummyRetrieveTool()
    monkeypatch.setattr(rt, "retrieve_subsection_evidence_tool", dummy)
    monkeypatch.setattr(
        rt,
        "summarize_section_from_context",
        lambda section_name, grounded_context, model_name=None: (
            f"SUMMARY::{section_name}::{grounded_context[:12]}"
        ),
    )
    monkeypatch.setattr(
        rt,
        "label_one_dtc",
        lambda *, title, summary, model_name=None: "DTC",
    )

    def _validator(*, title, summary, predicted_label, model_name=None):
        if title == "A":
            return {"decision": "REJECT", "reason": "Weak thematic match."}
        return {"decision": "ACCEPT", "reason": ""}

    monkeypatch.setattr(rt, "validate_label_dtc", _validator)

    ctx = rt.build_react_run_context(
        analysis_mode=AnalysisMode.DTC_ONLY,
        file_search_store_name="store",
        source_filename="report.pdf",
        gemini_model_name="gem-model",
        synthesis_model_name="syn-model",
        pdf_hash=None,
        manifest=[
            {"subsection_key": "a", "display_title": "A"},
            {"subsection_key": "b", "display_title": "B"},
        ],
    )

    with rt.activate_react_tool_context(ctx):
        first = rt.judge_section_dtc.invoke({"subsection_key": "a"})
        assert first["validator_decision"] == "REJECT"
        assert first["accepted_for_target"] is False
        not_done = rt.finish.invoke({})
        assert "Missing targets: DTC" in not_done

        second = rt.judge_section_dtc.invoke({"subsection_key": "b"})
        assert second["validator_decision"] == "ACCEPT"
        assert second["accepted_for_target"] is True
        assert rt.finish.invoke({}) == "FINISH_OK"

    rows = rt.materialize_judged_rows(ctx)
    assert [row["subsection_key"] for row in rows] == ["a", "b"]
    assert rows[0]["accepted_for_target"] is False
    assert rows[1]["accepted_for_target"] is True

