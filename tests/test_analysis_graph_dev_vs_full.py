from __future__ import annotations

from src.config.analysis_mode import AnalysisMode
from src.config.run_profiles import RunProfile
from src.graphs import analysis_graph as ag


def test_route_after_filter_uses_full_vs_dev_profile() -> None:
    assert (
        ag._route_after_filter({"run_profile": RunProfile.FULL.value})
        == "legacy_parallel_router"
    )
    assert (
        ag._route_after_filter(
            {"run_profile": RunProfile.DEV_ONE_PER_CATEGORY.value}
        )
        == "react_mode_router"
    )


def test_route_react_mode_selects_expected_node() -> None:
    assert (
        ag._route_react_mode({"analysis_mode": AnalysisMode.DTC_ONLY.value})
        == "dtc_react_node"
    )
    assert (
        ag._route_react_mode({"analysis_mode": AnalysisMode.IP_ONLY.value})
        == "ip_react_node"
    )
    assert (
        ag._route_react_mode({"analysis_mode": AnalysisMode.BOTH.value})
        == "both_react_node"
    )


def test_react_node_applies_recursion_guardrail(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_react_agent(
        *, mode, model_name, recursion_limit, on_trace_update=None
    ):
        captured["mode"] = mode
        captured["model_name"] = model_name
        captured["recursion_limit"] = recursion_limit
        if on_trace_update is not None:
            on_trace_update([{"step": 1, "kind": "thought", "text": "live"}])
        return [], [{"step": 1, "kind": "thought", "text": "live"}]

    monkeypatch.setattr(ag, "run_react_agent", _fake_run_react_agent)

    out = ag.dtc_react_node(
        {
            "manifest": [
                {"subsection_key": "a", "display_title": "A"},
            ],
            "file_search_store_name": "store",
            "source_filename": "report.pdf",
            "gemini_model_name": "gem-model",
            "synthesis_model_name": "syn-model",
            "errors": [],
            "timings": {},
        }
    )
    assert captured["mode"] == AnalysisMode.DTC_ONLY
    assert captured["recursion_limit"] >= 30
    assert out["timings"]["react_recursion_limit"] == float(
        captured["recursion_limit"]
    )
    assert out["react_messages"][0]["text"] == "live"


def test_react_recursion_limit_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ANALYSIS_REACT_RECURSION_LIMIT", "77")
    assert ag._resolve_react_recursion_limit(2) == 77


def test_react_node_threads_mode_into_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_builder = ag.build_react_run_context

    def _builder(**kwargs):
        captured["analysis_mode"] = kwargs["analysis_mode"]
        return original_builder(**kwargs)

    def _fake_run_react_agent(
        *, mode, model_name, recursion_limit, on_trace_update=None
    ):
        return [], []

    monkeypatch.setattr(ag, "build_react_run_context", _builder)
    monkeypatch.setattr(ag, "run_react_agent", _fake_run_react_agent)

    ag.ip_react_node(
        {
            "manifest": [{"subsection_key": "a", "display_title": "A"}],
            "file_search_store_name": "store",
            "source_filename": "report.pdf",
            "gemini_model_name": "gem-model",
            "synthesis_model_name": "syn-model",
            "errors": [],
            "timings": {},
        }
    )
    assert captured["analysis_mode"] == AnalysisMode.IP_ONLY

