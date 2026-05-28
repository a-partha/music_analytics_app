from __future__ import annotations

import pytest

from src.config.run_profiles import RunProfile
from src.graphs import analysis_graph as ag
from src.pipelines.analysis_pipeline import run_analysis
from src.services.file_search_retrieval import RetrievalError


def test_run_analysis_graph_requires_mode_in_dev_profile() -> None:
    with pytest.raises(RetrievalError):
        ag.run_analysis_graph(
            file_search_store_name="store",
            manifest=[{"subsection_key": "a", "display_title": "A"}],
            source_filename="report.pdf",
            pdf_hash="hash",
            run_profile=RunProfile.DEV_ONE_PER_CATEGORY,
            analysis_mode=None,
        )


def test_run_analysis_graph_allows_missing_mode_in_full_profile(monkeypatch) -> None:
    class _FakeGraph:
        def __init__(self) -> None:
            self.initial = None

        def invoke(self, initial, config=None):
            self.initial = initial
            return {
                "dtc_results": {},
                "ip_results": {},
                "section_results": {},
                "labeled_rows": [],
                "timings": {},
                "react_messages": [],
            }

    fake = _FakeGraph()
    monkeypatch.setattr(ag, "get_analysis_graph", lambda: fake)

    run_analysis(
        file_search_store_name="store",
        manifest=[{"subsection_key": "a", "display_title": "A"}],
        source_filename="report.pdf",
        pdf_hash="hash",
        run_profile=RunProfile.FULL.value,
        analysis_mode=None,
    )
    assert fake.initial is not None
    assert fake.initial["analysis_mode"] is None
    assert fake.initial["run_profile"] == RunProfile.FULL.value


def test_run_analysis_threads_mode_to_graph(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_analysis_graph(**kwargs):
        captured.update(kwargs)
        return {}, {}, {}, []

    monkeypatch.setattr(ag, "run_analysis_graph", _fake_run_analysis_graph)

    callback = lambda rows: rows
    run_analysis(
        file_search_store_name="store",
        manifest=[{"subsection_key": "a", "display_title": "A"}],
        source_filename="report.pdf",
        pdf_hash="hash",
        run_profile=RunProfile.DEV_ONE_PER_CATEGORY.value,
        analysis_mode="ip_only",
        react_trace_callback=callback,
    )
    assert captured["analysis_mode"] == "ip_only"
    assert captured["react_trace_callback"] is callback

