"""Tests for neutral summary worker env key resolution."""

from __future__ import annotations

from src.graphs import analysis_graph
from src.pipelines import analysis_pipeline


def test_worker_count_uses_plural_key(monkeypatch) -> None:
    monkeypatch.setenv("NEUTRAL_SUMMARY_MAX_WORKERS", "7")
    monkeypatch.setenv("NEUTRAL_SUMMARY_MAX_WORKER", "99")

    assert analysis_pipeline._resolve_neutral_summary_workers() == 7
    assert analysis_graph._resolve_neutral_summary_workers() == 7


def test_worker_count_ignores_singular_key(monkeypatch) -> None:
    monkeypatch.delenv("NEUTRAL_SUMMARY_MAX_WORKERS", raising=False)
    monkeypatch.setenv("NEUTRAL_SUMMARY_MAX_WORKER", "11")

    assert analysis_pipeline._resolve_neutral_summary_workers() == 4
    assert analysis_graph._resolve_neutral_summary_workers() == 4
