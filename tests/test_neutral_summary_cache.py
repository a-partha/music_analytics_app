"""Tests for neutral summary cache keys and manifest ordering."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.pipelines import analysis_pipeline
from src.services.neutral_summary_cache import (
    get_neutral_row,
    make_neutral_cache_key,
    put_neutral_row,
)


def test_make_neutral_cache_key_stable() -> None:
    k1 = make_neutral_cache_key("abc", "sec1", "gemini-x", "synthesis-y")
    k2 = make_neutral_cache_key("abc", "sec1", "gemini-x", "synthesis-y")
    assert k1 == k2
    assert len(k1) == 64


def test_make_neutral_cache_key_differs_on_inputs() -> None:
    base = ("pdfhash", "sk", "g", "o")
    k0 = make_neutral_cache_key(*base)
    assert make_neutral_cache_key("other", *base[1:]) != k0
    assert make_neutral_cache_key(base[0], "other", *base[2:]) != k0
    assert make_neutral_cache_key(*base[:2], "other", base[3]) != k0
    assert make_neutral_cache_key(*base[:3], "other") != k0


def test_cache_always_writes_even_if_disable_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache is mandatory; legacy disable env vars are ignored."""
    sec_json = tmp_path / "section_cache.json"
    monkeypatch.setenv("SECTION_CACHE_PATH", str(sec_json))
    monkeypatch.setenv("NEUTRAL_SUMMARY_CACHE_DISABLED", "true")

    put_neutral_row("h2", "k2", "g2", "o2", "ev2", "sum2")
    assert get_neutral_row("h2", "k2", "g2", "o2") == {
        "evidence": "ev2",
        "summary": "sum2",
    }


def test_cache_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sec_json = tmp_path / "section_cache.json"
    monkeypatch.setenv("SECTION_CACHE_PATH", str(sec_json))

    put_neutral_row("h1", "k1", "g1", "o1", "ev text", "sum text")
    row = get_neutral_row("h1", "k1", "g1", "o1")
    assert row == {"evidence": "ev text", "summary": "sum text"}

    neutral_path = tmp_path / "neutral_summary_cache.json"
    assert neutral_path.exists()
    data = json.loads(neutral_path.read_text(encoding="utf-8"))
    key = make_neutral_cache_key("h1", "k1", "g1", "o1")
    assert data[key]["evidence"] == "ev text"


def test_parallel_manifest_order(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = {"A": 0.06, "B": 0.02, "C": 0.04}

    def fake_retrieve(**kwargs: object) -> str:
        return "ev"

    def fake_summarize(
        section_name: str,
        grounded_context: str,
        model_name: str | None = None,
    ) -> str:
        time.sleep(delays[section_name])
        return f"sum-{section_name}"

    monkeypatch.setattr(
        analysis_pipeline,
        "retrieve_subsection_evidence",
        fake_retrieve,
    )
    monkeypatch.setattr(
        analysis_pipeline,
        "summarize_section_from_context",
        fake_summarize,
    )

    manifest = [
        {"subsection_key": "ka", "display_title": "A"},
        {"subsection_key": "kb", "display_title": "B"},
        {"subsection_key": "kc", "display_title": "C"},
    ]
    rows = analysis_pipeline.run_neutral_section_summaries_for_manifest(
        file_search_store_name="store",
        manifest=manifest,
        source_filename="report.pdf",
        gemini_model_name="gemini-test",
        synthesis_model_name="synthesis-test",
        pdf_hash=None,
    )
    titles = [r["display_title"] for r in rows]
    assert titles == ["A", "B", "C"]
