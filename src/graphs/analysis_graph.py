from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.analysis.nodes import (
    assemble_outputs_node,
    filter_manifest_node,
    label_sections_node,
    prepare_subsection_fan_out,
    summarize_subsection_node,
)
from src.config.run_profiles import RunProfile, resolve_run_profile
from src.graphs.state import AnalysisState
from src.services.file_search_retrieval import RetrievalError
from src.services.neutral_summary_cache import (
    resolved_gemini_model,
    resolved_synthesis_model,
)

_analysis_graph: CompiledStateGraph | None = None


def _resolve_neutral_summary_workers() -> int:
    import os

    raw = os.getenv("NEUTRAL_SUMMARY_MAX_WORKERS", "").strip() or "4"
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _build_analysis_graph() -> CompiledStateGraph:
    builder = StateGraph(AnalysisState)

    builder.add_node("filter_manifest", filter_manifest_node)
    builder.add_node("summarize_subsection", summarize_subsection_node)
    builder.add_node("label_sections", label_sections_node)
    builder.add_node("assemble_outputs", assemble_outputs_node)

    builder.add_edge(START, "filter_manifest")
    builder.add_conditional_edges(
        "filter_manifest",
        prepare_subsection_fan_out,
        ["summarize_subsection"],
    )
    builder.add_edge("summarize_subsection", "label_sections")
    builder.add_edge("label_sections", "assemble_outputs")
    builder.add_edge("assemble_outputs", END)

    return builder.compile()


def get_analysis_graph() -> CompiledStateGraph:
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = _build_analysis_graph()
    return _analysis_graph


def run_analysis_graph(
    *,
    file_search_store_name: str,
    manifest: list[dict[str, str]],
    source_filename: str | None,
    pdf_hash: str | None = None,
    gemini_model_name: str | None = None,
    synthesis_model_name: str | None = None,
    run_profile: RunProfile | str | None = None,
    dev_mode_flag: bool = False,
    timings_out: dict[str, float] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
]:
    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval."
        )
    if not pdf_hash:
        raise RetrievalError(
            "Missing pdf_hash; required for neutral summary disk cache."
        )

    if run_profile is not None:
        profile = resolve_run_profile(run_profile, use_env=False)
    elif dev_mode_flag:
        profile = RunProfile.DEV_ONE_PER_CATEGORY
    else:
        profile = resolve_run_profile(None, dev_mode_flag=False, use_env=True)
    gemini = resolved_gemini_model(gemini_model_name)
    synthesis = resolved_synthesis_model(synthesis_model_name)
    max_workers = _resolve_neutral_summary_workers()

    t0 = time.perf_counter()
    final = get_analysis_graph().invoke(
        {
            "file_search_store_name": file_search_store_name,
            "manifest": manifest,
            "source_filename": source_filename,
            "pdf_hash": pdf_hash,
            "gemini_model_name": gemini,
            "synthesis_model_name": synthesis,
            "run_profile": profile.value,
            "neutral_rows": [],
            "errors": [],
        },
        config={"max_concurrency": max_workers},
    )
    wall_s = time.perf_counter() - t0

    if timings_out is not None:
        timings_out.update(final.get("timings") or {})
        timings_out["analysis_graph_wall_s"] = wall_s

    labeled = list(final.get("labeled_rows") or [])
    return (
        final.get("dtc_results") or {},
        final.get("ip_results") or {},
        final.get("section_results") or {},
        labeled,
    )
