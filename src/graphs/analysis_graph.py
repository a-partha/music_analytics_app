from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import GraphRecursionError

from src.agents.analysis.nodes import (
    assemble_outputs_react_node,
    assemble_outputs_node,
    filter_manifest_node,
    label_sections_node,
    prepare_subsection_fan_out,
    summarize_subsection_node,
)
from src.agents.analysis.react_agents import run_react_agent
from src.agents.analysis.react_tools import (
    activate_react_tool_context,
    build_react_run_context,
    materialize_judged_rows,
)
from src.config.analysis_mode import AnalysisMode, resolve_analysis_mode
from src.config.run_profiles import (
    RunProfile,
    profile_from_state_value,
    resolve_run_profile,
)
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


def _resolve_react_recursion_limit(section_count: int) -> int:
    import os

    raw = os.getenv("ANALYSIS_REACT_RECURSION_LIMIT", "").strip()
    if raw:
        try:
            return max(10, int(raw))
        except ValueError:
            pass
    return max(30, 8 * max(1, section_count))


def _route_after_filter(state: AnalysisState) -> str:
    profile = profile_from_state_value(state.get("run_profile"))
    if profile == RunProfile.DEV_ONE_PER_CATEGORY:
        return "react_mode_router"
    return "legacy_parallel_router"


def _route_react_mode(state: AnalysisState) -> str:
    mode = resolve_analysis_mode(state.get("analysis_mode"))
    if mode == AnalysisMode.DTC_ONLY:
        return "dtc_react_node"
    if mode == AnalysisMode.IP_ONLY:
        return "ip_react_node"
    if mode == AnalysisMode.BOTH:
        return "both_react_node"
    raise RetrievalError(
        "analysis_mode is required in Dev mode. Choose DTC only, IP only, or Both."
    )


def _legacy_parallel_router_node(_: AnalysisState) -> dict[str, Any]:
    return {}


def _react_mode_router_node(_: AnalysisState) -> dict[str, Any]:
    return {}


def _run_react_mode_node(
    state: AnalysisState,
    *,
    mode: AnalysisMode,
) -> dict[str, Any]:
    manifest = list(state.get("manifest") or [])
    context = build_react_run_context(
        analysis_mode=mode,
        file_search_store_name=state["file_search_store_name"],
        source_filename=state["source_filename"],
        gemini_model_name=state["gemini_model_name"] or "",
        synthesis_model_name=state["synthesis_model_name"] or "",
        pdf_hash=state.get("pdf_hash"),
        manifest=manifest,
    )
    recursion_limit = _resolve_react_recursion_limit(
        len(context.manifest_order)
    )
    t0 = time.perf_counter()
    messages: list[Any] = []
    trace: list[dict[str, Any]] = []
    trace_callback = state.get("react_trace_callback")

    def _emit_live_trace(rows: list[dict[str, Any]]) -> None:
        if not callable(trace_callback):
            return
        try:
            trace_callback(list(rows))
        except Exception:
            # Live UI rendering callback should never break the pipeline.
            return

    with activate_react_tool_context(context):
        try:
            messages, trace = run_react_agent(
                mode=mode,
                model_name=state.get("synthesis_model_name"),
                recursion_limit=recursion_limit,
                on_trace_update=_emit_live_trace,
            )
        except GraphRecursionError:
            context.errors.append(
                "ReAct recursion limit reached "
                f"(limit={recursion_limit}). Returning partial judged rows "
                "and backfilling the rest as OTHER."
            )
            trace.append(
                {
                    "step": 0,
                    "kind": "observation",
                    "tool": "runtime",
                    "text": (
                        "Recursion limit reached; partial rows retained and "
                        "remaining rows will be backfilled as OTHER."
                    ),
                }
            )
            _emit_live_trace(trace)
    timings = dict(state.get("timings") or {})
    timings[f"react_{mode.value}_s"] = time.perf_counter() - t0
    timings["react_recursion_limit"] = float(recursion_limit)
    timings["react_message_count"] = float(len(messages))
    out: dict[str, Any] = {
        "react_messages": trace,
        "judged_rows": materialize_judged_rows(context),
        "timings": timings,
    }
    if context.errors:
        out["errors"] = list(context.errors)
    return out


def dtc_react_node(state: AnalysisState) -> dict[str, Any]:
    return _run_react_mode_node(state, mode=AnalysisMode.DTC_ONLY)


def ip_react_node(state: AnalysisState) -> dict[str, Any]:
    return _run_react_mode_node(state, mode=AnalysisMode.IP_ONLY)


def both_react_node(state: AnalysisState) -> dict[str, Any]:
    return _run_react_mode_node(state, mode=AnalysisMode.BOTH)


def _build_analysis_graph() -> CompiledStateGraph:
    builder = StateGraph(AnalysisState)

    builder.add_node("filter_manifest", filter_manifest_node)
    builder.add_node("legacy_parallel_router", _legacy_parallel_router_node)
    builder.add_node("react_mode_router", _react_mode_router_node)
    builder.add_node("summarize_subsection", summarize_subsection_node)
    builder.add_node("label_sections", label_sections_node)
    builder.add_node("assemble_outputs", assemble_outputs_node)
    builder.add_node("dtc_react_node", dtc_react_node)
    builder.add_node("ip_react_node", ip_react_node)
    builder.add_node("both_react_node", both_react_node)
    builder.add_node("assemble_outputs_react", assemble_outputs_react_node)

    builder.add_edge(START, "filter_manifest")
    builder.add_conditional_edges(
        "filter_manifest",
        _route_after_filter,
        {
            "legacy_parallel_router": "legacy_parallel_router",
            "react_mode_router": "react_mode_router",
        },
    )
    builder.add_conditional_edges(
        "legacy_parallel_router",
        prepare_subsection_fan_out,
        ["summarize_subsection"],
    )
    builder.add_edge("summarize_subsection", "label_sections")
    builder.add_edge("label_sections", "assemble_outputs")
    builder.add_conditional_edges(
        "react_mode_router",
        _route_react_mode,
        {
            "dtc_react_node": "dtc_react_node",
            "ip_react_node": "ip_react_node",
            "both_react_node": "both_react_node",
        },
    )
    builder.add_edge("dtc_react_node", "assemble_outputs_react")
    builder.add_edge("ip_react_node", "assemble_outputs_react")
    builder.add_edge("both_react_node", "assemble_outputs_react")
    builder.add_edge("assemble_outputs", END)
    builder.add_edge("assemble_outputs_react", END)

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
    analysis_mode: str | None = None,
    dev_mode_flag: bool = False,
    timings_out: dict[str, float] | None = None,
    react_messages_out: list[dict[str, Any]] | None = None,
    react_trace_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    errors_out: list[str] | None = None,
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
    mode = resolve_analysis_mode(analysis_mode)
    if profile == RunProfile.DEV_ONE_PER_CATEGORY and mode is None:
        raise RetrievalError(
            "analysis_mode is required in Dev mode. Choose DTC only, IP only, or Both."
        )
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
            "analysis_mode": mode.value if mode else None,
            "neutral_rows": [],
            "errors": [],
            "react_messages": [],
            "judged_rows": [],
            "react_trace_callback": react_trace_callback,
        },
        config={"max_concurrency": max_workers},
    )
    wall_s = time.perf_counter() - t0

    if timings_out is not None:
        timings_out.update(final.get("timings") or {})
        timings_out["analysis_graph_wall_s"] = wall_s
    if react_messages_out is not None:
        react_messages_out.clear()
        react_messages_out.extend(list(final.get("react_messages") or []))
    if errors_out is not None:
        errors_out.clear()
        errors_out.extend(list(final.get("errors") or []))

    labeled = list(final.get("labeled_rows") or [])
    return (
        final.get("dtc_results") or {},
        final.get("ip_results") or {},
        final.get("section_results") or {},
        labeled,
    )
