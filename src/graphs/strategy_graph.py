from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.strategy.nodes import (
    build_bundle_node,
    generate_strategy_node,
    grounded_check_node,
    parse_and_dedupe_node,
    route_after_grounded_check,
)
from src.agents.strategy.postprocess import Recommendation
from src.graphs.state import StrategyState

_strategy_graph: CompiledStateGraph | None = None


def _build_strategy_graph() -> CompiledStateGraph:
    builder = StateGraph(StrategyState)

    builder.add_node("build_bundle", build_bundle_node)
    builder.add_node("generate_strategy", generate_strategy_node)
    builder.add_node("parse_and_dedupe", parse_and_dedupe_node)
    builder.add_node("grounded_check", grounded_check_node)

    builder.add_edge(START, "build_bundle")
    builder.add_edge("build_bundle", "generate_strategy")
    builder.add_edge("generate_strategy", "parse_and_dedupe")
    builder.add_edge("parse_and_dedupe", "grounded_check")
    builder.add_conditional_edges(
        "grounded_check",
        route_after_grounded_check,
        {"generate_strategy": "generate_strategy", "__end__": END},
    )

    return builder.compile()


def get_strategy_graph() -> CompiledStateGraph:
    global _strategy_graph
    if _strategy_graph is None:
        _strategy_graph = _build_strategy_graph()
    return _strategy_graph


def run_strategy_graph(
    *,
    dtc_results: dict[str, Any] | None = None,
    ip_results: dict[str, Any] | None = None,
    section_results: dict[str, dict[str, str]] | None = None,
    analysis_bundle: str | None = None,
    model_name: str | None = None,
    timings_out: dict[str, float] | None = None,
) -> list[Recommendation]:
    t0 = time.perf_counter()
    initial: StrategyState = {
        "dtc_results": dtc_results,
        "ip_results": ip_results,
        "section_results": section_results,
        "model_name": model_name,
        "grounded_retry_count": 0,
    }
    if analysis_bundle is not None:
        initial["analysis_bundle"] = analysis_bundle

    final = get_strategy_graph().invoke(initial)
    if timings_out is not None:
        timings_out.update(final.get("timings") or {})
        timings_out["strategy_graph_wall_s"] = time.perf_counter() - t0

    return list(final.get("recommendations") or [])
