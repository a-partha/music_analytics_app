from __future__ import annotations

import time
from typing import Any

from src.agents.strategy.postprocess import (
    dedupe_recommendations,
    filter_grounded_recommendations,
    parse_strategy_markdown,
)
from src.chains.strategy_chain import generate_strategy_recommendations
from src.graphs.state import StrategyState
from src.services.strategy_bundle import build_strategy_bundle


def build_bundle_node(state: StrategyState) -> dict[str, Any]:
    if state.get("analysis_bundle"):
        return {}
    bundle = build_strategy_bundle(
        state.get("dtc_results"),
        state.get("ip_results"),
        state.get("section_results"),
    )
    return {"analysis_bundle": bundle}


def generate_strategy_node(state: StrategyState) -> dict[str, Any]:
    t0 = time.perf_counter()
    bundle = state.get("analysis_bundle") or ""
    nudge = ""
    if state.get("grounded_retry_count", 0) > 0:
        nudge = (
            "\n\nIMPORTANT: Each recommendation MUST include non-empty "
            "**Evidence:** and **Action:** fields citing the source sections."
        )
    raw = generate_strategy_recommendations(
        bundle + nudge,
        model_name=state.get("model_name"),
    )
    timings = dict(state.get("timings") or {})
    timings["generate_strategy_s"] = time.perf_counter() - t0
    return {"raw_markdown": raw, "timings": timings}


def parse_and_dedupe_node(state: StrategyState) -> dict[str, Any]:
    t0 = time.perf_counter()
    raw = state.get("raw_markdown") or ""
    recs = dedupe_recommendations(parse_strategy_markdown(raw))
    timings = dict(state.get("timings") or {})
    timings["parse_dedupe_s"] = time.perf_counter() - t0
    return {"recommendations": recs, "timings": timings}


def grounded_check_node(state: StrategyState) -> dict[str, Any]:
    recs = list(state.get("recommendations") or [])
    grounded = filter_grounded_recommendations(recs)
    retry = int(state.get("grounded_retry_count") or 0)
    if len(grounded) >= 3 or retry >= 1:
        return {
            "recommendations": grounded or recs,
            "grounded_retry_count": retry,
            "needs_regenerate": False,
        }
    return {
        "recommendations": recs,
        "grounded_retry_count": retry + 1,
        "needs_regenerate": True,
    }


def route_after_grounded_check(state: StrategyState) -> str:
    if state.get("needs_regenerate"):
        return "generate_strategy"
    return "__end__"
