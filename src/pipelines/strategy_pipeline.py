from __future__ import annotations

from src.agents.strategy.postprocess import (
    Recommendation,
    dedupe_recommendations,
    parse_strategy_markdown,
)
from src.graphs.strategy_graph import run_strategy_graph
from src.services.strategy_bundle import build_strategy_bundle

__all__ = (
    "Recommendation",
    "dedupe_recommendations",
    "parse_strategy_markdown",
    "run_strategy_pipeline",
    "run_strategy",
)


def run_strategy_pipeline(
    analysis_bundle: str,
    *,
    model_name: str | None = None,
    timings_out: dict[str, float] | None = None,
) -> list[Recommendation]:
    return run_strategy_graph(
        analysis_bundle=analysis_bundle,
        model_name=model_name,
        timings_out=timings_out,
    )


def run_strategy(
    dtc_results: dict | None = None,
    ip_results: dict | None = None,
    section_results: dict | None = None,
    *,
    analysis_bundle: str | None = None,
    model_name: str | None = None,
    timings_out: dict[str, float] | None = None,
) -> list[Recommendation]:
    """Strategy LangGraph from analysis dicts or a pre-built bundle."""
    if analysis_bundle is not None:
        return run_strategy_graph(
            analysis_bundle=analysis_bundle,
            model_name=model_name,
            timings_out=timings_out,
        )
    return run_strategy_graph(
        dtc_results=dtc_results,
        ip_results=ip_results,
        section_results=section_results,
        model_name=model_name,
        timings_out=timings_out,
    )
