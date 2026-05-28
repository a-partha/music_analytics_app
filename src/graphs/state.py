from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AnalysisState(TypedDict, total=False):
    file_search_store_name: str
    manifest: list[dict[str, str]]
    source_filename: str
    pdf_hash: str | None
    gemini_model_name: str | None
    synthesis_model_name: str | None
    run_profile: str
    analysis_mode: str | None
    react_trace_callback: Any
    neutral_rows: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    react_messages: Annotated[list[dict[str, Any]], operator.add]
    judged_rows: Annotated[list[dict[str, Any]], operator.add]
    labeled_rows: list[dict[str, Any]]
    dtc_synthesis: str
    ip_synthesis: str
    dtc_results: dict[str, Any]
    ip_results: dict[str, Any]
    section_results: dict[str, dict[str, str]]
    timings: dict[str, float]


class SubsectionWorkState(TypedDict, total=False):
    row: dict[str, str]
    row_index: int
    file_search_store_name: str
    source_filename: str
    gemini_model_name: str
    synthesis_model_name: str
    pdf_hash: str | None
    neutral_rows: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]


class StrategyState(TypedDict, total=False):
    dtc_results: dict[str, Any] | None
    ip_results: dict[str, Any] | None
    section_results: dict[str, dict[str, str]] | None
    analysis_bundle: str
    model_name: str | None
    raw_markdown: str
    recommendations: list[dict[str, str]]
    grounded_retry_count: int
    needs_regenerate: bool
    timings: dict[str, float]
