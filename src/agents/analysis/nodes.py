from __future__ import annotations

import time
from typing import Any

from src.chains.section_label_chain import label_neutral_rows_with_synthesis
from src.chains.section_summary_chain import summarize_section_from_context
from src.config.run_profiles import RunProfile, profile_from_state_value
from src.graphs.state import AnalysisState, SubsectionWorkState
from src.services.file_search_retrieval import RetrievalError
from src.services.manifest_filter import (
    apply_run_profile_to_manifest,
    limit_labeled_rows_one_per_category,
)
from src.services.neutral_summary_cache import (
    get_neutral_row,
    put_neutral_row,
    resolved_gemini_model,
    resolved_synthesis_model,
)
from src.services.strategy_bundle import pipeline_results_from_labeled_neutral_rows
from src.tools.file_search_tools import (
    retrieve_subsection_evidence_tool,
)

def filter_manifest_node(state: AnalysisState) -> dict[str, Any]:
    t0 = time.perf_counter()
    profile = profile_from_state_value(state.get("run_profile"))
    manifest = apply_run_profile_to_manifest(
        list(state.get("manifest") or []),
        profile,
    )
    timings = dict(state.get("timings") or {})
    timings["filter_manifest_s"] = time.perf_counter() - t0
    timings["run_profile"] = profile.value
    if not manifest:
        raise RetrievalError(
            "No sections in manifest to summarize (empty or invalid manifest)."
        )
    return {"manifest": manifest, "timings": timings}


def summarize_subsection_node(state: SubsectionWorkState) -> dict[str, Any]:
    row = state.get("row") or {}
    key = row.get("subsection_key") or ""
    title = row.get("display_title") or ""
    raw_row_index = state.get("row_index", -1)
    try:
        row_index = int(raw_row_index)
    except (TypeError, ValueError):
        row_index = -1
    if not key or not title:
        return {"errors": [f"Skipped invalid manifest row: {row!r}"]}

    store = state["file_search_store_name"]
    source = state["source_filename"]
    gemini = state["gemini_model_name"]
    synthesis = state["synthesis_model_name"]
    pdf_hash = state.get("pdf_hash")

    if pdf_hash:
        cached = get_neutral_row(pdf_hash, key, gemini, synthesis)
        if cached is not None:
            return {
                "neutral_rows": [
                    {
                        "subsection_key": key,
                        "display_title": title,
                        "summary": cached["summary"],
                        "evidence": cached["evidence"],
                        "__manifest_index": row_index,
                    }
                ]
            }

    evidence = ""
    try:
        evidence = retrieve_subsection_evidence_tool.invoke(
            {
                "file_search_store_name": store,
                "subsection_key": key,
                "display_title": title,
                "source_filename": source,
                "model_name": gemini,
            }
        )
    except RetrievalError as exc:
        return {
            "neutral_rows": [
                {
                    "subsection_key": key,
                    "display_title": title,
                    "summary": "_Retrieval failed._",
                    "evidence": "",
                    "__manifest_index": row_index,
                }
            ],
            "errors": [f"{title}: {exc}"],
        }

    if not evidence.strip():
        return {
            "neutral_rows": [
                {
                    "subsection_key": key,
                    "display_title": title,
                    "summary": "_Retrieval failed._",
                    "evidence": "",
                    "__manifest_index": row_index,
                }
            ],
            "errors": [f"{title}: no evidence retrieved from file search."],
        }

    summary = summarize_section_from_context(
        section_name=title,
        grounded_context=evidence,
        model_name=synthesis,
    )

    if pdf_hash:
        put_neutral_row(pdf_hash, key, gemini, synthesis, evidence, summary)

    return {
        "neutral_rows": [
            {
                "subsection_key": key,
                "display_title": title,
                "summary": summary,
                "evidence": evidence,
                "__manifest_index": row_index,
            }
        ]
    }


def label_sections_node(state: AnalysisState) -> dict[str, Any]:
    t0 = time.perf_counter()
    profile = profile_from_state_value(state.get("run_profile"))
    rows = list(state.get("neutral_rows") or [])
    if not rows:
        raise RetrievalError(
            "No sections in manifest to summarize (empty or invalid manifest)."
        )
    try:
        labeled = label_neutral_rows_with_synthesis(
            rows, model_name=state.get("synthesis_model_name")
        )
    except ValueError as exc:
        timings = dict(state.get("timings") or {})
        timings["batch_label_s"] = time.perf_counter() - t0
        timings["label_parse_failed"] = 1.0
        raise RetrievalError(f"Section labeling failed: {exc}") from exc
    timings = dict(state.get("timings") or {})
    if profile == RunProfile.DEV_ONE_PER_CATEGORY:
        timings["dev_rows_before"] = float(len(labeled))
        labeled = limit_labeled_rows_one_per_category(labeled)
        timings["dev_rows_after"] = float(len(labeled))
    timings["batch_label_s"] = time.perf_counter() - t0
    return {"labeled_rows": labeled, "timings": timings}




def assemble_outputs_node(state: AnalysisState) -> dict[str, Any]:
    t0 = time.perf_counter()
    labeled = list(state.get("labeled_rows") or [])
    dtc_r, ip_r, sec_r = pipeline_results_from_labeled_neutral_rows(labeled)

    timings = dict(state.get("timings") or {})
    timings["bundle_shape_s"] = time.perf_counter() - t0
    return {
        "dtc_results": dtc_r,
        "ip_results": ip_r,
        "section_results": sec_r,
        "timings": timings,
    }


def prepare_subsection_fan_out(state: AnalysisState) -> list[Any]:
    from langgraph.types import Send

    gemini = resolved_gemini_model(state.get("gemini_model_name"))
    synthesis = resolved_synthesis_model(state.get("synthesis_model_name"))
    manifest = state.get("manifest") or []
    if not manifest:
        return []

    sends: list[Any] = []
    for idx, row in enumerate(manifest):
        sk = row.get("subsection_key") or ""
        dt = row.get("display_title") or ""
        if not sk or not dt:
            continue
        sends.append(
            Send(
                "summarize_subsection",
                {
                    "row": row,
                    "file_search_store_name": state["file_search_store_name"],
                    "source_filename": state["source_filename"],
                    "gemini_model_name": gemini,
                    "synthesis_model_name": synthesis,
                    "pdf_hash": state.get("pdf_hash"),
                    "row_index": idx,
                },
            )
        )
    return sends
