from __future__ import annotations

from collections.abc import Callable
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.chains.section_label_chain import label_neutral_rows_with_synthesis
from src.chains.section_summary_chain import summarize_section_from_context
from src.services.file_search_retrieval import (
    RetrievalError,
    retrieve_dtc_evidence,
    retrieve_ip_evidence,
    retrieve_section_context,
    retrieve_subsection_evidence,
)
from src.services.neutral_summary_cache import (
    get_neutral_row,
    put_neutral_row,
    resolved_gemini_model,
    resolved_synthesis_model,
)
from src.services.strategy_bundle import pipeline_results_from_labeled_neutral_rows

PASS1_SECTION_NAMES = (
    "Midyear Metrics",
    "Streaming Atlas",
    "Import / Export",
    "Engagement Horizon",
    "Artist Spectrum",
    "Future in Focus",
    "Midyear Charts",
)

DEFAULT_SECTION_NAMES = ("Midyear Metrics",)

DTC_SECTION_NAMES = (
    "Engagement Horizon",
)

IP_SECTION_NAMES = (
    "Import / Export",
    "Streaming Atlas",
)

OTHER_SECTION_NAMES = (
    "Midyear Metrics",
    "Artist Spectrum",
    "Future in Focus",
    "Midyear Charts",
)


def _resolve_neutral_summary_workers() -> int:
    raw = os.getenv("NEUTRAL_SUMMARY_MAX_WORKERS", "").strip() or "4"
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def run_section_summary_pipeline(
    file_search_store_name: str,
    section_names: tuple[str, ...] = DEFAULT_SECTION_NAMES,
    source_filename: str | None = None,
    model_name: str | None = None,
    *,
    splitter_mode: str = "dynamic",
    other_subsection_key: str | None = None,
) -> dict[str, dict[str, str]]:
    if splitter_mode == "legacy":
        summaries: dict[str, dict[str, str]] = {}
        for section_name in section_names:
            context = retrieve_section_context(
                file_search_store_name=file_search_store_name,
                section_name=section_name,
                source_filename=source_filename,
                model_name=model_name,
            )
            summary = summarize_section_from_context(
                section_name=section_name,
                grounded_context=context,
                model_name=model_name,
            )
            summaries[section_name] = {
                "summary": summary,
                "evidence": context,
            }
        return summaries

    if other_subsection_key is None:
        raise RetrievalError(
            "Dynamic splitter mode requires other_subsection_key for "
            "single-section summaries."
        )
    if len(section_names) != 1:
        raise RetrievalError(
            "Dynamic mode expects exactly one display title in section_names."
        )
    display_title = section_names[0]
    context = retrieve_subsection_evidence(
        file_search_store_name=file_search_store_name,
        subsection_key=other_subsection_key,
        display_title=display_title,
        source_filename=source_filename,
        model_name=model_name,
    )
    summary = summarize_section_from_context(
        section_name=display_title,
        grounded_context=context,
        model_name=model_name,
    )
    return {
        display_title: {
            "summary": summary,
            "evidence": context,
        }
    }


def run_section_retrieval_pipeline(
    file_search_store_name: str,
    section_names: tuple[str, ...] = DEFAULT_SECTION_NAMES,
    source_filename: str | None = None,
    model_name: str | None = None,
    *,
    splitter_mode: str = "dynamic",
    other_subsection_key: str | None = None,
) -> dict[str, str]:
    if splitter_mode == "legacy":
        evidence_map: dict[str, str] = {}
        for section_name in section_names:
            evidence_map[section_name] = retrieve_section_context(
                file_search_store_name=file_search_store_name,
                section_name=section_name,
                source_filename=source_filename,
                model_name=model_name,
            )
        return evidence_map

    if other_subsection_key is None or len(section_names) != 1:
        raise RetrievalError(
            "Dynamic mode requires one section name and other_subsection_key."
        )
    display_title = section_names[0]
    text = retrieve_subsection_evidence(
        file_search_store_name=file_search_store_name,
        subsection_key=other_subsection_key,
        display_title=display_title,
        source_filename=source_filename,
        model_name=model_name,
    )
    return {display_title: text}


def _neutral_row_one(
    idx: int,
    m: dict[str, str],
    file_search_store_name: str,
    source_filename: str,
    gemini_resolved: str,
    synthesis_resolved: str,
    pdf_hash: str | None,
) -> tuple[int, dict[str, Any]]:
    key = m.get("subsection_key") or ""
    title = m.get("display_title") or ""

    if pdf_hash:
        cached = get_neutral_row(
            pdf_hash, key, gemini_resolved, synthesis_resolved
        )
        if cached is not None:
            return (
                idx,
                {
                    "subsection_key": key,
                    "display_title": title,
                    "summary": cached["summary"],
                    "evidence": cached["evidence"],
                },
            )

    try:
        evidence = retrieve_subsection_evidence(
            file_search_store_name=file_search_store_name,
            subsection_key=key,
            display_title=title,
            source_filename=source_filename,
            model_name=gemini_resolved,
        )
    except RetrievalError:
        evidence = ""

    summary = summarize_section_from_context(
        section_name=title,
        grounded_context=evidence,
        model_name=synthesis_resolved,
    )

    if pdf_hash:
        put_neutral_row(
            pdf_hash,
            key,
            gemini_resolved,
            synthesis_resolved,
            evidence,
            summary,
        )

    return (
        idx,
        {
            "subsection_key": key,
            "display_title": title,
            "summary": summary,
            "evidence": evidence,
        },
    )


def run_neutral_section_summaries_for_manifest(
    file_search_store_name: str,
    manifest: list[dict[str, str]],
    source_filename: str | None = None,
    gemini_model_name: str | None = None,
    synthesis_model_name: str | None = None,
    pdf_hash: str | None = None,
) -> list[dict[str, Any]]:
    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval."
        )

    gemini_resolved = resolved_gemini_model(gemini_model_name)
    synthesis_resolved = resolved_synthesis_model(synthesis_model_name)

    indexed_valid: list[tuple[int, dict[str, str]]] = []
    for idx, m in enumerate(manifest):
        sk = m.get("subsection_key") or ""
        dt = m.get("display_title") or ""
        if not sk or not dt:
            continue
        indexed_valid.append((idx, m))

    neutral: list[dict[str, Any]] = []
    if not indexed_valid:
        return neutral

    max_workers = _resolve_neutral_summary_workers()

    results_by_idx: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list = []
        for idx, m in indexed_valid:
            futures.append(
                executor.submit(
                    _neutral_row_one,
                    idx,
                    m,
                    file_search_store_name,
                    source_filename,
                    gemini_resolved,
                    synthesis_resolved,
                    pdf_hash,
                )
            )
        for fut in futures:
            row_idx, row = fut.result()
            results_by_idx[row_idx] = row

    for idx, _ in indexed_valid:
        neutral.append(results_by_idx[idx])

    return neutral


def run_dynamic_summarize_and_classify(
    file_search_store_name: str,
    manifest: list[dict[str, str]],
    source_filename: str | None = None,
    gemini_model_name: str | None = None,
    synthesis_model_name: str | None = None,
    pdf_hash: str | None = None,
    *,
    analysis_mode: str | None = None,
    timings_out: dict[str, float] | None = None,
    react_messages_out: list[dict[str, Any]] | None = None,
    react_trace_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    errors_out: list[str] | None = None,
    run_profile: str | None = None,
    dev_mode_flag: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
]:
    """
    LangGraph analysis agent: filter manifest → parallel subsection summaries
    → label → DTC/IP synthesis → assemble bundle-shaped dicts.

    If ``timings_out`` is passed, it is filled with graph step timings.
    """
    from src.graphs.analysis_graph import run_analysis_graph

    return run_analysis_graph(
        file_search_store_name=file_search_store_name,
        manifest=manifest,
        source_filename=source_filename,
        pdf_hash=pdf_hash,
        gemini_model_name=gemini_model_name,
        synthesis_model_name=synthesis_model_name,
        run_profile=run_profile,
        analysis_mode=analysis_mode,
        dev_mode_flag=dev_mode_flag,
        timings_out=timings_out,
        react_messages_out=react_messages_out,
        react_trace_callback=react_trace_callback,
        errors_out=errors_out,
    )


def run_analysis(
    file_search_store_name: str,
    manifest: list[dict[str, str]],
    source_filename: str | None = None,
    gemini_model_name: str | None = None,
    synthesis_model_name: str | None = None,
    pdf_hash: str | None = None,
    *,
    analysis_mode: str | None = None,
    timings_out: dict[str, float] | None = None,
    react_messages_out: list[dict[str, Any]] | None = None,
    react_trace_callback: Callable[[list[dict[str, Any]]], None] | None = None,
    errors_out: list[str] | None = None,
    run_profile: str | None = None,
    dev_mode_flag: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
]:
    """Public facade for the analysis LangGraph."""
    return run_dynamic_summarize_and_classify(
        file_search_store_name=file_search_store_name,
        manifest=manifest,
        source_filename=source_filename,
        gemini_model_name=gemini_model_name,
        synthesis_model_name=synthesis_model_name,
        pdf_hash=pdf_hash,
        analysis_mode=analysis_mode,
        timings_out=timings_out,
        react_messages_out=react_messages_out,
        react_trace_callback=react_trace_callback,
        errors_out=errors_out,
        run_profile=run_profile,
        dev_mode_flag=dev_mode_flag,
    )


def run_dtc_insights_pipeline(
    file_search_store_name: str,
    section_names: tuple[str, ...] = DTC_SECTION_NAMES,
    source_filename: str | None = None,
    model_name: str | None = None,
    *,
    splitter_mode: str = "dynamic",
) -> dict[str, object]:
    if splitter_mode != "legacy":
        raise RetrievalError(
            "In dynamic mode use summarize & classify for the full report "
            "instead of the per-category DTC pipeline."
        )
    from src.chains.dtc_insights_chain import summarize_dtc_insights

    evidence_by_section: dict[str, str] = {}
    for section_name in section_names:
        try:
            evidence_by_section[section_name] = retrieve_dtc_evidence(
                file_search_store_name=file_search_store_name,
                section_name=section_name,
                source_filename=source_filename,
                model_name=model_name,
            )
        except RetrievalError:
            evidence_by_section[section_name] = ""

    if not any(evidence_by_section.values()):
        raise RetrievalError(
            "No DTC evidence found across any of: "
            f"{', '.join(section_names)}."
        )

    combined_evidence = "\n\n".join(
        f"=== {section} ===\n{evidence}"
        for section, evidence in evidence_by_section.items()
        if evidence
    )
    insights = summarize_dtc_insights(
        grounded_context=combined_evidence,
        model_name=model_name,
    )
    return {
        "insights": insights,
        "evidence_by_section": evidence_by_section,
    }


def run_ip_insights_pipeline(
    file_search_store_name: str,
    section_names: tuple[str, ...] = IP_SECTION_NAMES,
    source_filename: str | None = None,
    model_name: str | None = None,
    *,
    splitter_mode: str = "dynamic",
) -> dict[str, object]:
    if splitter_mode != "legacy":
        raise RetrievalError(
            "In dynamic mode use summarize & classify for the full report "
            "instead of the per-category IP pipeline."
        )
    from src.chains.ip_insights_chain import summarize_ip_insights

    evidence_by_section: dict[str, str] = {}
    for section_name in section_names:
        try:
            evidence_by_section[section_name] = retrieve_ip_evidence(
                file_search_store_name=file_search_store_name,
                section_name=section_name,
                source_filename=source_filename,
                model_name=model_name,
            )
        except RetrievalError:
            evidence_by_section[section_name] = ""

    if not any(evidence_by_section.values()):
        raise RetrievalError(
            "No IP evidence found across any of: "
            f"{', '.join(section_names)}."
        )

    combined_evidence = "\n\n".join(
        f"=== {section} ===\n{evidence}"
        for section, evidence in evidence_by_section.items()
        if evidence
    )
    insights = summarize_ip_insights(
        grounded_context=combined_evidence,
        model_name=model_name,
    )
    return {
        "insights": insights,
        "evidence_by_section": evidence_by_section,
    }
