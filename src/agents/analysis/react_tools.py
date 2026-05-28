from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import tool

from src.chains.section_label_single_chains import (
    label_one_dtc,
    label_one_dtc_ip,
    label_one_ip,
    validate_label_dtc,
    validate_label_dtc_ip,
    validate_label_ip,
    VALIDATOR_ACCEPT,
    VALIDATOR_REJECT,
)
from src.chains.section_summary_chain import summarize_section_from_context
from src.config.analysis_mode import AnalysisMode
from src.services.file_search_retrieval import RetrievalError
from src.services.neutral_summary_cache import get_neutral_row, put_neutral_row
from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
)
from src.tools.file_search_tools import retrieve_subsection_evidence_tool


@dataclass
class ReactRunContext:
    analysis_mode: AnalysisMode
    target_categories: set[str]
    file_search_store_name: str
    source_filename: str
    gemini_model_name: str
    synthesis_model_name: str
    pdf_hash: str | None
    manifest_by_key: dict[str, dict[str, str]]
    manifest_order: list[str]
    accepted_categories: set[str] = field(default_factory=set)
    fetched_rows: dict[str, dict[str, str]] = field(default_factory=dict)
    judged_rows_by_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    validator_outcomes_by_key: dict[str, dict[str, str]] = field(
        default_factory=dict
    )
    errors: list[str] = field(default_factory=list)


_ACTIVE_REACT_CONTEXT: ContextVar[ReactRunContext | None] = ContextVar(
    "ACTIVE_REACT_CONTEXT",
    default=None,
)


def build_react_run_context(
    *,
    analysis_mode: AnalysisMode,
    file_search_store_name: str,
    source_filename: str,
    gemini_model_name: str,
    synthesis_model_name: str,
    pdf_hash: str | None,
    manifest: list[dict[str, str]],
) -> ReactRunContext:
    manifest_by_key: dict[str, dict[str, str]] = {}
    manifest_order: list[str] = []
    for row in manifest:
        key = str(row.get("subsection_key") or "").strip()
        title = str(row.get("display_title") or "").strip()
        if not key or not title:
            continue
        manifest_by_key[key] = {
            "subsection_key": key,
            "display_title": title,
        }
        manifest_order.append(key)
    return ReactRunContext(
        analysis_mode=analysis_mode,
        target_categories=_target_categories_for_mode(analysis_mode),
        file_search_store_name=file_search_store_name,
        source_filename=source_filename,
        gemini_model_name=gemini_model_name,
        synthesis_model_name=synthesis_model_name,
        pdf_hash=pdf_hash,
        manifest_by_key=manifest_by_key,
        manifest_order=manifest_order,
    )


def _target_categories_for_mode(mode: AnalysisMode) -> set[str]:
    if mode == AnalysisMode.DTC_ONLY:
        return {CATEGORY_DTC}
    if mode == AnalysisMode.IP_ONLY:
        return {CATEGORY_IP}
    return {CATEGORY_DTC, CATEGORY_IP}


def missing_target_categories(ctx: ReactRunContext) -> list[str]:
    return sorted(cat for cat in ctx.target_categories if cat not in ctx.accepted_categories)


def is_target_satisfied(ctx: ReactRunContext) -> bool:
    return not missing_target_categories(ctx)


def materialize_judged_rows(ctx: ReactRunContext) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ctx.manifest_order:
        row = ctx.judged_rows_by_key.get(key)
        if row:
            rows.append(dict(row))
    return rows


@contextmanager
def activate_react_tool_context(ctx: ReactRunContext):
    token = _ACTIVE_REACT_CONTEXT.set(ctx)
    try:
        yield
    finally:
        _ACTIVE_REACT_CONTEXT.reset(token)


def _require_context() -> ReactRunContext:
    ctx = _ACTIVE_REACT_CONTEXT.get()
    if ctx is None:
        raise RuntimeError("ReAct tool used outside an active run context.")
    return ctx


def _fetch_summary_payload(
    ctx: ReactRunContext, subsection_key: str
) -> dict[str, str]:
    row = ctx.manifest_by_key.get(subsection_key)
    if row is None:
        raise ValueError(f"Unknown subsection_key: {subsection_key!r}")
    if subsection_key in ctx.fetched_rows:
        return ctx.fetched_rows[subsection_key]

    title = row["display_title"]
    evidence = ""
    summary = ""

    if ctx.pdf_hash:
        cached = get_neutral_row(
            ctx.pdf_hash,
            subsection_key,
            ctx.gemini_model_name,
            ctx.synthesis_model_name,
        )
        if cached is not None:
            evidence = cached["evidence"]
            summary = cached["summary"]

    if not summary:
        try:
            evidence = retrieve_subsection_evidence_tool.invoke(
                {
                    "file_search_store_name": ctx.file_search_store_name,
                    "subsection_key": subsection_key,
                    "display_title": title,
                    "source_filename": ctx.source_filename,
                    "model_name": ctx.gemini_model_name,
                }
            )
        except RetrievalError as exc:
            ctx.errors.append(f"{title}: {exc}")
            evidence = ""

        if evidence.strip():
            summary = summarize_section_from_context(
                section_name=title,
                grounded_context=evidence,
                model_name=ctx.synthesis_model_name,
            )
            if ctx.pdf_hash:
                put_neutral_row(
                    ctx.pdf_hash,
                    subsection_key,
                    ctx.gemini_model_name,
                    ctx.synthesis_model_name,
                    evidence,
                    summary,
                )
        else:
            summary = "_Retrieval failed._"

    payload = {
        "subsection_key": subsection_key,
        "display_title": title,
        "summary": summary,
        "evidence": evidence,
    }
    ctx.fetched_rows[subsection_key] = payload
    return payload


def _judge_and_store(
    *,
    subsection_key: str,
    judge_fn: Any,
    validator_fn: Any,
) -> dict[str, Any]:
    ctx = _require_context()
    payload = _fetch_summary_payload(ctx, subsection_key)
    label = str(
        judge_fn(
            title=payload["display_title"],
            summary=payload["summary"],
            model_name=ctx.synthesis_model_name,
        )
    )
    decision = VALIDATOR_REJECT
    reason = ""
    try:
        validator_result = validator_fn(
            title=payload["display_title"],
            summary=payload["summary"],
            predicted_label=label,
            model_name=ctx.synthesis_model_name,
        )
        decision = str(validator_result.get("decision") or "").strip().upper()
        reason = str(validator_result.get("reason") or "").strip()
        if decision not in {VALIDATOR_ACCEPT, VALIDATOR_REJECT}:
            decision = VALIDATOR_REJECT
            if not reason:
                reason = "Validator returned unsupported decision."
    except Exception as exc:
        decision = VALIDATOR_REJECT
        reason = f"Validator failed: {exc}"
        ctx.errors.append(
            f"{payload['display_title']}: validator pass failed ({exc})."
        )

    accepted_for_target = False
    if (
        decision == VALIDATOR_ACCEPT
        and label in ctx.target_categories
        and label not in ctx.accepted_categories
    ):
        accepted_for_target = True
        ctx.accepted_categories.add(label)
    elif (
        decision == VALIDATOR_ACCEPT
        and label in ctx.target_categories
        and label in ctx.accepted_categories
        and not reason
    ):
        reason = "Category target already satisfied by an earlier subsection."

    ctx.validator_outcomes_by_key[subsection_key] = {
        "decision": decision,
        "reason": reason,
    }
    judged_row = dict(payload)
    judged_row["category"] = label
    judged_row["validator_decision"] = decision
    judged_row["validator_reason"] = reason
    judged_row["accepted_for_target"] = accepted_for_target
    ctx.judged_rows_by_key[subsection_key] = judged_row
    return {
        "subsection_key": subsection_key,
        "label": label,
        "validator_decision": decision,
        "validator_reason": reason,
        "accepted_for_target": accepted_for_target,
        "missing_targets": missing_target_categories(ctx),
        "target_satisfied": is_target_satisfied(ctx),
    }


@tool
def list_pending_subsections() -> list[dict[str, str]]:
    """Return subsection keys/titles not judged yet in this run."""
    ctx = _require_context()
    pending: list[dict[str, str]] = []
    for key in ctx.manifest_order:
        if key in ctx.judged_rows_by_key:
            continue
        row = ctx.manifest_by_key[key]
        pending.append(
            {
                "subsection_key": key,
                "display_title": row["display_title"],
            }
        )
    return pending


@tool
def fetch_and_summarize_subsection(subsection_key: str) -> dict[str, str]:
    """Retrieve grounded evidence and produce a neutral summary for one key."""
    ctx = _require_context()
    payload = _fetch_summary_payload(ctx, subsection_key)
    return {
        "subsection_key": payload["subsection_key"],
        "display_title": payload["display_title"],
        "summary": payload["summary"],
        "evidence_excerpt": payload["evidence"][:1200],
    }


@tool
def judge_section_dtc(subsection_key: str) -> dict[str, Any]:
    """Label one summary as DTC or OTHER."""
    return _judge_and_store(
        subsection_key=subsection_key,
        judge_fn=label_one_dtc,
        validator_fn=validate_label_dtc,
    )


@tool
def judge_section_ip(subsection_key: str) -> dict[str, Any]:
    """Label one summary as IP or OTHER."""
    return _judge_and_store(
        subsection_key=subsection_key,
        judge_fn=label_one_ip,
        validator_fn=validate_label_ip,
    )


@tool
def judge_section_dtc_ip(subsection_key: str) -> dict[str, Any]:
    """Label one summary as DTC, IP, or OTHER."""
    return _judge_and_store(
        subsection_key=subsection_key,
        judge_fn=label_one_dtc_ip,
        validator_fn=validate_label_dtc_ip,
    )


@tool
def finish() -> str:
    """Signal completion once mode target categories are satisfied."""
    ctx = _require_context()
    if is_target_satisfied(ctx):
        return "FINISH_OK"

    pending = list_pending_subsections.invoke({})
    missing = missing_target_categories(ctx)
    if not pending:
        if missing:
            ctx.errors.append(
                "Reached end of manifest before satisfying mode targets: "
                + ", ".join(missing)
            )
        return "FINISH_OK"

    if pending:
        keys = ", ".join(item["subsection_key"] for item in pending[:12])
        suffix = "..." if len(pending) > 12 else ""
        missing_txt = ", ".join(missing) if missing else "none"
        return (
            "Cannot finish yet. Missing targets: "
            f"{missing_txt}. Remaining subsection_keys: {keys}{suffix}"
        )
    return "FINISH_OK"

