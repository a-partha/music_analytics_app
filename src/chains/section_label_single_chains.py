from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.services.langchain_llm import get_langchain_gemini_model
from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
)

VALIDATOR_ACCEPT = "ACCEPT"
VALIDATOR_REJECT = "REJECT"
_VALIDATOR_DECISIONS = frozenset({VALIDATOR_ACCEPT, VALIDATOR_REJECT})


def _normalize_label(raw: str) -> str:
    normalized = str(raw or "").strip().upper()
    if normalized == "DTC":
        return CATEGORY_DTC
    if normalized == "IP":
        return CATEGORY_IP
    if normalized in ("OTHER", "OTHER_SECTIONS", "MISC"):
        return CATEGORY_OTHER
    raise ValueError(f"Unsupported category label: {raw!r}")


def parse_single_label_json(
    llm_text: str,
    *,
    allowed: frozenset[str],
) -> str:
    cleaned = (llm_text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    payload: object
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = cleaned

    label_raw: str | None = None
    if isinstance(payload, dict):
        maybe = payload.get("label")
        if isinstance(maybe, str):
            label_raw = maybe
    elif isinstance(payload, str):
        label_raw = payload

    if not label_raw:
        raise ValueError("Expected JSON object with a string `label` field.")

    label = _normalize_label(label_raw)
    if label not in allowed:
        raise ValueError(
            f"Label {label!r} is outside allowed set {sorted(allowed)!r}."
        )
    return label


def parse_label_validator_json(llm_text: str) -> dict[str, str]:
    cleaned = (llm_text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    payload: object
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = cleaned

    decision_raw: str | None = None
    reason = ""
    if isinstance(payload, dict):
        maybe = payload.get("decision")
        if isinstance(maybe, str):
            decision_raw = maybe
        maybe_reason = payload.get("reason")
        if isinstance(maybe_reason, str):
            reason = maybe_reason.strip()
    elif isinstance(payload, str):
        decision_raw = payload

    decision = str(decision_raw or "").strip().upper()
    if decision not in _VALIDATOR_DECISIONS:
        raise ValueError(
            "Expected validator decision ACCEPT or REJECT, "
            f"got {decision_raw!r}."
        )
    return {"decision": decision, "reason": reason}


def _label_one_row(
    *,
    title: str,
    summary: str,
    category_policy: str,
    allowed: frozenset[str],
    model_name: str | None = None,
) -> str:
    prompt = ChatPromptTemplate.from_template(
        "You are labeling ONE document subsection into an allowed category.\n"
        "{category_policy}\n\n"
        "Input subsection:\n"
        "- display_title: {title}\n"
        "- neutral_summary: {summary}\n\n"
        "Return STRICT JSON only in this shape:\n"
        '{{"label": "<one_of_allowed_labels>"}}\n'
        "No other text.\n"
    )

    llm = get_langchain_gemini_model(model_name=model_name, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "category_policy": category_policy,
            "title": title,
            "summary": summary[:8000],
        }
    ).strip()
    return parse_single_label_json(raw, allowed=allowed)


def _validate_predicted_label(
    *,
    title: str,
    summary: str,
    predicted_label: str,
    mode_policy: str,
    allowed_labels: frozenset[str],
    model_name: str | None = None,
) -> dict[str, str]:
    normalized_predicted = _normalize_label(predicted_label)
    if normalized_predicted not in allowed_labels:
        return {
            "decision": VALIDATOR_REJECT,
            "reason": (
                "Predicted label is outside mode-allowed label set."
            ),
        }

    prompt = ChatPromptTemplate.from_template(
        "You are validating a single category label predicted for a report "
        "subsection.\n"
        "{mode_policy}\n"
        "Input subsection:\n"
        "- display_title: {title}\n"
        "- neutral_summary: {summary}\n"
        "- predicted_label: {predicted_label}\n"
        "Return STRICT JSON only in this shape:\n"
        '{{"decision": "ACCEPT|REJECT", "reason": "<short reason>"}}\n'
        "No other text.\n"
    )
    llm = get_langchain_gemini_model(model_name=model_name, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "mode_policy": mode_policy,
            "title": title,
            "summary": summary[:8000],
            "predicted_label": normalized_predicted,
        }
    ).strip()
    return parse_label_validator_json(raw)


def label_one_dtc(
    *,
    title: str,
    summary: str,
    model_name: str | None = None,
) -> str:
    return _label_one_row(
        title=title,
        summary=summary,
        model_name=model_name,
        allowed=frozenset({CATEGORY_DTC, CATEGORY_OTHER}),
        category_policy=(
            "Allowed labels: DTC or OTHER.\n"
            "Choose DTC only if the dominant theme is Direct-to-Consumer "
            "such as fan engagement, superfans, short-form video, "
            "social media/gaming music, podcasts, merch, ticketing, direct "
            "artist–fan monetization, and so on.\n"
            "If the subsection is not dominantly DTC, return OTHER."
        ),
    )


def label_one_ip(
    *,
    title: str,
    summary: str,
    model_name: str | None = None,
) -> str:
    return _label_one_row(
        title=title,
        summary=summary,
        model_name=model_name,
        allowed=frozenset({CATEGORY_IP, CATEGORY_OTHER}),
        category_policy=(
            "Allowed labels: IP or OTHER.\n"
            "Choose IP only if the dominant theme is Intellectual Property "
            "such as catalog age/splits, export flows, local vs foreign "
            "streaming, territory charts, cross-border genre trends, and so on.\n"
            "If the subsection is not dominantly IP, return OTHER."
        ),
    )


def label_one_dtc_ip(
    *,
    title: str,
    summary: str,
    model_name: str | None = None,
) -> str:
    return _label_one_row(
        title=title,
        summary=summary,
        model_name=model_name,
        allowed=frozenset({CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER}),
        category_policy=(
            "Allowed labels: DTC, IP, OTHER.\n"
            "DTC: Direct-to-Consumer theme such as fan engagement, superfans, "
            "short-form video, social media/gaming music, podcasts, merch, "
            "ticketing, direct artist–fan monetization, and so on.\n"
            "IP: Intellectual Property theme such as catalog age/splits, "
            "export flows, local vs foreign streaming, territory charts, "
            "cross-border genre trends, and so on.\n"
            "OTHER: everything else. If mixed, choose the dominant theme."
        ),
    )


def validate_label_dtc(
    *,
    title: str,
    summary: str,
    predicted_label: str,
    model_name: str | None = None,
) -> dict[str, str]:
    return _validate_predicted_label(
        title=title,
        summary=summary,
        predicted_label=predicted_label,
        model_name=model_name,
        allowed_labels=frozenset({CATEGORY_DTC, CATEGORY_OTHER}),
        mode_policy=(
            "Mode: DTC_ONLY.\n"
            "Allowed predicted labels: DTC or OTHER.\n"
            "Category definitions (same as judge):\n"
            "- DTC: Direct-to-Consumer theme such as fan engagement, superfans, "
            "short-form video, social media/gaming music, podcasts, merch, "
            "ticketing, direct artist–fan monetization, and so on. Use DTC "
            "only if the subsection is dominantly Direct-to-Consumer themed.\n"
            "- OTHER: the subsection is not dominantly Direct-to-Consumer themed.\n"
            "ACCEPT predicted_label unless it clearly violates these "
            "definitions. REJECT only when the label is clearly wrong."
        ),
    )


def validate_label_ip(
    *,
    title: str,
    summary: str,
    predicted_label: str,
    model_name: str | None = None,
) -> dict[str, str]:
    return _validate_predicted_label(
        title=title,
        summary=summary,
        predicted_label=predicted_label,
        model_name=model_name,
        allowed_labels=frozenset({CATEGORY_IP, CATEGORY_OTHER}),
        mode_policy=(
            "Mode: IP_ONLY.\n"
            "Allowed predicted labels: IP or OTHER.\n"
            "Category definitions (same as judge):\n"
            "- IP: Intellectual Property theme such as catalog age/splits, export "
            "flows, local vs foreign streaming, territory charts, "
            "cross-border genre trends, and so on. Use IP only if the "
            "subsection is dominantly Intellectual Property themed.\n"
            "- OTHER: the subsection is not dominantly Intellectual Property themed.\n"
            "ACCEPT predicted_label unless it clearly violates these "
            "definitions. REJECT only when the label is clearly wrong."
        ),
    )


def validate_label_dtc_ip(
    *,
    title: str,
    summary: str,
    predicted_label: str,
    model_name: str | None = None,
) -> dict[str, str]:
    return _validate_predicted_label(
        title=title,
        summary=summary,
        predicted_label=predicted_label,
        model_name=model_name,
        allowed_labels=frozenset({CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER}),
        mode_policy=(
            "Mode: BOTH.\n"
            "Allowed predicted labels: DTC, IP, OTHER.\n"
            "Category definitions (same as judge):\n"
            "- DTC: Direct-to-Consumer theme such as fan engagement, "
            "superfans, short-form video, social media/gaming music, "
            "podcasts, merch, ticketing, direct artist–fan monetization, "
            "and so on.\n"
            "- IP: Intellectual Property theme such as catalog age/splits, "
            "export flows, local vs foreign streaming, territory charts, "
            "cross-border genre trends, and so on.\n"
            "- OTHER: everything else. If mixed, the dominant theme applies.\n"
            "ACCEPT predicted_label unless it clearly violates these "
            "definitions. REJECT only when the label is clearly wrong."
        ),
    )

