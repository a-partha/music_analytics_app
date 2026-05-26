from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.services.langchain_llm import get_langchain_gemini_model
from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
)

_VALID = frozenset({CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER})


def _normalize_label(raw: str) -> str:
    u = raw.strip().upper()
    if u == "DTC":
        return CATEGORY_DTC
    if u == "IP":
        return CATEGORY_IP
    if u in ("OTHER", "OTHER_SECTIONS", "MISC"):
        return CATEGORY_OTHER
    raise ValueError(f"Unsupported category label: {raw!r}")


def parse_section_label_json(llm_text: str) -> dict[str, str]:
    """Parse model output into subsection_key -> DTC|IP|OTHER."""
    cleaned = (llm_text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    payload = json.loads(cleaned)
    if isinstance(payload, dict) and "labels" in payload:
        payload = payload["labels"]
    if not isinstance(payload, list):
        raise ValueError("Expected JSON array of label objects.")
    out: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = item.get("subsection_key")
        label_raw = item.get("label")
        if isinstance(key, str) and key.strip() and isinstance(
            label_raw, str
        ):
            out[key.strip()] = _normalize_label(label_raw)
    return out


def label_neutral_rows_with_synthesis(
    rows: list[dict[str, Any]],
    *,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Assign DTC / IP / OTHER to each row using summary (+ title) via Ollama.
    Each input row: subsection_key, display_title, summary, evidence (evidence
    preserved in output, not sent to the classifier).
    """
    if not rows:
        return []

    lines: list[str] = []
    for i, r in enumerate(rows, start=1):
        key = str(r.get("subsection_key", ""))
        title = str(r.get("display_title", ""))
        summary = str(r.get("summary", ""))[:8000]
        lines.append(
            f"{i}. subsection_key={key!r} display_title={title!r}\n"
            f"summary={summary!r}"
        )
    blocks = "\n\n".join(lines)

    prompt = ChatPromptTemplate.from_template(
        "You label music-industry document SECTIONS into exactly one category "
        "each, using the section title and the neutral summary below.\n"
        "Categories:\n"
        "- DTC: fan engagement, superfans, short-form video, social/gaming "
        "music, podcasts, merch, ticketing, direct artist–fan monetization.\n"
        "- IP: catalog age/splits, export flows, local vs foreign streaming, "
        "territory charts, cross-border genre trends.\n"
        "- OTHER: everything else (macro metrics, artist rankings, forward "
        "outlook, generic charts) that is not mainly DTC or IP.\n"
        "If a section mixes themes, pick the dominant one.\n\n"
        "Sections:\n{blocks}\n\n"
        "Return STRICT JSON only — a JSON array of objects, one per "
        "subsection_key above, with keys subsection_key and label. "
        "label must be exactly DTC, IP, or OTHER. Include every "
        "subsection_key exactly once. No other text.\n"
    )

    llm = get_langchain_gemini_model(model_name=model_name, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"blocks": blocks}).strip()
    try:
        by_key = parse_section_label_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        snippet = raw.replace("\n", " ")[:240]
        raise ValueError(
            "Label model returned malformed JSON payload "
            f"(snippet={snippet!r})."
        ) from exc

    expected_keys: list[str] = []
    for row in rows:
        key = str(row.get("subsection_key", "")).strip()
        if key and key not in expected_keys:
            expected_keys.append(key)
    missing = [key for key in expected_keys if key not in by_key]
    if missing:
        short = ", ".join(missing[:8])
        tail = " ..." if len(missing) > 8 else ""
        raise ValueError(
            "Label model omitted subsection_key values: "
            f"{short}{tail}."
        )

    merged: list[dict[str, Any]] = []
    for r in rows:
        key = str(r.get("subsection_key", ""))
        cat = by_key.get(key, CATEGORY_OTHER)
        if cat not in _VALID:
            cat = CATEGORY_OTHER
        row = dict(r)
        row["category"] = cat
        merged.append(row)
    return merged
