from __future__ import annotations

from typing import Any

from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
)


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def pipeline_results_from_labeled_neutral_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    """
    Build dtc_results / ip_results / section_results dicts for Streamlit and
    build_strategy_bundle. Insights for DTC/IP are concatenated neutral
    summaries (not the legacy DTC/IP insight chains).
    """
    dtc_blocks: list[str] = []
    ip_blocks: list[str] = []
    section_results: dict[str, dict[str, str]] = {}
    
    dtc_evidence: dict[str, str] = {}
    ip_evidence: dict[str, str] = {}

    for r in rows:
        title = str(r.get("display_title", ""))
        summ = str(r.get("summary", "")).strip()
        evidence = str(r.get("evidence", "") or "")
        cat = str(r.get("category", CATEGORY_OTHER))
        block = f"### {title}\n{summ}" if summ else f"### {title}\n_No summary._"

        if cat == CATEGORY_DTC:
            dtc_blocks.append(block)
            if title:
                dtc_evidence[title] = evidence
        elif cat == CATEGORY_IP:
            ip_blocks.append(block)
            if title:
                ip_evidence[title] = evidence
        else:
            if title:
                section_results[title] = {
                    "summary": summ or "_No summary._",
                    "evidence": evidence,
                }

    dtc_results: dict[str, Any] = {
        "insights": "\n\n".join(dtc_blocks).strip(),
        "evidence_by_section": dtc_evidence,
    }
    ip_results: dict[str, Any] = {
        "insights": "\n\n".join(ip_blocks).strip(),
        "evidence_by_section": ip_evidence,
    }
    return dtc_results, ip_results, section_results


def build_strategy_bundle(
    dtc_results: dict[str, Any] | None,
    ip_results: dict[str, Any] | None,
    section_results: dict[str, dict[str, str]] | None,
) -> str:
    """
    Assemble one strategy payload from DTC/IP insights only.

    Evidence and OTHER sections stay in analysis outputs/UI, but are intentionally
    excluded from strategy bundle input.
    """
    parts: list[str] = []
    _ = section_results

    dtc = dtc_results or {}
    insights = _clean_text(dtc.get("insights"))
    if insights:
        parts.append("=== DTC (Fan / Engagement) ===\n" + insights)

    ip = ip_results or {}
    ip_text = _clean_text(ip.get("insights"))
    if ip_text:
        parts.append("=== IP (Catalog / Export) ===\n" + ip_text)

    return "\n\n".join(parts).strip()


def strategy_bundle_has_content(bundle: str) -> bool:
    return bool(bundle and bundle.strip())
