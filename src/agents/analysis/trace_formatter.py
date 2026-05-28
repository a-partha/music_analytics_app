from __future__ import annotations

import ast
import json
from typing import Any


def _safe_parse_payload(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)
        except Exception:
            continue
    return None


def _join_keys_from_pending(payload: Any) -> str:
    if not isinstance(payload, list):
        return ""
    keys: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = str(item.get("subsection_key") or "").strip()
        if key:
            keys.append(key)
    if not keys:
        return ""
    if len(keys) <= 4:
        return ", ".join(keys)
    return ", ".join(keys[:4]) + ", ..."


def _describe_judge_observation(text: str) -> str:
    payload = _safe_parse_payload(text)
    if not isinstance(payload, dict):
        return "Judge returned a result."
    label = str(payload.get("label") or "").strip().upper()
    decision = str(payload.get("validator_decision") or "").strip().upper()
    reason = str(payload.get("validator_reason") or "").strip()
    accepted = bool(payload.get("accepted_for_target"))
    missing = payload.get("missing_targets")
    missing_text = ""
    if isinstance(missing, list) and missing:
        missing_text = f" Missing targets: {', '.join(str(v) for v in missing)}."

    if decision == "ACCEPT" and accepted:
        return (
            f"Initial label {label or 'UNKNOWN'} accepted by validator; "
            f"counts toward target.{missing_text}"
        )
    if decision == "ACCEPT" and not accepted:
        return (
            f"Initial label {label or 'UNKNOWN'} accepted, but target slot "
            f"was already filled.{missing_text}"
        )
    if decision == "REJECT":
        if reason:
            return (
                f"Initial label {label or 'UNKNOWN'} rejected by validator "
                f"(reason: {reason}).{missing_text}"
            )
        return (
            f"Initial label {label or 'UNKNOWN'} rejected by validator."
            f"{missing_text}"
        )
    return "Judge returned an unrecognized validator result."


def _format_one_event(event: dict[str, Any]) -> str | None:
    step = event.get("step", "?")
    kind = str(event.get("kind") or "")
    tool = str(event.get("tool") or "")
    text = str(event.get("text") or "").strip()

    if kind == "thought":
        return f"Step {step}: Planned the next action."

    if kind == "tool":
        args = event.get("args") or {}
        if tool == "list_pending_subsections":
            return f"Step {step}: Checked which sections are still pending."
        if tool == "fetch_and_summarize_subsection":
            key = "?"
            if isinstance(args, dict):
                key = str(args.get("subsection_key") or "?")
            return f"Step {step}: Fetched and summarized section {key}."
        if tool.startswith("judge_section_"):
            key = "?"
            if isinstance(args, dict):
                key = str(args.get("subsection_key") or "?")
            return f"Step {step}: Asked the judge to label section {key}."
        if tool == "finish":
            return f"Step {step}: Checked if category targets are satisfied."
        return f"Step {step}: Called tool {tool}."

    if kind == "observation":
        if tool == "list_pending_subsections":
            payload = _safe_parse_payload(text)
            if isinstance(payload, list):
                keys = _join_keys_from_pending(payload)
                if keys:
                    return (
                        f"Step {step}: Pending list received ({len(payload)}): "
                        f"{keys}."
                    )
                return f"Step {step}: Pending list received ({len(payload)})."
            return f"Step {step}: Pending list received."
        if tool == "fetch_and_summarize_subsection":
            return f"Step {step}: Summary returned and stored."
        if tool.startswith("judge_section_"):
            return f"Step {step}: {_describe_judge_observation(text)}"
        if tool == "finish":
            if text == "FINISH_OK":
                return f"Step {step}: Finish check passed."
            if text:
                return f"Step {step}: Finish check: {text}"
            return f"Step {step}: Finish check returned."
        if tool == "runtime":
            if text:
                return f"Step {step}: Runtime note: {text}"
            return f"Step {step}: Runtime note received."
        return f"Step {step}: Received observation from {tool}."

    return None


def format_trace_rows_plain_english(
    rows: list[dict[str, Any]],
    *,
    max_lines: int = 160,
) -> str:
    lines: list[str] = []
    for event in rows:
        line = _format_one_event(event)
        if line:
            lines.append(line)
    if not lines:
        return "Waiting for agent activity..."
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)

