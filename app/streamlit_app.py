from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.run_profiles import RunProfile
from src.pipelines.analysis_pipeline import run_analysis
from src.pipelines.strategy_pipeline import run_strategy
from src.services.strategy_bundle import (
    build_strategy_bundle,
    strategy_bundle_has_content,
)
from src.services.audit_pack import build_audit_filename, build_audit_pack
from src.services.file_search_retrieval import RetrievalError
from src.services.langchain_llm import resolved_analysis_model
from src.services.section_cache import hash_pdf_bytes
from src.services.file_search_store import (
    DEFAULT_STORE_DISPLAY_NAME,
    ensure_sections_in_store_from_bytes,
)
from src.services.section_splitter import SectionDetectionError
from src.agents.analysis.trace_formatter import format_trace_rows_plain_english

import streamlit as st

MAX_UPLOAD_MB = 100
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv()


_UI_VARS_LIGHT = """
            --ui-bg: #faf7f2;
            --ui-surface: #fffcf8;
            --ui-muted: #f5efe6;
            --ui-muted-high: #ebe3d6;
            --ui-text: #292018;
            --ui-secondary: #5c4f42;
            --ui-tertiary: #7a6b5d;
            --ui-line: #ddd3c4;
            --ui-accent: #c2410c;
            --ui-accent-hover: #9a3412;
            --ui-accent-container: #ffedd5;
            --ui-on-accent-container: #7c2d12;
            --ui-on-accent: #ffffff;
            --ui-success: #166534;
            --ui-success-container: #ecfdf3;
            --ui-success-border: #bbf7d0;
            --ui-accent-border: #fdba74;
            --ui-accent-grad-start: #ea580c;
            --ui-accent-grad-mid: #d97706;
            --ui-accent-grad-end: #b45309;
            --ui-btn-disabled-bg: #d6cbb8;
            --ui-btn-disabled-text: #6b5d4f;
            --ui-radius: 16px;
            --ui-radius-sm: 12px;
            --ui-elevation-1: 0 1px 2px rgba(41, 32, 24, 0.06),
                0 2px 8px rgba(194, 65, 12, 0.1);
            --ui-elevation-2: 0 2px 4px rgba(41, 32, 24, 0.08),
                0 8px 20px rgba(194, 65, 12, 0.14);
"""

_UI_VARS_DARK = """
            --ui-bg: #1a1714;
            --ui-surface: #242018;
            --ui-muted: #2e2923;
            --ui-muted-high: #3a342c;
            --ui-text: #f5efe6;
            --ui-secondary: #c4b8a8;
            --ui-tertiary: #9a8f82;
            --ui-line: #4a433c;
            --ui-accent: #fb923c;
            --ui-accent-hover: #fdba74;
            --ui-accent-container: #431407;
            --ui-on-accent-container: #ffedd5;
            --ui-on-accent: #ffffff;
            --ui-success: #4ade80;
            --ui-success-container: #14532d;
            --ui-success-border: #166534;
            --ui-accent-border: #9a3412;
            --ui-accent-grad-start: #fb923c;
            --ui-accent-grad-mid: #f97316;
            --ui-accent-grad-end: #ea580c;
            --ui-btn-disabled-bg: #3a342c;
            --ui-btn-disabled-text: #7a6b5d;
            --ui-radius: 16px;
            --ui-radius-sm: 12px;
            --ui-elevation-1: 0 1px 2px rgba(0, 0, 0, 0.35),
                0 2px 8px rgba(251, 146, 60, 0.08);
            --ui-elevation-2: 0 2px 4px rgba(0, 0, 0, 0.45),
                0 8px 20px rgba(251, 146, 60, 0.12);
"""


def _resolved_theme() -> str:
    theme = getattr(st.context, "theme", None)
    theme_type = theme.get("type") if theme is not None else None
    if isinstance(theme_type, str) and theme_type.lower() in {"light", "dark"}:
        return theme_type.lower()
    return "light"


def _root_vars_css() -> str:
    theme = _resolved_theme()
    vars_block = _UI_VARS_LIGHT if theme == "light" else _UI_VARS_DARK
    return f"""
        :root {{{vars_block}
        }}
        .stApp {{
            color-scheme: {theme};
        }}
    """


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Roboto:wght@400;500;700&display=swap');
        """
        + _root_vars_css()
        + """
        .stApp {
            background: var(--ui-bg);
        }

        header[data-testid="stHeader"] {
            background: var(--ui-surface);
            border-bottom: 1px solid var(--ui-line);
        }

        [data-testid="stMainBlockContainer"],
        .main .block-container {
            padding-top: max(2.75rem, calc(env(safe-area-inset-top, 0px) + 1.5rem));
            padding-bottom: 3rem;
            padding-left: clamp(1rem, 3vw, 2rem);
            padding-right: clamp(1rem, 3vw, 2rem);
            max-width: min(1120px, 100%);
        }

        [data-testid="stWidgetLabel"] {
            margin-bottom: 0.2rem !important;
        }

        div[data-testid="stMarkdownContainer"]:has(.ui-hero),
        div[data-testid="stMarkdownContainer"]:has(.ui-label),
        div[data-testid="stMarkdownContainer"]:has(.ui-note),
        div[data-testid="stMarkdownContainer"]:has(.ui-timer),
        div[data-testid="stMarkdownContainer"]:has(.ui-steps),
        div[data-testid="stMarkdownContainer"]:has(.ui-divider) {
            margin-bottom: 0 !important;
        }

        div[data-testid="stMarkdownContainer"]:has(.ui-hero) {
            padding-top: 0.15rem;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stWidgetLabel"],
        .stRadio label,
        .stCheckbox label {
            font-family: "Google Sans", Roboto, Arial, sans-serif !important;
            color: var(--ui-text);
        }

        .stButton button {
            font-family: "Google Sans", Roboto, Arial, sans-serif !important;
        }

        [data-testid="stMarkdownContainer"] h3 {
            font-size: 1.375rem !important;
            font-weight: 500 !important;
            letter-spacing: 0 !important;
            color: var(--ui-text) !important;
        }

        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--ui-secondary) !important;
        }

        .ui-hero {
            padding: 0.35rem 0 1rem;
            margin-bottom: 0.5rem;
        }
        .ui-accent-bar {
            width: 7rem;
            height: 4px;
            border-radius: 999px;
            margin-bottom: 0.85rem;
            background: linear-gradient(
                90deg,
                var(--ui-accent-grad-start) 0%,
                var(--ui-accent-grad-mid) 55%,
                var(--ui-accent-grad-end) 100%
            );
        }
        .ui-hero-eyebrow {
            font-size: clamp(0.8125rem, 2vw, 0.875rem);
            line-height: 1.45;
            color: var(--ui-accent);
            font-weight: 500;
            margin: 0 0 0.35rem 0;
            padding-top: 0.1rem;
        }
        .ui-hero h1 {
            font-size: clamp(2rem, 3.6vw, 2.75rem) !important;
            font-weight: 400 !important;
            letter-spacing: 0 !important;
            line-height: 1.12 !important;
            margin: 0 0 0.5rem 0 !important;
            color: var(--ui-text) !important;
        }
        .ui-hero-desc {
            margin: 0 0 0.45rem 0;
            max-width: min(42rem, 100%);
            font-size: clamp(0.9375rem, 2.2vw, 1rem);
            line-height: 1.55;
            color: var(--ui-secondary);
            font-weight: 400;
        }
        .ui-hero-stack {
            margin: 0;
            max-width: min(42rem, 100%);
            font-size: clamp(0.75rem, 1.8vw, 0.8125rem);
            line-height: 1.45;
            color: var(--ui-tertiary);
            letter-spacing: 0.01em;
            overflow-wrap: anywhere;
        }

        .ui-label {
            font-size: 0.6875rem;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--ui-secondary);
            margin: 0 0 0.5rem 0;
        }

        .ui-steps {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0 0 0.65rem 0;
        }
        .ui-step {
            flex: 1;
            min-width: 120px;
            background: var(--ui-muted);
            border-radius: var(--ui-radius-sm);
            padding: 0.85rem 1rem;
            border: 1px solid transparent;
        }
        .ui-step-num {
            font-size: 0.6875rem;
            font-weight: 500;
            color: var(--ui-tertiary);
            margin-bottom: 0.15rem;
        }
        .ui-step-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--ui-text);
        }
        .ui-step.active {
            background: var(--ui-accent-container);
            border-color: var(--ui-accent-border);
        }
        .ui-step.active .ui-step-title {
            color: var(--ui-on-accent-container);
        }
        .ui-step.done {
            background: var(--ui-success-container);
            border-color: var(--ui-success-border);
        }
        .ui-step.done .ui-step-title {
            color: var(--ui-success);
        }

        .ui-note {
            background: var(--ui-accent-container);
            border-radius: var(--ui-radius-sm);
            padding: 0.75rem 0.9rem;
            margin: 0 0 0.65rem 0;
            color: var(--ui-on-accent-container);
            font-size: 0.875rem;
            line-height: 1.5;
        }
        .ui-note strong {
            font-weight: 500;
        }

        .ui-timer {
            background: var(--ui-muted);
            border: 1px solid var(--ui-line);
            border-radius: var(--ui-radius-sm);
            padding: 0.7rem 0.9rem;
            margin: 0 0 0.65rem 0;
            color: var(--ui-text);
            font-size: 0.875rem;
            line-height: 1.45;
        }

        .ui-card-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--ui-text);
            margin: 0 0 0.85rem 0;
            line-height: 1.35;
        }
        .ui-card-header-kicker {
            color: var(--ui-accent);
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .ui-card-section-label {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--ui-secondary);
            margin: 0 0 0.75rem 0;
            line-height: 1.4;
            letter-spacing: 0.01em;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] h1,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] h2,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] h3,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] h4,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] h5,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] h6 {
            font-size: 1rem !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            margin: 1rem 0 0.5rem 0 !important;
            color: var(--ui-text) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] p,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] ul,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] ol {
            font-size: 1rem !important;
            line-height: 1.6 !important;
            margin: 0 0 0.75rem 0 !important;
            color: var(--ui-text);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] li {
            font-size: 1rem !important;
            line-height: 1.6 !important;
            margin: 0 0 0.35rem 0 !important;
            color: var(--ui-text);
        }
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] > p:last-child,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] > ul:last-child,
        div[data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stMarkdownContainer"] > ol:last-child {
            margin-bottom: 0 !important;
        }

        .ui-card-meta {
            background: var(--ui-muted);
            border-radius: var(--ui-radius-sm);
            padding: 0.85rem 1rem;
            margin-top: 0.75rem;
            font-size: 0.875rem;
            color: var(--ui-secondary);
            line-height: 1.5;
            border: 1px solid var(--ui-line);
        }
        .ui-card-meta strong {
            color: var(--ui-text);
            font-weight: 500;
        }

        .ui-rec-index {
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: var(--ui-accent);
            margin-bottom: 0.35rem;
        }
        .ui-rec-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--ui-text);
            margin: 0 0 0.75rem 0;
            line-height: 1.35;
        }

        .ui-empty {
            text-align: center;
            padding: 2.25rem 1.25rem;
            color: var(--ui-secondary);
            background: var(--ui-muted);
            border: 1px solid var(--ui-line);
            border-radius: var(--ui-radius);
            height: auto;
            min-height: 0;
            flex: 0 0 auto;
        }
        .ui-empty-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--ui-text);
            margin-bottom: 0.35rem;
        }
        .ui-empty-body {
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .ui-divider {
            height: 1px;
            background: var(--ui-line);
            margin: 1.25rem 0 1rem;
            border: none;
        }

        div[data-testid="stMarkdownContainer"]:has(.ui-empty) {
            flex: 0 0 auto !important;
            height: auto !important;
            margin-bottom: 0 !important;
        }

        div[data-testid="stVerticalBlock"] > div:has(.ui-empty) {
            flex: 0 0 auto !important;
            flex-grow: 0 !important;
            height: auto !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--ui-surface);
            border-radius: var(--ui-radius) !important;
            border-color: var(--ui-line) !important;
            box-shadow: var(--ui-elevation-1) !important;
            padding: 1rem;
            height: auto !important;
            align-self: flex-start;
            width: 100%;
            box-sizing: border-box;
        }

        div[data-testid="stFileUploader"] section {
            background: var(--ui-muted);
            border: 1px dashed var(--ui-line);
            border-radius: var(--ui-radius-sm);
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: var(--ui-accent-border);
            background: var(--ui-accent-container);
        }

        .stButton > button[kind="primary"],
        .stButton button[data-testid="baseButton-primary"],
        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div,
        .stButton button[data-testid="baseButton-primary"] p,
        .stButton button[data-testid="baseButton-primary"] span,
        .stButton button[data-testid="baseButton-primary"] div {
            color: var(--ui-on-accent) !important;
            -webkit-text-fill-color: var(--ui-on-accent) !important;
        }

        .stButton > button[kind="primary"],
        .stButton button[data-testid="baseButton-primary"] {
            background: linear-gradient(
                180deg,
                var(--ui-accent-grad-start) 0%,
                var(--ui-accent) 100%
            ) !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 0.58rem 1.35rem !important;
            font-weight: 500 !important;
            box-shadow: var(--ui-elevation-1) !important;
            letter-spacing: 0.01em !important;
        }
        .stButton > button[kind="primary"]:hover:not(:disabled),
        .stButton button[data-testid="baseButton-primary"]:hover:not(:disabled) {
            background: linear-gradient(
                180deg,
                var(--ui-accent-grad-start) 0%,
                var(--ui-accent-hover) 100%
            ) !important;
            box-shadow: var(--ui-elevation-2) !important;
        }
        .stButton > button[kind="primary"]:disabled,
        .stButton button[data-testid="baseButton-primary"]:disabled,
        .stButton > button[kind="primary"]:disabled p,
        .stButton > button[kind="primary"]:disabled span,
        .stButton > button[kind="primary"]:disabled div,
        .stButton button[data-testid="baseButton-primary"]:disabled p,
        .stButton button[data-testid="baseButton-primary"]:disabled span,
        .stButton button[data-testid="baseButton-primary"]:disabled div {
            background: var(--ui-btn-disabled-bg) !important;
            color: var(--ui-btn-disabled-text) !important;
            -webkit-text-fill-color: var(--ui-btn-disabled-text) !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        .stButton > button[kind="secondary"] {
            border-radius: 20px !important;
            border-color: var(--ui-line) !important;
            background: var(--ui-surface) !important;
        }
        .stButton > button[kind="secondary"],
        .stButton > button[kind="secondary"] p,
        .stButton > button[kind="secondary"] span {
            color: var(--ui-accent) !important;
            -webkit-text-fill-color: var(--ui-accent) !important;
        }

        .stProgress > div > div {
            background: var(--ui-accent) !important;
            border-radius: 999px !important;
        }
        .stProgress > div {
            background: var(--ui-muted-high) !important;
            border-radius: 999px !important;
            height: 6px !important;
        }

        div[data-testid="stExpander"] {
            background: transparent;
            border: none;
            box-shadow: none;
        }
        div[data-testid="stExpander"] details {
            background: var(--ui-surface);
            border-radius: var(--ui-radius-sm);
            border: 1px solid var(--ui-line);
            box-shadow: none;
        }

        div[data-testid="stAlert"] {
            background: var(--ui-muted) !important;
            border: 1px solid var(--ui-line) !important;
            color: var(--ui-text) !important;
            border-radius: var(--ui-radius-sm) !important;
        }

        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary span {
            color: var(--ui-text) !important;
        }

        div[data-testid="stCodeBlock"],
        div[data-testid="stCodeBlock"] pre,
        div[data-testid="stCodeBlock"] code {
            background: var(--ui-muted) !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-line);
            border-radius: var(--ui-radius-sm);
        }

        [data-testid="stMarkdownContainer"] code,
        [data-testid="stCaptionContainer"] code {
            background: var(--ui-muted) !important;
            color: var(--ui-text) !important;
            border-radius: 4px;
            padding: 0.05rem 0.35rem;
        }

        div[data-testid="stTextArea"] textarea {
            background: var(--ui-muted) !important;
            color: var(--ui-text) !important;
            border: 1px solid var(--ui-line) !important;
            border-radius: var(--ui-radius-sm) !important;
            font-family: "Google Sans", Roboto, Arial, sans-serif;
        }
        div[data-testid="stTextArea"] textarea:disabled {
            opacity: 1 !important;
            -webkit-text-fill-color: var(--ui-text) !important;
        }

        .stCheckbox label[data-baseweb="checkbox"],
        .stRadio label[data-disabled="true"] {
            color: var(--ui-tertiary) !important;
        }

        [data-testid="stMarkdownContainer"] a {
            color: var(--ui-accent) !important;
        }

        @media (max-width: 960px) {
            [data-testid="stMainBlockContainer"],
            .main .block-container {
                padding-top: max(2.25rem, calc(env(safe-area-inset-top, 0px) + 1.25rem));
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0.75rem !important;
            }

            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 0 !important;
            }

            .ui-step {
                min-width: calc(50% - 0.35rem);
            }
        }

        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"],
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .ui-step {
                min-width: 100%;
            }

            div[data-testid="stRadio"] > div[role="radiogroup"] {
                flex-direction: column !important;
                align-items: stretch !important;
                gap: 0.35rem !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                padding: 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _esc(text: str) -> str:
    return html.escape(str(text or ""))


_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_BOLD_LINE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")


def _prettify_section_label(label: str) -> str:
    cleaned = str(label or "").strip().rstrip(":").strip()
    if not cleaned:
        return ""
    return cleaned.title()


def _split_insight_body(text: str) -> tuple[str | None, str]:
    """Pull a leading report-section title out of the insight markdown body."""
    raw = str(text or "").strip()
    if not raw:
        return None, ""

    lines = raw.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return None, ""

    first = lines[index].strip()
    section: str | None = None
    heading = _HEADING_LINE_RE.match(first)
    bold = _BOLD_LINE_RE.match(first)
    if heading:
        section = _prettify_section_label(heading.group(2))
    elif bold:
        section = _prettify_section_label(bold.group(1))
    else:
        # Plain leading label like "ENGAGEMENT HORIZON:" (no markdown).
        candidate = first.rstrip(":").strip()
        letters = [ch for ch in candidate if ch.isalpha()]
        if (
            candidate
            and letters
            and all(ch.isupper() for ch in letters)
            and not first.startswith(("-", "*", ">"))
            and len(candidate.split()) <= 8
        ):
            section = _prettify_section_label(candidate)

    if not section:
        # Demote any remaining headings so they never outsize the card header.
        return None, _normalize_card_markdown(raw)

    rest_lines = lines[index + 1 :]
    while rest_lines and not rest_lines[0].strip():
        rest_lines.pop(0)
    rest = _normalize_card_markdown("\n".join(rest_lines))
    return section, rest


def _normalize_card_markdown(text: str) -> str:
    """Return text as-is. CSS handles heading sizes."""
    return str(text or "").strip()


def _render_insight_card(
    *,
    kicker: str,
    title: str,
    body: str,
    evidence_by_section: dict[str, str],
    empty_text: str,
) -> None:
    with st.container(border=True):
        md_parts = [
            f"""<div class="ui-card-header">
<span class="ui-card-header-kicker">{_esc(kicker)}</span>: {_esc(title)}
</div>"""
        ]
        
        if body.strip():
            section_label, rest = _split_insight_body(body)
            if section_label:
                md_parts.append(f'<div class="ui-card-section-label">{_esc(section_label)}</div>')
            if rest:
                md_parts.append(rest)
            elif not section_label:
                md_parts.append(f"_{empty_text}_")
        else:
            md_parts.append(f"_{empty_text}_")
            
        st.markdown("\n\n".join(md_parts), unsafe_allow_html=True)
        
        if evidence_by_section:
            with st.expander("Supporting evidence from report"):
                for section_name, evidence in evidence_by_section.items():
                    st.markdown(f"**{_esc(section_name)}**")
                    if evidence:
                        st.write(evidence)
                    else:
                        st.write("_No excerpt available._")


def _render_recommendation_card(*, index: int, rec: dict[str, str]) -> None:
    title = rec.get("title") or f"Recommendation {index}"
    insight = rec.get("insight") or ""
    evidence = rec.get("evidence") or ""
    action = rec.get("action") or ""
    with st.container(border=True):
        md_parts = [
            f"""<div class="ui-rec-index">Recommendation {index}</div>
<div class="ui-rec-title">{_esc(title)}</div>"""
        ]
        
        if insight:
            md_parts.append(_normalize_card_markdown(insight))
            
        meta_parts: list[str] = []
        if evidence:
            meta_parts.append(f"<strong>Evidence</strong><br>{_esc(evidence)}")
        if action:
            meta_parts.append(f"<strong>Recommended action</strong><br>{_esc(action)}")
            
        if meta_parts:
            meta_html = '<div class="ui-card-meta">' + "<br><br>".join(meta_parts) + "</div>"
            md_parts.append(meta_html)
            
        st.markdown("\n\n".join(md_parts), unsafe_allow_html=True)


def _render_hero() -> None:
    st.markdown(
        """
        <div class="ui-hero">
            <div class="ui-accent-bar"></div>
            <div class="ui-hero-eyebrow">Powered by Agentic AI</div>
            <h1>Music Industry Intelligence Brief</h1>
            <p class="ui-hero-desc">
                Upload a report to generate agent-driven insights and strategic
                recommendations linked to the source evidence in your document.
            </p>
            <p class="ui-hero-stack">
                Streamlit → Gemini File Search → LangGraph → LangChain → Gemini LLM
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _workflow_step(num: str, title: str, state: str) -> str:
    cls = "ui-step"
    if state == "active":
        cls += " active"
    elif state == "done":
        cls += " done"
    return (
        f'<div class="{cls}">'
        f'<div class="ui-step-num">{num}</div>'
        f'<div class="ui-step-title">{_esc(title)}</div>'
        f"</div>"
    )


def _render_workflow_steps(
    *,
    has_upload: bool,
    has_analysis: bool,
    has_strategy: bool,
) -> None:
    upload_state = "done" if has_upload else "active"
    analysis_state = "done" if has_analysis else ("active" if has_upload else "")
    strategy_state = "done" if has_strategy else (
        "active" if has_analysis else ""
    )
    steps = "".join(
        [
            _workflow_step("1", "Upload", upload_state),
            _workflow_step("2", "Analyze", analysis_state),
            _workflow_step("3", "Brief", strategy_state),
        ]
    )
    st.markdown(f'<div class="ui-steps">{steps}</div>', unsafe_allow_html=True)


def _render_panel(title: str) -> None:
    st.markdown(f'<div class="ui-label">{_esc(title)}</div>', unsafe_allow_html=True)


def _render_note(text: str) -> None:
    st.markdown(f'<div class="ui-note">{text}</div>', unsafe_allow_html=True)


def _render_timer_message(message: str) -> None:
    safe = _esc(message).replace("\n", "<br>")
    st.markdown(f'<div class="ui-timer">{safe}</div>', unsafe_allow_html=True)


def _render_empty_state(*, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ui-empty">
            <div class="ui-empty-title">{_esc(title)}</div>
            <div class="ui-empty-body">{_esc(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


load_env()

st.set_page_config(
    page_title="Industry Intelligence Brief",
    layout="wide",
    initial_sidebar_state="collapsed",
)
_current_theme = _resolved_theme()
_previous_theme = st.session_state.get("_ui_theme_type")
if _previous_theme is not None and _previous_theme != _current_theme:
    st.session_state["_ui_theme_type"] = _current_theme
    st.rerun()
st.session_state["_ui_theme_type"] = _current_theme
_inject_styles()
_render_hero()

_analysis_profile = RunProfile.DEV_ONE_PER_CATEGORY

_setup_col, _output_col = st.columns([1, 1.55], gap="medium")

with _setup_col:
    with st.container(border=True):
        _render_panel("Configure")
        _render_note(
            "<strong>Demo Scope.</strong> One report subsection per category to "
            "save resources. Choose a focus, upload your PDF, then run analysis."
        )

        _analysis_mode_label = st.radio(
            "Strategic focus",
            options=("Fan & audience (DTC)", "Catalog & IP"),
            index=None,
            horizontal=True,
            key="analysis_mode_choice",
            help="Choose which dimension of the report to analyze in this session.",
        )
        st.checkbox(
            "Combined view (DTC + IP)",
            value=False,
            disabled=True,
            help="Full combined analysis is unavailable in this demo due to resource limits.",
        )

        _analysis_mode_value: str | None = None
        if _analysis_mode_label == "Fan & audience (DTC)":
            _analysis_mode_value = "dtc_only"
        elif _analysis_mode_label == "Catalog & IP":
            _analysis_mode_value = "ip_only"

        uploaded_file = st.file_uploader(
            "Document (PDF)",
            type=["pdf"],
            accept_multiple_files=False,
            help=f"Upload a music industry report (PDF, max {MAX_UPLOAD_MB} MB).",
        )

        _file_too_large = (
            uploaded_file is not None and uploaded_file.size > MAX_UPLOAD_BYTES
        )

        summarize_classify_clicked = st.button(
            "Run analysis",
            type="primary",
            disabled=(
                uploaded_file is None
                or not _analysis_mode_value
                or _file_too_large
            ),
            use_container_width=True,
            key="btn_summarize_classify",
        )

with _output_col:
    st.session_state.setdefault("file_search_store_name", None)
    st.session_state.setdefault("file_search_section_doc_names", None)
    st.session_state.setdefault("file_search_manifest", None)
    st.session_state.setdefault("file_search_splitter_mode", None)
    st.session_state.setdefault("file_search_sections_reused", None)
    st.session_state.setdefault("dtc_results", None)
    st.session_state.setdefault("ip_results", None)
    st.session_state.setdefault("ip_section_used", None)
    st.session_state.setdefault("section_results", None)
    st.session_state.setdefault("section_used", None)
    st.session_state.setdefault("neutral_labeled_rows", None)
    st.session_state.setdefault("strategy_results", None)
    st.session_state.setdefault("last_summarize_timings", None)
    st.session_state.setdefault("last_strategy_timings", None)
    st.session_state.setdefault("analysis_react_messages", None)
    st.session_state.setdefault("analysis_mode_used", None)
    st.session_state.setdefault("analysis_warnings", None)
    st.session_state.setdefault("live_timer_message", None)

    if uploaded_file and st.session_state.get("pdf_name") != uploaded_file.name:
        for key in (
            "file_search_store_name",
            "file_search_section_doc_names",
            "file_search_manifest",
            "file_search_splitter_mode",
            "file_search_sections_reused",
            "dtc_results",
            "ip_results",
            "ip_section_used",
            "section_results",
            "section_used",
            "neutral_labeled_rows",
            "strategy_results",
            "last_summarize_timings",
            "last_strategy_timings",
            "analysis_react_messages",
            "analysis_mode_used",
            "analysis_warnings",
            "live_timer_message",
        ):
            st.session_state[key] = None

    with st.container(border=True):
        _render_panel("Workspace")

        _has_analysis = bool(st.session_state.get("neutral_labeled_rows"))
        _has_strategy = bool(st.session_state.get("strategy_results"))
        _render_workflow_steps(
            has_upload=uploaded_file is not None,
            has_analysis=_has_analysis,
            has_strategy=_has_strategy,
        )

        if st.session_state.get("live_timer_message"):
            _render_timer_message(st.session_state["live_timer_message"])

        if uploaded_file:
            if _file_too_large:
                size_mb = uploaded_file.size / (1024 * 1024)
                st.error(
                    f"**{uploaded_file.name}** is {size_mb:.1f} MB — "
                    f"max allowed is {MAX_UPLOAD_MB} MB."
                )
            else:
                size_kb = uploaded_file.size / 1024
                st.success(f"**{uploaded_file.name}** ready, {size_kb:.0f} KB")

                _reused = st.session_state.get("file_search_sections_reused")
                _store_name = st.session_state.get("file_search_store_name")
                _section_doc_names = st.session_state.get(
                    "file_search_section_doc_names"
                )
                if _reused is not None or (_store_name and _section_doc_names):
                    with st.expander(
                        "Document processing details", expanded=False
                    ):
                        if _reused is not None:
                            st.caption(
                                "Reused cached sections."
                                if _reused
                                else "Indexed new sections from PDF."
                            )
                        if _store_name and _section_doc_names:
                            st.caption(
                                f"Indexed sections: **{len(_section_doc_names)}**"
                            )
        else:
            _render_empty_state(
                title="No report uploaded",
                body="Upload a PDF to begin.",
            )


def _ensure_store(pdf_bytes: bytes, display_name: str) -> str:
    store_display_name = os.getenv(
        "FILE_SEARCH_STORE_NAME", DEFAULT_STORE_DISPLAY_NAME
    )
    (
        store_name,
        section_doc_names,
        reused,
        manifest,
        splitter_mode,
    ) = ensure_sections_in_store_from_bytes(
        file_bytes=pdf_bytes,
        display_name=display_name,
        store_display_name=store_display_name,
    )
    st.session_state["file_search_store_name"] = store_name
    st.session_state["file_search_section_doc_names"] = section_doc_names
    st.session_state["file_search_manifest"] = manifest
    st.session_state["file_search_splitter_mode"] = splitter_mode
    st.session_state["file_search_sections_reused"] = reused
    return store_name


def _make_progress():
    progress_bar = st.progress(0)
    progress_label = st.empty()

    def update_progress(percent: int, message: str) -> None:
        progress_bar.progress(percent)
        progress_label.markdown(f"**{percent}%**  {message}")

    return update_progress


def _make_live_timer():
    slot = st.empty()
    t0 = time.perf_counter()
    completed: dict[str, float] = {}

    def _render(*, stage: str, done_label: str | None = None) -> None:
        elapsed = time.perf_counter() - t0
        if done_label:
            lines = [f"**{done_label}**  {elapsed:.1f}s total"]
        else:
            lines = [f"**In progress**  {elapsed:.1f}s  {stage}"]
        for label, secs in completed.items():
            lines.append(f"- {label}: {secs:.1f}s")
        message = "\n".join(lines)
        slot.markdown(
            f'<div class="ui-timer">{_esc(message).replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )
        st.session_state["live_timer_message"] = message

    def update(
        *,
        stage: str,
        mark_complete: tuple[str, float] | None = None,
    ) -> None:
        if mark_complete:
            label, secs = mark_complete
            completed[label] = secs
        _render(stage=stage)

    def finalize(done_label: str) -> None:
        _render(stage="", done_label=done_label)

    def clear() -> None:
        slot.empty()
        st.session_state["live_timer_message"] = None

    return update, finalize, clear


def _check_api_key() -> bool:
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Service configuration is incomplete. Contact your administrator.")
        return False
    return True


def _render_react_trace(
    *,
    target,
    trace_rows: list[dict[str, object]],
    title: str,
    key_prefix: str,
) -> None:
    with target.container():
        st.caption(title)
        st.text_area(
            "trace",
            value=format_trace_rows_plain_english(trace_rows),
            height=280,
            disabled=True,
            label_visibility="collapsed",
            key=f"{key_prefix}_trace_{len(trace_rows)}",
        )


if (
    summarize_classify_clicked
    and uploaded_file
    and not _file_too_large
    and _check_api_key()
):
    st.session_state["live_timer_message"] = None
    live_trace_slot = st.empty()
    react_trace: list[dict[str, object]] = []
    update_timer, finalize_timer, clear_timer = _make_live_timer()
    try:
        update_progress = _make_progress()
        with st.spinner("Analyzing report..."):
            update_timer(stage="Preparing document")
            update_progress(5, "Preparing document")
            pdf_bytes = uploaded_file.getvalue()
            pdf_hash = hash_pdf_bytes(pdf_bytes)
            update_timer(stage="Structuring report sections")
            update_progress(15, "Structuring report sections")
            t_wall0 = time.perf_counter()
            t_split0 = time.perf_counter()
            store_name = _ensure_store(pdf_bytes, uploaded_file.name)
            split_s = time.perf_counter() - t_split0
            manifest = st.session_state.get("file_search_manifest") or []
            nsec = len(manifest)
            update_timer(
                stage="AI analysis agent",
                mark_complete=("Report structuring", split_s),
            )
            update_progress(
                25,
                f"AI analysis ({nsec} sections, demo: 1 per category)",
            )
            inner_timings: dict[str, float] = {}
            react_trace.clear()
            analysis_errors: list[str] = []

            _render_react_trace(
                target=live_trace_slot,
                trace_rows=react_trace,
                title="Analysis in progress",
                key_prefix="live",
            )

            def _on_react_trace_update(
                trace_rows: list[dict[str, object]],
            ) -> None:
                react_trace.clear()
                react_trace.extend(trace_rows)
                update_timer(stage="AI analysis agent")
                _render_react_trace(
                    target=live_trace_slot,
                    trace_rows=react_trace,
                    title="Analysis in progress",
                    key_prefix="live",
                )

            t_dyn0 = time.perf_counter()
            dtc_r, ip_r, sec_r, labeled = run_analysis(
                file_search_store_name=store_name,
                manifest=manifest,
                source_filename=uploaded_file.name,
                gemini_model_name=None,
                synthesis_model_name=None,
                pdf_hash=pdf_hash,
                analysis_mode=_analysis_mode_value,
                timings_out=inner_timings,
                react_messages_out=react_trace,
                react_trace_callback=_on_react_trace_update,
                errors_out=analysis_errors,
                run_profile=_analysis_profile,
            )
            dyn_wall_s = time.perf_counter() - t_dyn0
            wall_total = time.perf_counter() - t_wall0
            update_timer(
                stage="Finalizing brief",
                mark_complete=("AI analysis agent", dyn_wall_s),
            )
            st.session_state["last_summarize_timings"] = {
                "vision_split_file_search_s": split_s,
                **inner_timings,
                "summarize_classify_wall_s": wall_total,
                "dynamic_pipeline_wall_s": dyn_wall_s,
            }
            update_progress(92, "Finalizing insights")
        st.session_state["pdf_name"] = uploaded_file.name
        st.session_state["dtc_results"] = dtc_r
        st.session_state["ip_results"] = ip_r
        st.session_state["section_results"] = sec_r
        st.session_state["neutral_labeled_rows"] = labeled
        st.session_state["ip_section_used"] = None
        st.session_state["section_used"] = None
        st.session_state["strategy_results"] = None
        st.session_state["analysis_react_messages"] = react_trace
        st.session_state["analysis_mode_used"] = _analysis_mode_value
        st.session_state["analysis_warnings"] = analysis_errors or None
        update_progress(100, "Complete")
        finalize_timer("Analysis complete")
        live_trace_slot.empty()
    except RetrievalError as exc:
        st.session_state["last_summarize_timings"] = None
        if react_trace:
            st.session_state["analysis_react_messages"] = list(react_trace)
            st.session_state["analysis_mode_used"] = _analysis_mode_value
        for key in (
            "dtc_results",
            "ip_results",
            "ip_section_used",
            "section_results",
            "section_used",
            "neutral_labeled_rows",
            "strategy_results",
            "last_strategy_timings",
            "analysis_warnings",
            "live_timer_message",
        ):
            st.session_state[key] = None
        clear_timer()
        live_trace_slot.empty()
        st.error(f"Analysis could not be completed: {exc}")
    except SectionDetectionError as exc:
        st.session_state["last_summarize_timings"] = None
        clear_timer()
        live_trace_slot.empty()
        st.error(f"Could not parse report structure: {exc}")
    except Exception as exc:
        st.session_state["last_summarize_timings"] = None
        clear_timer()
        live_trace_slot.empty()
        st.error(f"An unexpected error occurred: {exc}")

st.markdown('<div class="ui-divider"></div>', unsafe_allow_html=True)

_insights_col, _strategy_col = st.columns([1, 1], gap="medium")

with _insights_col:
    st.markdown("### Insights")
    _labeled = st.session_state.get("neutral_labeled_rows")
    _analysis_warnings = st.session_state.get("analysis_warnings") or []
    if _analysis_warnings:
        for _warning in _analysis_warnings:
            st.warning(_warning)

    dtc_results = st.session_state.get("dtc_results") or {}
    _mode_used = st.session_state.get("analysis_mode_used")
    if dtc_results and _mode_used != "ip_only":
        _render_insight_card(
            kicker="Direct-to-consumer",
            title="Fan & audience dynamics",
            body=dtc_results.get("insights", ""),
            evidence_by_section=dtc_results.get("evidence_by_section", {}) or {},
            empty_text="No audience-focused insights were identified in this pass.",
        )
    elif _mode_used == "dtc_only" and not dtc_results.get("insights"):
        _render_empty_state(
            title="Awaiting analysis",
            body="Run analysis to surface fan and audience insights from your report.",
        )

    ip_results = st.session_state.get("ip_results") or {}
    if ip_results and _mode_used != "dtc_only":
        _render_insight_card(
            kicker="Intellectual property",
            title="Catalog & export performance",
            body=ip_results.get("insights", ""),
            evidence_by_section=ip_results.get("evidence_by_section", {}) or {},
            empty_text="No catalog-focused insights were identified in this pass.",
        )
    elif _mode_used == "ip_only" and not ip_results.get("insights"):
        _render_empty_state(
            title="Awaiting analysis",
            body="Run analysis to surface catalog and IP insights from your report.",
        )

    if not _mode_used and not _labeled:
        _render_empty_state(
            title="Insights will appear here",
            body="Select a focus, upload a report, and run analysis.",
        )

    if _labeled:
        with st.expander("Section classification detail", expanded=False):
            for row in _labeled:
                st.caption(
                    f"**{row.get('display_title', '')}**  "
                    f"`{row.get('category', '')}`"
                )

    _react_trace = st.session_state.get("analysis_react_messages") or []
    if _react_trace:
        with st.expander("How the AI reached these insights", expanded=False):
            _render_react_trace(
                target=st,
                trace_rows=_react_trace,
                title="Reasoning trace",
                key_prefix="final",
            )

    if _react_trace or _labeled:
        _audit_pack = build_audit_pack(
            source_filename=st.session_state.get("pdf_name"),
            analysis_mode=_mode_used,
            run_profile=_analysis_profile.value,
            model_name=resolved_analysis_model(None),
            timings=st.session_state.get("last_summarize_timings"),
            labeled_rows=_labeled,
            dtc_results=st.session_state.get("dtc_results"),
            ip_results=st.session_state.get("ip_results"),
            strategy_results=st.session_state.get("strategy_results"),
            react_trace=_react_trace,
            react_trace_plain_english=format_trace_rows_plain_english(_react_trace),
        )
        st.download_button(
            "Download run audit (JSON)",
            data=json.dumps(_audit_pack, indent=2),
            file_name=build_audit_filename(
                source_filename=st.session_state.get("pdf_name"),
                analysis_mode=_mode_used,
            ),
            mime="application/json",
            use_container_width=True,
            key="btn_download_audit",
        )

with _strategy_col:
    st.markdown("### Recommendations")
    st.caption(
        "Strategic actions from the analysis, with evidence and "
        "next steps for leadership review."
    )

    _bundle = build_strategy_bundle(
        st.session_state.get("dtc_results"),
        st.session_state.get("ip_results"),
        st.session_state.get("section_results"),
    )
    _strategy_ready = strategy_bundle_has_content(_bundle)
    strategy_clicked = st.button(
        "Generate executive brief",
        disabled=not _strategy_ready,
        use_container_width=True,
        key="btn_strategy",
    )
    if not _strategy_ready:
        _render_empty_state(
            title="Recommendations pending",
            body="Complete analysis first to generate your executive brief.",
        )

    if strategy_clicked and _strategy_ready:
        st.session_state["live_timer_message"] = None
        try:
            update_progress = _make_progress()
            update_timer, finalize_timer, clear_timer = _make_live_timer()
            update_timer(stage="Preparing strategy inputs")
            update_progress(15, "Preparing strategy inputs")
            t_strat0 = time.perf_counter()
            with st.spinner("Generating executive brief..."):
                update_timer(stage="Drafting recommendations")
                update_progress(45, "Drafting recommendations")
                recs = run_strategy(
                    st.session_state.get("dtc_results"),
                    st.session_state.get("ip_results"),
                    st.session_state.get("section_results"),
                    model_name=None,
                )
                strat_s = time.perf_counter() - t_strat0
                update_timer(
                    stage="Formatting brief",
                    mark_complete=("Strategy generation", strat_s),
                )
                update_progress(90, "Formatting brief")
            st.session_state["last_strategy_timings"] = {
                "strategy_pipeline_s": strat_s,
            }
            st.session_state["strategy_results"] = recs
            update_progress(100, "Complete")
            finalize_timer("Executive brief ready")
            if not recs:
                st.warning(
                    "No recommendations could be generated. Please try again."
                )
        except Exception as exc:
            st.session_state["last_strategy_timings"] = None
            clear_timer()
            st.error(f"Brief generation failed: {exc}")

    _strat = st.session_state.get("strategy_results")
    if _strat:
        for i, rec in enumerate(_strat, start=1):
            _render_recommendation_card(index=i, rec=rec)


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner_utils.script_run_context import (
        get_script_run_ctx,
    )

    if get_script_run_ctx() is None:
        from streamlit.web import cli as stcli

        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ]
        raise SystemExit(stcli.main())
