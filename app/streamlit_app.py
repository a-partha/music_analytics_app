from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.run_profiles import RunProfile, resolve_run_profile
from src.pipelines.analysis_pipeline import run_analysis
from src.pipelines.strategy_pipeline import run_strategy
from src.services.strategy_bundle import (
    build_strategy_bundle,
    strategy_bundle_has_content,
)
from src.services.file_search_retrieval import RetrievalError
from src.services.section_cache import hash_pdf_bytes
from src.services.file_search_store import (
    DEFAULT_STORE_DISPLAY_NAME,
    ensure_sections_in_store_from_bytes,
)
from src.services.section_splitter import SectionDetectionError
from src.agents.analysis.trace_formatter import format_trace_rows_plain_english

import streamlit as st


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv()


load_env()

st.set_page_config(page_title="Music Analytics Pipeline", layout="wide")
st.title("Music Analytics Pipeline")
st.caption(
    "Upload a report PDF. LangGraph orchestrates analysis (File Search + "
    "Gemini) and strategy recommendations."
)

if "dev_mode" not in st.session_state:
    st.session_state["dev_mode"] = (
        resolve_run_profile() == RunProfile.DEV_ONE_PER_CATEGORY
    )

dev_mode = st.checkbox(
    "Dev mode (1 section per category)",
    key="dev_mode",
    help="Limits analysis to one DTC, one IP, and one Other subsection. "
    "When unchecked, runs the full manifest (ignores dev env vars).",
)

_analysis_profile = (
    RunProfile.DEV_ONE_PER_CATEGORY
    if dev_mode
    else RunProfile.FULL
)
st.caption(
    f"Analysis profile: **`{_analysis_profile.value}`** "
    f"({'3 subsections max' if dev_mode else 'all manifest subsections'})"
)

_analysis_mode_label: str | None = None
_analysis_mode_value: str | None = None
if dev_mode:
    _analysis_mode_label = st.radio(
        "Dev analysis mode",
        options=("DTC only", "IP only", "Both"),
        index=None,
        key="analysis_mode_choice",
        help="Required in Dev mode. Select which ReAct agent to run.",
    )
    if _analysis_mode_label == "DTC only":
        _analysis_mode_value = "dtc_only"
    elif _analysis_mode_label == "IP only":
        _analysis_mode_value = "ip_only"
    elif _analysis_mode_label == "Both":
        _analysis_mode_value = "both"

uploaded_file = st.file_uploader(
    "Upload Luminate report (PDF)", type=["pdf"], accept_multiple_files=False
)

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
    ):
        st.session_state[key] = None


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
        progress_label.text(f"{percent}% - {message}")

    return update_progress


def _check_api_key() -> bool:
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Missing GEMINI_API_KEY. Add it to `.env` to use File Search.")
        return False
    return True


_SUMMARIZE_TIMING_ORDER: tuple[tuple[str, str], ...] = (
    ("vision_split_file_search_s", "Vision split + File Search upload"),
    ("filter_manifest_s", "Apply run profile to manifest"),
    ("run_profile", "Run profile used"),
    ("batch_label_s", "Batch section labels (Gemini)"),
    ("bundle_shape_s", "Build DTC / IP / Other bundle"),
    ("analysis_graph_wall_s", "Analysis graph (wall clock)"),
    ("summarize_classify_wall_s", "Total (Summarize & classify)"),
)


def _render_run_timings_expander() -> None:
    summarize_t = st.session_state.get("last_summarize_timings")
    strategy_t = st.session_state.get("last_strategy_timings")
    if not summarize_t and not strategy_t:
        return
    with st.expander("Run timings (seconds)", expanded=False):
        if summarize_t:
            st.markdown("**Summarize & classify**")
            for key, label in _SUMMARIZE_TIMING_ORDER:
                if key not in summarize_t:
                    continue
                val = summarize_t[key]
                if isinstance(val, (int, float)):
                    st.caption(f"{label}: **{val:.2f}**")
                else:
                    st.caption(f"{label}: **{val}**")
        if strategy_t and "strategy_pipeline_s" in strategy_t:
            st.markdown("**Strategy**")
            st.caption(
                "Recommendations (Gemini): "
                f"**{strategy_t['strategy_pipeline_s']:.2f}**"
            )


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
            height=360,
            disabled=True,
            label_visibility="collapsed",
            key=f"{key_prefix}_trace_{len(trace_rows)}",
        )


summarize_classify_clicked = st.button(
    "Summarize & classify report",
    type="primary",
    disabled=uploaded_file is None
    or (dev_mode and not _analysis_mode_value),
    use_container_width=True,
    key="btn_summarize_classify",
)

if uploaded_file:
    size_kb = uploaded_file.size / 1024
    st.success(f"Ready: {uploaded_file.name} ({size_kb:.1f} KB)")
    reused = st.session_state.get("file_search_sections_reused")
    store_name = st.session_state.get("file_search_store_name")
    section_doc_names = st.session_state.get(
        "file_search_section_doc_names"
    )
    if reused is not None:
        status = (
            "Reused cached section documents."
            if reused
            else "Uploaded new section documents."
        )
        st.caption(status)
        if store_name and section_doc_names:
            st.caption(
                f"Store: {store_name} · Sections: {len(section_doc_names)}"
            )
else:
    st.info("Upload a PDF to enable analysis.")

if summarize_classify_clicked and uploaded_file and _check_api_key():
    live_trace_slot = st.empty() if dev_mode else None
    react_trace: list[dict[str, object]] = []
    try:
        update_progress = _make_progress()
        with st.spinner("Summarize & classify…"):
            update_progress(5, "Preparing PDF")
            pdf_bytes = uploaded_file.getvalue()
            pdf_hash = hash_pdf_bytes(pdf_bytes)
            update_progress(15, "Vision split + File Search upload")
            t_wall0 = time.perf_counter()
            t_split0 = time.perf_counter()
            store_name = _ensure_store(pdf_bytes, uploaded_file.name)
            split_s = time.perf_counter() - t_split0
            manifest = st.session_state.get("file_search_manifest") or []
            nsec = len(manifest)
            dev_suffix = " · dev picks 1/category after labeling" if dev_mode else ""
            update_progress(
                25,
                f"Neutral retrieval + summaries ({nsec} sections{dev_suffix}) — Gemini",
            )
            inner_timings: dict[str, float] = {}
            react_trace.clear()
            analysis_errors: list[str] = []

            if live_trace_slot is not None:
                _render_react_trace(
                    target=live_trace_slot,
                    trace_rows=react_trace,
                    title="Agent thinking (live)",
                    key_prefix="live",
                )

            def _on_react_trace_update(
                trace_rows: list[dict[str, object]],
            ) -> None:
                react_trace.clear()
                react_trace.extend(trace_rows)
                if live_trace_slot is not None:
                    _render_react_trace(
                        target=live_trace_slot,
                        trace_rows=react_trace,
                        title="Agent thinking (live)",
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
                react_trace_callback=_on_react_trace_update if dev_mode else None,
                errors_out=analysis_errors,
                run_profile=_analysis_profile,
            )
            dyn_wall_s = time.perf_counter() - t_dyn0
            wall_total = time.perf_counter() - t_wall0
            st.session_state["last_summarize_timings"] = {
                "vision_split_file_search_s": split_s,
                **inner_timings,
                "summarize_classify_wall_s": wall_total,
                "dynamic_pipeline_wall_s": dyn_wall_s,
            }
            update_progress(92, "Done labeling")
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
        update_progress(100, "Done")
        if live_trace_slot is not None:
            live_trace_slot.empty()
    except RetrievalError as exc:
        st.session_state["last_summarize_timings"] = None
        if dev_mode and react_trace:
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
        ):
            st.session_state[key] = None
        if live_trace_slot is not None:
            live_trace_slot.empty()
        st.error(f"Pipeline failed: {exc}")
    except SectionDetectionError as exc:
        st.session_state["last_summarize_timings"] = None
        if live_trace_slot is not None:
            live_trace_slot.empty()
        st.error(f"Section split failed: {exc}")
    except Exception as exc:
        st.session_state["last_summarize_timings"] = None
        if live_trace_slot is not None:
            live_trace_slot.empty()
        st.error(f"Run failed: {exc}")

_labeled = st.session_state.get("neutral_labeled_rows")
_analysis_warnings = st.session_state.get("analysis_warnings") or []
if _analysis_warnings:
    for _warning in _analysis_warnings:
        st.warning(_warning)
if _labeled:
    with st.expander("Section labels (Gemini)"):
        for row in _labeled:
            st.caption(
                f"**{row.get('display_title', '')}** → `{row.get('category', '')}`"
            )

dtc_results = st.session_state.get("dtc_results") or {}
_mode_used = st.session_state.get("analysis_mode_used")
if dtc_results and (not dev_mode or _mode_used != "ip_only"):
    st.subheader("DTC (Fan / Engagement)")
    insights = dtc_results.get("insights", "")
    evidence_by_section = dtc_results.get("evidence_by_section", {}) or {}
    st.caption("Grouped neutral summaries labeled DTC.")
    if insights:
        st.write(insights)
    else:
        st.write("No DTC-labeled sections in this pass.")
    with st.expander("Show evidence"):
        if evidence_by_section:
            for section_name, evidence in evidence_by_section.items():
                st.markdown(f"**{section_name}**")
                if evidence:
                    st.write(evidence)
                else:
                    st.write("_No bundled evidence._")
        else:
            st.write(
                "_Per-section Gemini excerpts are folded into summaries; "
                "see Other subsections for raw evidence where available._"
            )

ip_results = st.session_state.get("ip_results") or {}
if ip_results and (not dev_mode or _mode_used != "dtc_only"):
    st.subheader("IP (Catalog / Export)")
    evidence_by_section = ip_results.get("evidence_by_section", {}) or {}
    src_labels = sorted(evidence_by_section.keys())
    if src_labels:
        st.caption("Sources: " + ", ".join(src_labels))
    st.caption("Grouped neutral summaries labeled IP.")
    insights = ip_results.get("insights", "")
    if insights:
        st.write(insights)
    else:
        st.write("No IP-labeled sections in this pass.")
    with st.expander("Show evidence"):
        if evidence_by_section:
            for section_name, evidence in evidence_by_section.items():
                st.markdown(f"**{section_name}**")
                if evidence:
                    st.write(evidence)
                else:
                    st.write("_No bundled evidence._")
        else:
            st.write(
                "_Per-section excerpts are folded into summaries; see Other "
                "subsections for raw evidence where available._"
            )

section_results = st.session_state.get("section_results") or {}
if section_results and not dev_mode:
    st.subheader("Other sections")
    names = sorted(section_results.keys())
    st.caption("Sections labeled OTHER: " + ", ".join(names))
    for sec_name in names:
        result = section_results.get(sec_name) or {}
        st.markdown(f"**{sec_name}**")
        summary = result.get("summary", "")
        evidence = result.get("evidence", "")
        if summary:
            st.write(summary)
        else:
            st.write("_No summary returned._")
        with st.expander(f"Show evidence · {sec_name}"):
            if evidence:
                st.write(evidence)
            else:
                st.write("No evidence available.")

_react_trace = st.session_state.get("analysis_react_messages") or []
if dev_mode and _react_trace:
    _trace_expanded = not bool(st.session_state.get("neutral_labeled_rows"))
    with st.expander("Agent thinking (dev)", expanded=_trace_expanded):
        _render_react_trace(
            target=st,
            trace_rows=_react_trace,
            title="Final trace",
            key_prefix="final",
        )

st.divider()
st.subheader("Recommendations (Strategy)")
_strategy_help = (
    "Uses only the analysis above (Gemini strategy model). Run "
    "**Summarize & classify report** first, then generate."
)
st.caption(_strategy_help)

_bundle = build_strategy_bundle(
    st.session_state.get("dtc_results"),
    st.session_state.get("ip_results"),
    st.session_state.get("section_results"),
)
_strategy_ready = strategy_bundle_has_content(_bundle)
strategy_clicked = st.button(
    "Generate recommendations",
    disabled=not _strategy_ready,
    use_container_width=True,
    key="btn_strategy",
)
if not _strategy_ready:
    st.info(
        "Run **Summarize & classify report** on your PDF to enable this."
    )

if strategy_clicked and _strategy_ready:
    try:
        update_progress = _make_progress()
        update_progress(15, "Strategy: preparing bundle")
        t_strat0 = time.perf_counter()
        with st.spinner("Strategy agent (Gemini)..."):
            update_progress(45, "Strategy: Gemini inference")
            recs = run_strategy(
                st.session_state.get("dtc_results"),
                st.session_state.get("ip_results"),
                st.session_state.get("section_results"),
                model_name=None,
            )
            update_progress(90, "Strategy: parsing recommendations")
        st.session_state["last_strategy_timings"] = {
            "strategy_pipeline_s": time.perf_counter() - t_strat0,
        }
        st.session_state["strategy_results"] = recs
        update_progress(100, "Done")
        if not recs:
            st.warning(
                "No recommendations parsed. Check Gemini output or try again."
            )
    except Exception as exc:
        st.session_state["last_strategy_timings"] = None
        st.error(f"Strategy run failed: {exc}")

_strat = st.session_state.get("strategy_results")
if _strat:
    for i, rec in enumerate(_strat, start=1):
        title = rec.get("title") or f"Recommendation {i}"
        with st.container():
            st.markdown(f"#### {i}. {title}")
            if rec.get("insight"):
                st.markdown(rec["insight"])
            if rec.get("evidence"):
                st.markdown(f"- **Evidence:** {rec['evidence']}")
            if rec.get("action"):
                st.markdown(f"- **Action:** {rec['action']}")
            st.markdown("")

_render_run_timings_expander()
