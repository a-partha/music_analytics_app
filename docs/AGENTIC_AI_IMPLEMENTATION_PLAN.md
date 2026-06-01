# Music Analytics Pipeline — Agentic AI Implementation Plan

**Status:** Pass 1 and Pass 2 are implemented. This document describes the **current** app (post–repo cleanup), not a future roadmap.

---

## Project overview

**Goal:** Demo agentic AI for music-industry employers — grounded analysis and C-suite recommendations from a real industry PDF.

**Input:** Music industry reports.

**Stack:** Streamlit · Gemini (File Search + vision) · LangChain / LangGraph  
**Entry:** `python scripts/run.py` → `app/streamlit_app.py`

---

## Tech stack


| Layer         | Technology                  | Role                                                                            |
| ------------- | --------------------------- | ------------------------------------------------------------------------------- |
| UI            | Streamlit                   | Upload PDF, run analysis + strategy, show timings / trace / results             |
| Env           | `python-dotenv`             | `GEMINI_API_KEY`, model overrides                                               |
| Documents     | **Gemini File Search**      | Per-subsection PDFs indexed with metadata (`source_filename`, `subsection_key`) |
| PDF structure | **PyMuPDF** + Gemini vision | `section_splitter.py` detects subsection boundaries and slices the PDF          |
| LLM (chains)  | `langchain-google-genai`    | Summaries, batch labels, strategy text                                          |
| Orchestration | **LangGraph**               | `analysis_graph.py`, `strategy_graph.py`                                        |
| Tests         | pytest (`src/validation/`)  | No network; mocks for graph and ReAct                                           |


**Dependencies:** `requirements.txt` — no `langchain` meta-package, no `rapidfuzz` (title heuristics removed).

**Default models** (`src/services/langchain_llm.py`): analysis `gemini-3.1-flash-lite`, strategy `gemini-3.1-pro` (overridable via env).

---

## Repo layout (active code)

```text
app/streamlit_app.py          # UI
docs/                         # This plan + sample PDFs
scripts/run.py                # Streamlit launcher
scripts/create_file_search_store.py
src/
  config/                     # RunProfile, AnalysisMode
  graphs/                     # analysis_graph, strategy_graph, state
  agents/analysis/            # Graph nodes + ReAct agents/tools/trace
  agents/strategy/            # Strategy graph nodes + postprocess
  chains/                     # LCEL: summaries, labels, strategy
  pipelines/                  # run_analysis, run_strategy facades
  services/                   # File Search, splitter, cache, bundle
  tools/                      # retrieve_subsection_evidence_tool
  validation/                 # pytest suite
```

**Removed / archived** (not in working tree; recover from git tag `snapshot/legacy-unused-20260528-1416` if needed): legacy per-section DTC/IP insight chains, legacy PDF splitter, fuzzy `assign_category`, DTC/IP-specific retrieval helpers.

---

## End-to-end flow

```text
Streamlit: upload PDF
  ↓
Vision dynamic split (section_splitter) → manifest[{subsection_key, display_title}]
  ↓
Upload each subsection PDF to Gemini File Search (cached by pdf hash)
  ↓
[Analysis LangGraph]  ← run_profile + (dev only) analysis_mode
  ↓
dtc_results, ip_results, section_results  (from LLM labels, not title heuristics)
  ↓
[Strategy LangGraph]  ← bundle = DTC + IP insights text only
  ↓
Streamlit: DTC / IP / Other blocks + recommendation cards
```

---

## Ingestion and manifest

1. User uploads a full report PDF.
2. `**split_pdf_into_dynamic_slices**` renders pages, asks Gemini vision for subsection titles and page ranges, emits one mini-PDF per subsection.
3. `**ensure_sections_in_store_from_bytes**` uploads each slice to File Search with metadata; section list is cached on disk (`section_cache.json`).

There is **only** the dynamic splitter path (no legacy fixed-section upload mode).

---

## Analysis LangGraph (`src/graphs/analysis_graph.py`)

**Shared start:** `filter_manifest` → full manifest (no pre-filter by title).

**Branch on `run_profile`:**

### Full profile (`RunProfile.FULL`)

Checkbox **off** in Streamlit (authoritative; env vars do not override unchecked box).

```text
parallel_router
  → summarize_subsection (parallel, neutral evidence + summary, disk cache)
  → label_sections (batch LLM: DTC / IP / OTHER per row)
  → assemble_outputs
```

- Labeling: `section_label_chain.label_neutral_rows_with_synthesis`
- Categories come **only** from the LLM label JSON, not fuzzy title matching.

### Dev profile (`RunProfile.DEV_ONE_PER_CATEGORY`)

Checkbox **on** + required **analysis mode** radio: **DTC only** · **IP only** · **Both**.

```text
react_mode_router
  → dtc_react_node | ip_react_node | both_react_node
  → assemble_outputs_react
```

**ReAct agent** (`react_agents.py` + `react_tools.py`):

- Tools: list subsections, fetch+summarize subsection, judge (DTC / IP / DTC+IP), finish
- Judge + validator chains in `section_label_single_chains.py`
- Live + final trace in Streamlit (`trace_formatter.py`)

**Dev assembly rules** (`assemble_outputs_react_node`):


| Mode     | Success                  | Failure                                                         |
| -------- | ------------------------ | --------------------------------------------------------------- |
| DTC only | ≥1 accepted **DTC** row  | Hard fail (no fallback bundle)                                  |
| IP only  | ≥1 accepted **IP** row   | Hard fail                                                       |
| Both     | ≥1 **DTC** and ≥1 **IP** | Partial OK: show what was found + warning if one target missing |
| Both     | Neither target           | Hard fail                                                       |


Output is capped to **one row per target category** (not three-way DTC/IP/OTHER sampling).

**Note:** Full dev path used to mean “one subsection per category before labeling”; that heuristic path is **gone**. Dev reduction is **after** ReAct judging (or after batch label in any future reuse of full graph + dev cap).

---

## Strategy LangGraph (`src/graphs/strategy_graph.py`)

```text
build_bundle → generate_strategy → parse_and_dedupe → grounded_check (optional retry)
```

- **Input bundle** (`strategy_bundle.py`): concatenated **DTC** and **IP** insight text only — no OTHER sections, no raw evidence in the bundle.
- In dev mode, bundle reflects whichever categories the ReAct pass accepted (e.g. DTC-only run → DTC block only).
- **Output:** 3–5 recommendations (`title`, `insight`, `evidence`, `action`) shown in Streamlit.

---

## Streamlit surface

- **Dev mode** checkbox + **analysis mode** radio (when dev on)
- **Run analysis** → progress, timings expander
- **DTC / IP / Other** sections with summaries and evidence expanders
- **Agent thinking (dev)** — ReAct trace when dev profile was used
- **Recommendations (Strategy)** — executive cards

---

## Conceptual agents (as implemented)

### Analysis

**Inputs:** File Search store name, source filename, manifest, `pdf_hash` (for neutral summary cache), `run_profile`, optional `analysis_mode`.

**Outputs:**

- `dtc_results` / `ip_results`: `{ insights, evidence_by_section }` built from **labeled** neutral rows
- `section_results`: OTHER-labeled subsections
- `neutral_labeled_rows` / labels for UI
- Dev: `react_messages` trace; warnings on partial Both-mode success

**Not used anymore:** separate `run_dtc_insights_pipeline` / `run_ip_insights_pipeline`, per-section DTC/IP retrieval tools, title-based `assign_category`.

### Strategy

**Inputs:** Analysis dicts or pre-built `analysis_bundle` string.

**Behavior:** Grounded recommendations with evidence tied to report content; grounded-check node may trigger one regeneration.

---

## Implementation passes (historical)

### Pass 1 — Text-only analysis ✅

Grounded subsection summaries and DTC/IP/OTHER classification via LLM (full parallel path + dev ReAct path).

### Pass 2 — Strategy agent ✅

LangGraph strategy pipeline + Streamlit recommendation cards.

### LangGraph migration ✅

- `src/graphs/`, `src/agents/`, `src/tools/`, `src/config/`
- Dual analysis paths: **full** (`parallel_router`) vs **dev ReAct**
- Public API: `run_analysis()` / `run_strategy()` in `src/pipelines/`

---

## Checklist

### Done

- Streamlit shell, env config, File Search upload
- Dynamic vision subsection split + per-subsection indexing
- Analysis LangGraph (full + dev ReAct)
- Neutral summary disk cache
- LLM section labeling (batch + ReAct judge/validator)
- Strategy LangGraph + UI
- Unit tests under `src/validation/` (49+ tests)
- Repo cleanup: legacy chains/splitter/heuristics removed

### Remaining (manual / polish)

- QA on a full Luminate PDF (full profile end-to-end)
- QA dev modes: DTC only, IP only, Both (including partial Both warning path)
- Demo polish (copy, layout, employer-facing narrative)

---

## Success criteria

### Technical

- Arbitrary Luminate-style PDFs ingest via dynamic split
- File Search grounding for subsection evidence
- Full profile scales to full manifest; dev profile stays fast and mode-targeted
- Strategy consumes analysis without brittle JSON coupling (bundle text)

### Demo quality

- Clearly **agentic** (ReAct trace in dev; parallel graph in full)
- **Grounded** narrative with section-level evidence in UI
- **Executive** strategy output with evidence + action per recommendation

---

## Git snapshots (recovery)


| Tag                                    | Contents                                                    |
| -------------------------------------- | ----------------------------------------------------------- |
| `snapshot/dev-full-20260528-1416`      | App snapshot before local cleanup (dev ReAct + full linear) |
| `snapshot/legacy-unused-20260528-1416` | Four archived legacy modules only                           |


Local cleanup was working-tree only; commit when ready.