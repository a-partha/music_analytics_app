# Music Analytics Pipeline: Agentic AI Implementation

---

## Table of contents

- [Project overview](#project-overview)
- [LLM usage (Gemini)](#llm-usage-gemini)
- [Tech stack](#tech-stack)
- [Repo layout](#repo-layout)
- [End-to-end flow](#end-to-end-flow)
- [Ingestion and manifest](#ingestion-and-manifest)
- [Graphs, nodes, and agents](#graphs-nodes-and-agents)
- [Analysis LangGraph](#analysis-langgraph-srcgraphsanalysis_graphpy)
- [Strategy LangGraph](#strategy-langgraph-srcgraphsstrategy_graphpy)
- [Streamlit demo surface](#streamlit-demo-surface-appstreamlit_apppy)
- [Demo quality](#demo-quality)
- [Reference](#reference)
  - [Config](#config)
  - [LCEL chains](#lcel-chains)
  - [Services (key modules)](#services-key-modules)

---

## Project overview

**Goal:** Demo agentic AI for music-industry reporting: LLM-powered, grounded analysis and C-suite recommendations from a real industry PDF.

**Input:** Music industry report (PDF).

**Stack:** Streamlit · **Google Gemini LLMs** (vision, File Search RAG, LangChain chains, ReAct agent) · LangGraph  
**Live demo:** [music-agentic-analytics.streamlit.app](https://music-agentic-analytics.streamlit.app/) (Streamlit Community Cloud)  
**Local entry:** `python scripts/run.py` → `app/streamlit_app.py`

All analysis and strategy text is produced by LLMs. There are no rule-based insight generators or title heuristics. Classification, summarization, retrieval synthesis, agent reasoning, and executive recommendations each call Gemini with structured prompts.

---

## LLM usage (Gemini)

Every “intelligent” step in the pipeline is an LLM call. Two SDK paths are used:


| SDK                                                 | Used for                                                           |
| --------------------------------------------------- | ------------------------------------------------------------------ |
| `google-genai` (`genai.Client`)                     | PDF vision split, File Search retrieval (RAG over subsection PDFs) |
| `langchain-google-genai` (`ChatGoogleGenerativeAI`) | Summaries, labels, ReAct agent, strategy generation                |


### Where LLMs run


| Stage                    | Module                           | Model (default)         | What the LLM does                                                             |
| ------------------------ | -------------------------------- | ----------------------- | ----------------------------------------------------------------------------- |
| **PDF structure**        | `section_splitter.py`            | `gemini-2.5-flash`      | Vision over rendered pages: detect subsection titles and page ranges          |
| **Evidence retrieval**   | `file_search_retrieval.py`       | `gemini-2.5-flash`      | File Search RAG: pull EARLY/MIDDLE/LATE excerpts from indexed subsection PDFs |
| **Neutral summaries**    | `section_summary_chain.py`       | `gemini-3.1-flash-lite` | 3–4 bullet executive summary from retrieved evidence (full path + ReAct tool) |
| **Batch classification** | `section_label_chain.py`         | `gemini-3.1-flash-lite` | JSON label each summarized row as DTC / IP / OTHER (full profile only)        |
| **ReAct agent**          | `react_agents.py`                | `gemini-3.1-flash-lite` | Agent loop: choose subsections, call tools, decide when to finish             |
| **Per-row judge**        | `section_label_single_chains.py` | `gemini-3.1-flash-lite` | Classify one subsection for DTC/IP fit (ReAct judge tools)                    |
| **Label validator**      | `section_label_single_chains.py` | `gemini-3.1-flash-lite` | Second-pass ACCEPT/REJECT on judge output before accepting a row              |
| **Strategy brief**       | `strategy_chain.py`              | `gemini-3.1-pro`        | Markdown executive recommendations from the analysis bundle                   |


**Env overrides** (`src/services/langchain_llm.py`, retrieval, splitter):


| Variable                 | Affects                                                     |
| ------------------------ | ----------------------------------------------------------- |
| `GEMINI_API_KEY`         | All Gemini calls (required)                                 |
| `GEMINI_ANALYSIS_MODEL`  | LangChain analysis chains + ReAct agent                     |
| `GEMINI_STRATEGY_MODEL`  | Strategy chain                                              |
| `GEMINI_MODEL`           | Fallback for any unset model; also retrieval + vision split |
| `GEMINI_SYNTHESIS_MODEL` | Alias for analysis model in summary cache keys              |


**Demo path LLM sequence** (single DTC or IP run):

```text
Vision LLM (split PDF)
  → File Search RAG LLM (per subsection examined)
  → Summary LLM (per fetch)
  → Judge LLM + Validator LLM (per accepted candidate)
  → ReAct agent LLM (orchestrates the above via tool calls)
  → Strategy LLM (executive brief from accepted insights)
```

Postprocess steps (`postprocess.py`, `strategy_bundle.py`) are deterministic (parse, dedupe, filter). They do not call an LLM.

---

## Tech stack


| Layer         | Technology                      | Role                                                                                 |
| ------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| UI            | Streamlit                       | Upload PDF, run analysis + strategy, live timer, ReAct trace, executive cards        |
| Env           | `python-dotenv`                 | `GEMINI_API_KEY`, model overrides                                                    |
| LLM provider  | Google Gemini                   | All summarization, classification, retrieval synthesis, agent reasoning, strategy    |
| Documents     | Gemini File Search              | Per-subsection PDFs indexed with metadata (`source_filename`, `subsection_key`)      |
| PDF structure | PyMuPDF + Gemini vision         | `section_splitter.py`: LLM reads page images to find subsection boundaries           |
| LLM (chains)  | `langchain-google-genai`        | LCEL** chains: summaries, batch labels, strategy text                                 |
| Agents        | `langchain.agents.create_agent` | ReAct analysis agent: LLM selects tools and drives the dev profile loop              |
| Orchestration | LangGraph                       | `analysis_graph.py`, `strategy_graph.py`: wires LLM and deterministic workflow nodes |
| Tests         | pytest (`src/validation/`)      | No network; mocks for graph and ReAct (49 tests)                                     |


**Dependencies** (`requirements.txt`): `streamlit`, `python-dotenv`, `google-genai`, `langchain`, `langchain-core`, `langchain-google-genai`, `langgraph`, `PyMuPDF`.

**Default models** (`src/services/langchain_llm.py`): analysis `gemini-3.1-flash-lite`, strategy `gemini-3.1-pro` (overridable via `GEMINI_ANALYSIS_MODEL`, `GEMINI_STRATEGY_MODEL`, `GEMINI_MODEL`).

**Streamlit config** (`.streamlit/config.toml`): light + dark themes (warm orange palette), `maxUploadSize = 100` MB.

*LCEL (LangChain Expression Language) is LangChain's pipe-based syntax for composing prompt and model steps into chains. This app uses LCEL for summaries, batch labels, and strategy text (see [Reference](#reference) for module files).*

---

## Repo layout

```text
app/streamlit_app.py          # Demo UI (executive layout, custom CSS, theme-aware)
docs/                         # This plan + sample PDFs
scripts/run.py                # Streamlit launcher
scripts/create_file_search_store.py
.streamlit/config.toml        # Theme + upload limit
src/
  config/                     # RunProfile, AnalysisMode
  graphs/                     # analysis_graph, strategy_graph, state
  agents/analysis/            # Graph nodes + ReAct agent/tools/trace
  agents/strategy/            # Strategy graph nodes + postprocess
  chains/                     # LCEL chains: summaries, labels, strategy
  pipelines/                  # run_analysis, run_strategy facades
  services/                   # File Search, splitter, cache, bundle
  tools/                      # retrieve_subsection_evidence_tool
  validation/                 # pytest suite (14 files)
```

---

## End-to-end flow

### Demo profile

```text
Streamlit: choose strategic focus (DTC or IP) → upload PDF → Run analysis
  ↓
Vision LLM (section_splitter) → manifest[{subsection_key, display_title}]
  ↓
Upload each subsection PDF to Gemini File Search (cached by pdf hash)
  ↓
[Analysis LangGraph: DEV ReAct profile, single focus mode]
  ReAct agent LLM + retrieval/summary/judge/validator LLMs per tool call
  ↓
dtc_results and/or ip_results, section_results, neutral_labeled_rows
  ↓
[Strategy LangGraph]  ← bundle = DTC + IP insight text only → strategy LLM
  ↓
Streamlit: insight cards + evidence expanders + ReAct trace + recommendation cards
```

### Full profile

```text
[Analysis LangGraph: FULL profile]
  parallel_router → summarize_subsection → label_sections → assemble_outputs
  ↓
All manifest rows summarized and batch-labeled (DTC / IP / OTHER)
  ↓
Strategy graph as above
```

---

## Ingestion and manifest

1. User uploads a full report PDF (max 100 MB).
2. `split_pdf_into_dynamic_slices` renders pages, asks Gemini vision for subsection titles and page ranges, emits one mini-PDF per subsection.
3. `ensure_sections_in_store_from_bytes` uploads each slice to File Search with metadata; section list is cached on disk (`section_cache.json`).

---

## Graphs, nodes, and agents

- Graph = the workflow container (state + nodes + edges)
- Node = a function; can contain an agent or just deterministic code
- Agent = LLM-powered actor working toward a goal; lives inside a node

The app has two graphs, each mixing agent nodes with deterministic nodes:


| Graph                  | Agent node(s)                                                                                                                                          | Deterministic nodes                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Analysis (dev branch)  | `dtc_react_node` / `ip_react_node` / `both_react_node`: ReAct agent; tool-calling loop (list subsections → fetch → judge → finish)                     | `filter_manifest`, `react_mode_router`, `assemble_outputs_react` |
| Analysis (full branch) | `summarize_subsection`, `label_sections`: structured agents; single-shot LLM summarization/classification, no tool loop                                | `parallel_router`, `assemble_outputs`                            |
| Strategy               | `generate_strategy`: structured agent; goal-directed structured output (executive recommendations), with a conditional retry loop via `grounded_check` | `build_bundle`, `parse_and_dedupe`, `grounded_check`             |


---

## Analysis LangGraph (`src/graphs/analysis_graph.py`)

**Shared start:** `filter_manifest` → full manifest (no pre-filter by title; dev reduction happens after ReAct judging).

**Branch on `run_profile`:**

### Full profile (`RunProfile.FULL`)

Not exposed in the demo UI. Still available via API, env (`PIPELINE_RUN_PROFILE=full`), or tests.

```text
parallel_router
  → summarize_subsection (parallel, neutral evidence + summary, disk cache)
  → label_sections (batch LLM: DTC / IP / OTHER per row)
  → assemble_outputs
```

- Labeling: `section_label_chain.label_neutral_rows_with_synthesis`

### Dev profile (`RunProfile.DEV_ONE_PER_CATEGORY`)

Hardcoded in Streamlit (`_analysis_profile = RunProfile.DEV_ONE_PER_CATEGORY`). Demo UI offers DTC only or IP only; combined mode is shown disabled.

```text
react_mode_router
  → dtc_react_node | ip_react_node | both_react_node
  → assemble_outputs_react
```

**ReAct agent** (`react_agents.py` via `langchain.agents.create_agent` + `react_tools.py`):

- Tools: list subsections, fetch+summarize subsection, judge (DTC / IP / DTC+IP), finish
- Judge + validator chains in `section_label_single_chains.py`
- Live + final trace in Streamlit (`trace_formatter.py`)

**Dev assembly rules** (`assemble_outputs_react_node`):


| Mode     | Success             | Failure                                                         |
| -------- | ------------------- | --------------------------------------------------------------- |
| DTC only | ≥1 accepted DTC row | Hard fail (no fallback bundle)                                  |
| IP only  | ≥1 accepted IP row  | Hard fail                                                       |
| Both     | ≥1 DTC and ≥1 IP    | Partial OK: show what was found + warning if one target missing |
| Both     | Neither target      | Hard fail                                                       |


Output is capped to one row per target category (not three-way DTC/IP/OTHER sampling).

**Inputs:** File Search store name, source filename, manifest, `pdf_hash` (for neutral summary cache), `run_profile`, optional `analysis_mode`.

**Outputs:**

- `dtc_results` / `ip_results`: `{ insights, evidence_by_section }` built from labeled neutral rows
- `section_results`: OTHER-labeled subsections
- `neutral_labeled_rows` / labels for UI
- Dev: `react_messages` trace; warnings on partial Both-mode success

---

## Strategy LangGraph (`src/graphs/strategy_graph.py`)

```text
build_bundle → generate_strategy → parse_and_dedupe → grounded_check (optional retry)
```

- `generate_strategy` is the structured agent node here: LLM-powered, goal-directed output (executive recommendations). `build_bundle`, `parse_and_dedupe`, and `grounded_check` are deterministic (assemble inputs, parse markdown, dedupe, route).
- **Input bundle** (`strategy_bundle.py`): concatenated **DTC** and **IP** insight text only; no OTHER sections, no raw evidence in the bundle.
- In dev mode, bundle reflects whichever categories the ReAct pass accepted (e.g. DTC-only run → DTC block only).
- **Postprocess** (`postprocess.py`): markdown parse, Jaccard dedupe, file-agnostic grounded filter (no hardcoded regions or domain stopwords).
- **Output:** 3–5 recommendations (`title`, `insight`, `evidence`, `action`) shown in Streamlit.

**Inputs:** Analysis dicts or pre-built `analysis_bundle` string.

**Behavior:** Grounded recommendations with evidence tied to report content; grounded-check node may trigger one regeneration.

---

## Streamlit demo surface (`app/streamlit_app.py`)

### Layout and design

- **Hero:** eyebrow, title, description, tech stack line (`Streamlit → Gemini File Search → LangGraph → LangChain → Gemini LLM`)
- **Two-column top row:** Configure (left) · Workspace (right), both in bordered containers
- **Two-column bottom row:** Insights (left) · Recommendations (right)
- **Custom CSS:** warm cream/orange palette, Google Sans/Roboto, card-based insight and recommendation UI
- **Dark mode:** `.streamlit/config.toml` light/dark themes + CSS variables synced via `st.context.theme` (auto-rerun on theme change)

### Demo behavior


| Control                  | Behavior                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------- |
| Strategic focus          | Radio: Fan & audience (DTC) or Catalog & IP (required before run)                   |
| Combined DTC+IP          | Checkbox shown disabled (“resource limits”)                                         |
| Run profile              | Always `DEV_ONE_PER_CATEGORY`; no dev/full toggle                                   |
| Run analysis             | Disabled until focus + PDF selected; primary button with live progress              |
| Live timer               | Elapsed time shown during analysis and strategy runs                                |
| ReAct trace              | Streamed live during run; persisted in “How the AI reached these insights” expander |
| Generate executive brief | Runs strategy graph; disabled until analysis bundle has content                     |


### Results display

- **Insight cards** with evidence expanders (DTC and/or IP depending on mode used)
- **Section classification detail** expander (labeled rows)
- **Recommendation cards** with title, insight, evidence, action
- **Session state** cleared on new PDF upload

### Entry points

- **Production:** [https://music-agentic-analytics.streamlit.app/](https://music-agentic-analytics.streamlit.app/)
- **Local:** `python scripts/run.py` or `streamlit run app/streamlit_app.py`
- `python app/streamlit_app.py` includes a guarded launcher when no Streamlit context exists

**Cloud notes:** `GEMINI_API_KEY` is set in Streamlit Cloud secrets. Ephemeral disk on Cloud means section and neutral-summary caches do not persist across app restarts or redeploys (each session may re-upload subsections to File Search).

---

## Demo quality

- **Agentic** (live ReAct trace; reasoning expander after run)
- **Grounded** narrative with section-level evidence in UI
- **Executive** strategy output with evidence + action per recommendation
- **Polished** visual design with light/dark theme support

---

## Reference

### Config

#### `RunProfile` (`src/config/run_profiles.py`)

- `FULL`: parallel summarize + batch label
- `DEV_ONE_PER_CATEGORY`: ReAct path; requires `analysis_mode`
- Resolution: explicit profile > `dev_mode_flag` > env (`PIPELINE_RUN_PROFILE` or `ANALYSIS_DEV_MODE`)

#### `AnalysisMode` (`src/config/analysis_mode.py`)

- `dtc_only`, `ip_only`, `both`
- Demo UI exposes DTC and IP only; `both` remains in graph/tests but is disabled in UI

### LCEL chains


| File                             | Purpose                                                    |
| -------------------------------- | ---------------------------------------------------------- |
| `section_summary_chain.py`       | Neutral 3–4 bullet summary from EARLY/MIDDLE/LATE evidence |
| `section_label_chain.py`         | Batch JSON labeling (DTC/IP/OTHER) for full parallel path  |
| `section_label_single_chains.py` | Per-row judge + validator chains for ReAct tools           |
| `strategy_chain.py`              | Markdown strategy recommendations from analysis bundle     |


### Services (key modules)


| Service                    | Role                                                                     |
| -------------------------- | ------------------------------------------------------------------------ |
| `file_search_store.py`     | Upload subsection PDFs; disk cache keyed by pdf hash                     |
| `file_search_retrieval.py` | Metadata-scoped File Search; EARLY/MIDDLE/LATE excerpts                  |
| `section_splitter.py`      | Vision-based dynamic PDF split (only path)                               |
| `manifest_filter.py`       | Pass-through manifest; dev cap via `limit_labeled_rows_one_per_category` |
| `section_categories.py`    | `DTC`, `IP`, `OTHER` constants                                           |
| `strategy_bundle.py`       | Build result dicts + strategy bundle text (DTC + IP only)                |
| `section_cache.py`         | PDF hash → store name, doc names, manifest                               |
| `neutral_summary_cache.py` | Always-on disk cache for neutral summaries                               |


---

