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
- [Evaluation](#evaluation)
- [Reference](#reference)
  - [Config](#config)
  - [LCEL chains](#lcel-chains)
  - [Services (key modules)](#services-key-modules)

---



## Project overview

**Goal:** Demo agentic AI for music-industry reporting: LLM-powered, grounded analysis and C-suite recommendations from a real industry PDF.

**Input:** Music industry report (PDF).

**Stack:** Streamlit · **Google Gemini LLMs** (vision, File Search RAG, LangChain chains, ReAct agent) · LangGraph · **DeepEval & NVIDIA Nemotron** (separate retrieval-and-answer evaluation; not part of an app run)  
**Live demo:** [music-agentic-analytics.streamlit.app](https://music-agentic-analytics.streamlit.app/) (Streamlit Community Cloud)  
**Local entry:** `python scripts/run.py` -> `app/streamlit_app.py`

All analysis and strategy text is produced by LLMs. There are no rule-based insight generators or title heuristics. Classification, summarization, retrieval synthesis, agent reasoning, and executive recommendations each call Gemini with structured prompts.

---



## LLM usage (Gemini)

Every intelligent step in an app run is driven by an LLM. The evaluation judge is documented under [Evaluation](#evaluation); it does not run inside the app.


| Path                                                        | Used for                                                           |
| ----------------------------------------------------------- | ------------------------------------------------------------------ |
| `google-genai` SDK (`genai.Client`)                         | PDF vision split, File Search retrieval (RAG over subsection PDFs) |
| `langchain-google-genai` package (`ChatGoogleGenerativeAI`) | Summaries, labels, ReAct agent, strategy generation                |




### Where LLMs run


| Stage                    | Module                           | Model (default)         | What the LLM does                                                                                                                       |
| ------------------------ | -------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **PDF structure**        | `section_splitter.py`            | `gemini-2.5-flash`      | Vision over rendered pages: detect subsection titles and page ranges                                                                    |
| **Evidence retrieval**   | `file_search_retrieval.py`       | `gemini-2.5-flash`      | File Search RAG: pull grounded excerpts from indexed subsection PDFs. Default is `GEMINI_MODEL`, which falls back to `gemini-2.5-flash` |
| **Neutral summaries**    | `section_summary_chain.py`       | `gemini-3.1-flash-lite` | 3-4 bullet executive summary from retrieved evidence (full path + ReAct tool)                                                           |
| **Batch classification** | `section_label_chain.py`         | `gemini-3.1-flash-lite` | JSON label each summarized row as DTC / IP / OTHER (full profile only)                                                                  |
| **ReAct agent**          | `react_agents.py`                | `gemini-3.1-flash-lite` | Agent loop: choose subsections, call tools, decide when to finish                                                                       |
| **Per-row judge**        | `section_label_single_chains.py` | `gemini-3.1-flash-lite` | Classify one subsection for DTC/IP fit (ReAct judge tools)                                                                              |
| **Label validator**      | `section_label_single_chains.py` | `gemini-3.1-flash-lite` | Second-pass ACCEPT/REJECT on judge output before accepting a row                                                                        |
| **Strategy brief**       | `strategy_chain.py`              | `gemini-3.1-pro`        | Markdown executive recommendations from the analysis bundle                                                                             |


**Env overrides** (`src/services/langchain_llm.py`, retrieval, splitter, and `run_evals.py`). The `GEMINI_`* model variables do not change the eval. The eval reads `EVAL_RETRIEVAL_MODEL`, `EVAL_GENERATOR_MODEL`, and `EVAL_JUDGE_MODEL`.


| Variable                 | Affects                                                      |
| ------------------------ | ------------------------------------------------------------ |
| `GEMINI_API_KEY`         | All Gemini calls (required)                                  |
| `NVIDIA_API_KEY`         | Evaluation judge only. The app does not read it              |
| `GEMINI_ANALYSIS_MODEL`  | LangChain analysis chains + ReAct agent                      |
| `GEMINI_STRATEGY_MODEL`  | Strategy chain                                               |
| `GEMINI_MODEL`           | Fallback for any unset model; also retrieval + vision split  |
| `GEMINI_SYNTHESIS_MODEL` | Alias for analysis model in summary cache keys               |
| `EVAL_RETRIEVAL_MODEL`   | Eval retrieval only. Default `gemini-3.1-flash-lite`         |
| `EVAL_GENERATOR_MODEL`   | Eval answers only. Defaults to `EVAL_RETRIEVAL_MODEL`        |
| `EVAL_JUDGE_MODEL`       | Eval judge only. Default `nvidia/nemotron-3-super-120b-a12b` |


---



## Tech stack


| Layer         | Technology                      | Role                                                                                    |
| ------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| UI            | Streamlit                       | Upload PDF, run analysis + strategy, live timer, ReAct trace, executive cards           |
| Env           | `python-dotenv`                 | `GEMINI_API_KEY`, `NVIDIA_API_KEY`, model overrides                                     |
| LLM provider  | Google Gemini                   | Summarization, classification, retrieval synthesis, agent reasoning, strategy           |
| Eval judge    | NVIDIA NIM                      | Nemotron-3 Super (120B). Used only by `src/validation/evals/run_evals.py`               |
| Documents     | Gemini File Search              | Per-subsection PDFs indexed with metadata (`source_filename`, `subsection_key`)         |
| PDF structure | PyMuPDF + Gemini vision         | `section_splitter.py`: LLM reads page images to find subsection boundaries              |
| LLM (chains)  | `langchain-google-genai`        | LCEL chains: summaries, batch labels, strategy text                                     |
| Agents        | `langchain.agents.create_agent` | ReAct analysis agent: LLM selects tools and drives the dev profile loop                 |
| Orchestration | LangGraph                       | `analysis_graph.py`, `strategy_graph.py`: wires LLM and deterministic workflow nodes    |
| Evaluation    | DeepEval                        | Hand-run retrieval-and-answer scores: Faithfulness, Answer Relevancy, Contextual Recall |
| Tests         | pytest (`src/validation/`)      | No network; mocks for graph and ReAct (59 tests)                                        |


---



## Repo layout

```text
app/
  streamlit_app.py            # Demo UI (executive layout, custom CSS, theme-aware)
docs/                         # Documentation + sample PDFs
scripts/
  run.py                      # Streamlit launcher
  create_file_search_store.py
.streamlit/
  config.toml                 # Theme + upload limit
src/
  config/                     # RunProfile, AnalysisMode
  graphs/                     # analysis_graph, strategy_graph, state
  agents/
    analysis/                 # Graph nodes + ReAct agent/tools/trace
    strategy/                 # Strategy graph nodes + postprocess
  chains/                     # LCEL chains: summaries, labels, strategy
  pipelines/                  # run_analysis, run_strategy facades
  services/                   # File Search, splitter, cache, bundle
  tools/                      # retrieve_subsection_evidence_tool
  validation/                 # pytest unit tests (no network)
    evals/                    # run_evals.py, questions, results, manual_review.json
      archive/                # superseded section-locked baseline; exits if run
requirements-eval.txt         # DeepEval + OpenAI client; not installed with the app
```

---



## End-to-end flow



### Demo profile

```text
Streamlit: choose strategic focus (DTC or IP) -> upload PDF -> Run analysis
  |
Vision LLM (section_splitter) -> manifest[{subsection_key, display_title}]
  |
Upload each subsection PDF to Gemini File Search (cached by pdf hash)
  |
[Analysis LangGraph: DEV ReAct profile, single focus mode]
  ReAct agent LLM + retrieval/summary/judge/validator LLMs per tool call
  |
dtc_results and/or ip_results, section_results, neutral_labeled_rows
  |
[Strategy LangGraph]  <- bundle = DTC + IP insight text only -> strategy LLM
  |
Streamlit: insight cards + evidence expanders + ReAct trace + recommendation cards
```



### Full profile

```text
[Analysis LangGraph: FULL profile]
  parallel_router -> summarize_subsection -> label_sections -> assemble_outputs
  |
All manifest rows summarized and batch-labeled (DTC / IP / OTHER)
  |
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
| Analysis (dev branch)  | `dtc_react_node` / `ip_react_node` / `both_react_node`: ReAct agent; tool-calling loop (list subsections -> fetch -> judge -> finish)                  | `filter_manifest`, `react_mode_router`, `assemble_outputs_react` |
| Analysis (full branch) | `summarize_subsection`, `label_sections`: structured agents; single-shot LLM summarization/classification, no tool loop                                | `parallel_router`, `assemble_outputs`                            |
| Strategy               | `generate_strategy`: structured agent; goal-directed structured output (executive recommendations), with a conditional retry loop via `grounded_check` | `build_bundle`, `parse_and_dedupe`, `grounded_check`             |


---



## Analysis LangGraph (`src/graphs/analysis_graph.py`)

**Shared start:** `filter_manifest` -> full manifest (no pre-filter by title; dev reduction happens after ReAct judging).

**Branch on** `run_profile`**:**

### Full profile (`RunProfile.FULL`)

Not exposed in the demo UI. Still available via API, env (`PIPELINE_RUN_PROFILE=full`), or tests.

```text
parallel_router
  -> summarize_subsection (parallel, neutral evidence + summary, disk cache)
  -> label_sections (batch LLM: DTC / IP / OTHER per row)
  -> assemble_outputs
```

- Labeling: `section_label_chain.label_neutral_rows_with_synthesis`



### Dev profile (`RunProfile.DEV_ONE_PER_CATEGORY`)

Hardcoded in Streamlit (`_analysis_profile = RunProfile.DEV_ONE_PER_CATEGORY`). Demo UI offers DTC only or IP only; combined mode is shown disabled.

```text
react_mode_router
  -> dtc_react_node | ip_react_node | both_react_node
  -> assemble_outputs_react
```

**ReAct agent** (`react_agents.py` via `langchain.agents.create_agent` + `react_tools.py`):

- Tools: list subsections, fetch+summarize subsection, judge (DTC / IP / DTC+IP), finish
- Judge + validator chains in `section_label_single_chains.py`
- Live + final trace in Streamlit (`trace_formatter.py`)

**Dev assembly rules** (`assemble_outputs_react_node`):


| Mode     | Success               | Failure                                                         |
| -------- | --------------------- | --------------------------------------------------------------- |
| DTC only | >= 1 accepted DTC row | Hard fail (no fallback bundle)                                  |
| IP only  | >= 1 accepted IP row  | Hard fail                                                       |
| Both     | >= 1 DTC and >= 1 IP  | Partial OK: show what was found + warning if one target missing |
| Both     | Neither target        | Hard fail                                                       |


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
build_bundle -> generate_strategy -> parse_and_dedupe -> grounded_check (optional retry)
```

- `generate_strategy` is the structured agent node here: LLM-powered, goal-directed output (executive recommendations). `build_bundle`, `parse_and_dedupe`, and `grounded_check` are deterministic (assemble inputs, parse markdown, dedupe, route).
- **Input bundle** (`strategy_bundle.py`): concatenated **DTC** and **IP** insight text only; no OTHER sections, no raw evidence in the bundle.
- In dev mode, bundle reflects whichever categories the ReAct pass accepted (e.g. DTC-only run -> DTC block only).
- **Postprocess** (`postprocess.py`): markdown parse, Jaccard dedupe, file-agnostic grounded filter (no hardcoded regions or domain stopwords).
- **Output:** 3-5 recommendations (`title`, `insight`, `evidence`, `action`) shown in Streamlit.

**Inputs:** Analysis dicts or pre-built `analysis_bundle` string.

**Behavior:** Grounded recommendations with evidence tied to report content; grounded-check node may trigger one regeneration.

---



## Streamlit demo surface (`app/streamlit_app.py`)



### Layout and design

- **Hero:** eyebrow, title, description, tech stack line (`Streamlit -> Gemini File Search -> LangGraph -> LangChain -> Gemini LLM`)
- **Two-column top row:** Configure (left) · Workspace (right), both in bordered containers
- **Two-column bottom row:** Insights (left) · Recommendations (right)
- **Custom CSS:** warm cream/orange palette, Google Sans/Roboto, card-based insight and recommendation UI
- **Dark mode:** `.streamlit/config.toml` light/dark themes + CSS variables synced via `st.context.theme` (auto-rerun on theme change)



### Demo behavior


| Control                   | Behavior                                                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Strategic focus           | Radio: Fan & audience (DTC) or Catalog & IP (required before run)                                                    |
| Combined DTC+IP           | Checkbox shown disabled ("resource limits")                                                                          |
| Run profile               | Always `DEV_ONE_PER_CATEGORY`; no dev/full toggle                                                                    |
| Run analysis              | Disabled until focus + PDF selected; primary button with live progress                                               |
| Live timer                | Elapsed time shown during analysis and strategy runs                                                                 |
| ReAct trace               | Streamed live during run; persisted in "How the AI reached these insights" expander                                  |
| Generate executive brief  | Runs strategy graph; disabled until analysis bundle has content                                                      |
| Download run audit (JSON) | Writes a local snapshot via `src/services/audit_pack.py`: mode, timings, labeled rows, insights, and the ReAct trace |




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
- **Measured** on retrieval and answers only. See [Evaluation](#evaluation). The agent, the DTC/IP judge, and the brief are not in that score.

---



## Evaluation

`src/validation/evals/run_evals.py` asks 25 questions about the indexed *Luminate 2025 Year-End Music Report* and scores the answers. It evaluates the File Search index the app builds. It does not run the ReAct agent, the DTC/IP judge, or the executive brief.

### Two tiers


| Tier                  | How to run         | Retrieval filter                                     |
| --------------------- | ------------------ | ---------------------------------------------------- |
| Component             | `--section-locked` | `source_filename` and `subsection_key`               |
| Report-wide (default) | no extra flag      | `source_filename` only, so every section of that PDF |


The component-tier flag is implemented. This page quotes the report-wide cold run only, because the section-locked run has not been executed yet.

The app's own retrieval (`retrieve_subsection_evidence`) is still locked to one section and defaults to `gemini-2.5-flash`. Only the eval script uses `retrieve_query_chunks_with_usage`.

### Models, limits, and labels


| Piece                | Setting                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| Retrieval model      | `EVAL_RETRIEVAL_MODEL`, default `gemini-3.1-flash-lite`                                                     |
| Answer model         | `EVAL_GENERATOR_MODEL`, default the retrieval model                                                         |
| Judge                | `EVAL_JUDGE_MODEL`, default `nvidia/nemotron-3-super-120b-a12b` via NVIDIA NIM, temperature 0, thinking off |
| Generator rate limit | 12 requests per minute                                                                                      |
| Judge rate limit     | 30 requests per minute                                                                                      |
| Metric pass line     | score >= 0.7 on Faithfulness, Answer Relevancy, and Contextual Recall                                       |
| Manual labels        | `manual_review.json`, counted only when the answer hash matches the reviewed text                           |


`--use-cache` reuses a saved answer when the retrieved text is unchanged. Retrieval still runs live. Cached questions are left out of the latency stats. `--summarize-only` rebuilds the summary with no API calls.

A question's label is ignored when the current answer hash does not match, and that row is counted as unreviewed.

### Report-wide cold run

Source: `smoke_test_results_cold.json`, summary rewritten by `--summarize-only` on 2026-10-07. Costs are list-price estimates (`pricing` field in the summary), not invoices. List prices in the script assume the default models (`gemini-3.1-flash-lite` and `nvidia/nemotron-3-super-120b-a12b`): production input $0.25 / 1M tokens, output $1.50 / 1M; judge input $0.08 / 1M, output $0.45 / 1M. The summary records those names under `financials.pricing_models`.


| Result                           | Value                |
| -------------------------------- | -------------------- |
| Contextual recall                | 25/25                |
| Answer relevancy                 | 22/25                |
| Faithfulness                     | 21/25                |
| Passed all three                 | 18/25                |
| Correct on manual review         | 24/25 (0 unreviewed) |
| Judge agreement with that review | 19/25                |


Latency is the retrieval call plus the answer call. It excludes rate-limit waits and failed retries. Median is the middle question; P95 is the script's 95th-percentile index, not the slowest question.


| Stage              | Mean         | Median       | P95          | Max          | Mean tokens in | Mean tokens out |
| ------------------ | ------------ | ------------ | ------------ | ------------ | -------------- | --------------- |
| Retrieval          | 9,215.11 ms  | 7,580.95 ms  | 18,897.16 ms | 21,216.28 ms | 9,050.12       | 109.68          |
| Answer             | 5,151.36 ms  | 1,169.43 ms  | 17,955.36 ms | 18,776.56 ms | 1,261.52       | 29.68           |
| Retrieval + answer | 14,366.47 ms | 11,677.81 ms | 27,089.14 ms | 39,171.64 ms |                |                 |
| Judge              | 10,073.98 ms | 10,233.33 ms | 13,213.86 ms | 14,073.79 ms | 6,093.56       | 837.12          |


Estimated spend for this run: $0.091279 total, of which $0.021607 is the judge.

### Where the 7 failures came from

Five were judge mistakes. The answer matched the report:


| Question                   | Failed metric    | What the judge did                                                          |
| -------------------------- | ---------------- | --------------------------------------------------------------------------- |
| `welcome_to_transmedia_02` | Relevancy 0.0    | Treated the film title in the report as fake                                |
| `welcome_to_transmedia_03` | Relevancy 0.0    | Treated the soundtrack in the report as nonexistent                         |
| `consumption_metrics_04`   | Faithfulness 0.0 | Read a flattened table and took on-demand streams for physical sales        |
| `evolving_fandom_03`       | Faithfulness 0.5 | Read 37% (Engaged) as the superfan share; the funnel lists superfans at 20% |
| `premium_pricing_03`       | Faithfulness 0.0 | Rejected "2025" because that sentence did not repeat the year               |


One caveat: `year_end_charts_02` (relevancy 0.5). The sales figure was right, and the model added that the chart did not state the year.

One real error: `ai_artists_03` (faithfulness 0.0). The retrieved text says 44% and the model wrote 43%. The judge was right to fail it. The reason it gave (29%) does not match the 44% line.

Judge agreement is 19/25 because the 18 correct passes and this one real error match the manual labels. The other 6 are the 5 judge mistakes plus the caveat.

`welcome_to_transmedia_04` passed relevancy at 1.0 with a reason that said the actual output was not provided. A pass can still be given for a bad reason. Agreement does not catch that.

### Historical section-locked baseline

`src/validation/evals/archive/run_evals_baseline.py` locked each question to its section and scored 17/25 (`archive/smoke_test_results.json`, 2026-10-06). The script exits immediately if you run it. The current `--section-locked` flag is the replacement; do not compare a new run to 17/25 unless you say the baseline is historical. One known difference: `welcome_to_transmedia_04` failed contextual recall in that archived run and passed in the report-wide cold run.

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
| `section_summary_chain.py`       | Neutral 3-4 bullet summary from EARLY/MIDDLE/LATE evidence |
| `section_label_chain.py`         | Batch JSON labeling (DTC/IP/OTHER) for full parallel path  |
| `section_label_single_chains.py` | Per-row judge + validator chains for ReAct tools           |
| `strategy_chain.py`              | Markdown strategy recommendations from analysis bundle     |




### Services (key modules)


| Service                    | Role                                                                      |
| -------------------------- | ------------------------------------------------------------------------- |
| `file_search_store.py`     | Upload subsection PDFs; disk cache keyed by pdf hash                      |
| `file_search_retrieval.py` | Metadata-scoped File Search; EARLY/MIDDLE/LATE excerpts                   |
| `section_splitter.py`      | Vision-based dynamic PDF split (only path)                                |
| `manifest_filter.py`       | Pass-through manifest; dev cap via `limit_labeled_rows_one_per_category`  |
| `section_categories.py`    | `DTC`, `IP`, `OTHER` constants                                            |
| `strategy_bundle.py`       | Build result dicts + strategy bundle text (DTC + IP only)                 |
| `section_cache.py`         | PDF hash -> store name, doc names, manifest                               |
| `neutral_summary_cache.py` | Always-on disk cache for neutral summaries                                |
| `audit_pack.py`            | JSON download of one demo run: mode, timings, rows, insights, ReAct trace |


