# Music Analytics Pipeline - Agentic AI Implementation Plan

## Project Overview

**Goal:** Demo showcasing agentic AI for music industry employers
**Data:** Luminate 2025 Midyear Music Report (PDF) - 7 sections
**Stack:** Streamlit + Gemini API (chat) + Gemini File Search + LangChain agent orchestration  
**Target:** C-suite level insights grounded in the actual report

---

## Tech Stack and Overall Flow

### **Tech stack**

- **Frontend:** Streamlit  
  - Upload Luminate PDF  
  - “Run analysis” button  
  - Display section summaries, DTC/IP insights, and C-suite recommendations
- **LLM:** Gemini API  
  - Chat model (e.g. `gemini-2.0-flash` or similar)  
  - Used for: retrieval-grounded summaries, qualitative DTC/IP insights, and (later) strategy recommendations
- **RAG / document access:** Gemini **File Search**  
  - Streamlit backend uploads the PDF to a File Search store  
  - Google handles chunking + embeddings + indexing  
  - All analysis calls are *grounded* on this uploaded file
- **Agent / orchestration layer:** **LangGraph + LangChain**  
  - Gemini File Search exposed as LangChain `@tool` wrappers (`src/tools/`)  
  - **Analysis graph** (`src/graphs/analysis_graph.py`): filter manifest → parallel subsection summaries → label → DTC/IP synthesis → assemble  
  - **Strategy graph** (`src/graphs/strategy_graph.py`): bundle → generate → parse → grounded check (optional retry)  
  - Ollama LCEL chains stay in `src/chains/`; node glue in `src/agents/`; public entry via `src/pipelines/`  
  - **Dev profile:** `PIPELINE_RUN_PROFILE=dev_one_per_category` or Streamlit checkbox — one subsection per DTC / IP / OTHER

---

## Unified Architecture Flow (High Level)

```text
Streamlit (upload PDF)
  ↓
Gemini File Search
  - Upload PDF to store
  - Get file_id / store reference
  ↓
[Analysis Agent]
  - Uses Gemini + File Search
  - Produces:
      • Section summaries (per section)
      • DTC (fan/engagement) insights
      • IP (catalog/export) insights
  ↓
[Strategy Agent]
  - Uses Analysis output (summaries + DTC + IP)
  - Produces 3–5 C-suite recommendations
  - Each with: insight + evidence (section / what the report states) + action
  ↓
Streamlit Dashboard
  - DTC/IP/Other summaries + evidence
  - DTC/IP Strategy recommendations
```

---

## Core Agents (Conceptual Responsibilities)

### **Analysis Agent (Sections + DTC + IP)**

**Goal:** Turn the Luminate PDF into a structured, human-readable analysis that stays faithful to the source.

**Inputs:**

- `file_id` (or equivalent) from Gemini File Search  
- List of section names to target (e.g. “Midyear Metrics”, “Streaming Atlas”, “Import/Export”, “Engagement Horizon”, “Artist Spectrum”, “Future in Focus”, “Midyear Charts”)

**Behavior:**

- For each section:
  - Query File Search with a section-anchored prompt  
    - Example: “In the ‘Streaming Atlas’ section of this report, summarize the key points in 3–4 bullets. Include headline stats only if they are simple and clear.”
- For DTC (fan/engagement) insights:
  - Ask targeted questions grounded in File Search (“What does the report say about superfans, gaming, podcasts, short-form video, Discord/Twitch/WhatsApp, etc.?”)
- For IP (catalog/export) insights:
  - Ask questions about export power, local vs foreign shares, cross-country flows (“Import/Export” + related sections)

**Outputs (conceptual shape):**

- **Section summaries:**  
  - Map of section name → 3–4 bullet summary (text; may quote figures when they appear in the evidence)
- **DTC insights:**  
  - 4–6 bullets on superfans, platforms, engagement patterns (text)
- **IP insights:**  
  - 4–6 bullets on catalog and export power (text)

---

### **Strategy Agent**

**Goal:** Turn Analysis Agent output into a small set of C-suite-ready recommendations.

**Inputs:**

- Section summaries from the Analysis Agent  
- DTC insights  
- IP insights  

**Behavior:**

- Generate 3–5 recommendations that each include:
  - **Insight:** what’s happening (e.g., “Superfans in gaming and short-form video are over-indexing in genre X”)  
  - **Evidence:** specific reference to report section and what it states (“Streaming Atlas shows …”, “Import/Export describes …”)  
  - **Action:** what a label, publisher, or rights holder should do (e.g., “Prioritize Discord/Twitch activations in markets A/B with superfans in genre Y”)
- Model is instructed to:
  - Only use information grounded in the uploaded report  
  - Explicitly mention the supporting section (or equivalent) for each recommendation  
  - Drop ideas that cannot be tied back to the report

**Outputs:**

- List of 3–5 recommendation objects with:
  - `title`  
  - `insight`  
  - `evidence` (human-readable citation: section + claim)  
  - `action`

---

## Implementation Passes

### **Pass 1 – Text-Only Analysis (Happy Path)**

**Goal:** End-to-end demo focused on grounded narrative (no separate metrics or charting pipeline).

**Behavior:**

- Streamlit:
  - Upload PDF  
  - Run analysis per category / section (as implemented)  
  - Show:
    - Section-by-section summaries  
    - A block of DTC insights  
    - A block of IP insights

**Scope limits:**

- No separate structured-metric or charting layer  
- No Strategy Agent yet (C-suite recommendations can wait)

**Success:**  
The app reads the report and produces grounded, reasonable text insights. Quantitative mentions may appear in prose when the evidence includes them; they are not fed through a dedicated extraction step.

---

### **Pass 2 – Strategy Agent**

**Goal:** Turn the analysis from Pass 1 into C-suite recommendations.

**Behavior:**

- Implement a Strategy Agent that:
  - Consumes:
    - Section summaries  
    - DTC/IP insights  
  - Produces:
    - 3–5 recommendations with explicit evidence and actions
- Streamlit:
  - Add a top-level “Executive Summary / Recommendations” section  
  - Show each recommendation with:
    - Short title  
    - 1–2 sentence explanation  
    - Bullet “Evidence: …” (pointing to the report section and claim)  
    - Bullet “Action: …”

**Scope limits:**

- No need for perfect automation or flawless reasoning; the demo can tolerate a bit of manual curation.  
- It is acceptable (and honest) to trim or lightly edit recommendations before presenting them.

**Success:**  
The app feels like an intelligent analyst that:

- Reads the Luminate report  
- Explains what is happening (text insights)  
- Tells a music executive **what to do next and why**, citing the report

---

## Success Criteria

### **Technical**

- **Streamlit** app can ingest arbitrary Luminate PDFs and run the full pipeline  
- **File Search** grounding works reliably (no hallucinations about sections that don’t exist)  
- **Analysis Agent** produces coherent, section-aware summaries and insights  
- **Strategy Agent** consumes prior text outputs without brittle coupling  

### **Demo Quality**

- Clearly demonstrates **agentic AI** grounded in a real industry report  
- **Narrative** (insights + recommendations) is clear, cited, and executive-oriented  
- Speaks the language of **music industry C-suite** (labels, publishers, artist teams)  
- Shows you understand both **AI tooling** (Gemini, File Search) and **music analytics**  
- Feels polished and opinionated enough to impress potential employers

---

## Implementation Checklist

### Pass 1 – Text-Only Analysis
1. [x] ~Create Streamlit app shell (upload + Run analysis)~
2. [x] ~Add Gemini API key config (env var or Streamlit secrets)~
3. [x] ~Upload PDF to Gemini File Search and capture `file_id`~
4. [x] ~Wrap Gemini chat + File Search in LangChain~
5. [x] ~Implement Analysis Agent for 7 sections~
6. [x] ~Add DTC insights query (fan/engagement)~
7. [x] ~Add IP insights query (catalog/export)~
8. [x] ~Render summaries + insights in Streamlit~

### Pass 2 – Strategy Agent
9. [x] Implement Strategy Agent (3–5 recommendations) — LangGraph `strategy_graph`
10. [x] Display recommendations with evidence + action — Streamlit cards

### LangGraph migration (orchestration)
- [x] `src/graphs/`, `src/agents/`, `src/tools/`, `src/config/` layout
- [x] Dev manifest filter (one section per category)
- [x] Analysis + Strategy graphs wired from Streamlit via `run_analysis` / `run_strategy`

### Final QA
11. [ ] QA on at least one full Luminate PDF (use full profile; dev profile for routine checks)
12. [ ] Demo polish (layout, copy, section labels)
