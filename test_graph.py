from src.graphs.analysis_graph import get_analysis_graph
from src.graphs.state import AnalysisState

state = {
    "file_search_store_name": "test",
    "source_filename": "test.pdf",
    "run_profile": "dev_one_per_category",
    "manifest": [
        {"subsection_key": "a", "display_title": "Engagement Horizon"},
        {"subsection_key": "b", "display_title": "Streaming Atlas"},
        {"subsection_key": "c", "display_title": "Import / Export"},
        {"subsection_key": "d", "display_title": "Midyear Metrics"},
        {"subsection_key": "e", "display_title": "Artist Spectrum"},
    ],
    "neutral_rows": [],
    "errors": [],
    "gemini_model_name": "test",
    "synthesis_model_name": "test"
}

# mock the tools
import src.agents.analysis.nodes
src.agents.analysis.nodes.retrieve_subsection_evidence_tool.invoke = lambda *a, **k: "evidence"
src.agents.analysis.nodes.summarize_section_from_context = lambda *a, **k: "summary"
src.agents.analysis.nodes.label_neutral_rows_with_synthesis = lambda rows, **k: [
    {"display_title": r["display_title"], "category": "OTHER"} for r in rows
]
src.agents.analysis.nodes.pipeline_results_from_labeled_neutral_rows = lambda *a, **k: ({}, {}, {})
src.agents.analysis.nodes.get_neutral_row = lambda *a, **k: None
src.agents.analysis.nodes.put_neutral_row = lambda *a, **k: None

final = get_analysis_graph().invoke(state)
print("Labeled length:", len(final["labeled_rows"]))
print("Titles:")
for r in final["labeled_rows"]:
    print(" -", r["display_title"])
