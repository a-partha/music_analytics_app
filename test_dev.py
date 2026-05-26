from src.graphs.state import AnalysisState
from src.agents.analysis.nodes import filter_manifest_node

state = {
    "run_profile": "dev_one_per_category",
    "manifest": [
        {"subsection_key": "a", "display_title": "Engagement Horizon"},
        {"subsection_key": "b", "display_title": "Streaming Atlas"},
        {"subsection_key": "c", "display_title": "Import / Export"},
        {"subsection_key": "d", "display_title": "Midyear Metrics"},
        {"subsection_key": "e", "display_title": "Artist Spectrum"},
    ]
}

res = filter_manifest_node(state)
print("Filtered length:", len(res["manifest"]))
for r in res["manifest"]:
    print(" -", r["display_title"])
