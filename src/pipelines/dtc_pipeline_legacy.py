"""Legacy DTC pipeline preserved for runtime A/B switching.

This is a verbatim copy of the original `run_dtc_insights_pipeline` that
used the generic `retrieve_section_context` (EARLY/MIDDLE/LATE bucketed
excerpts) for DTC sections. Kept here as a safety net while the new
DTC-targeted retrieval path stabilizes. Not imported by default.

To compare results, manually swap the import in
`src.pipelines.analysis_pipeline` (or in the Streamlit app) for this
function.
"""

from __future__ import annotations

from src.chains.dtc_insights_chain import summarize_dtc_insights
from src.services.file_search_retrieval import retrieve_section_context

DTC_SECTION_NAMES = (
    "Engagement Horizon",
)


def run_dtc_insights_pipeline_legacy(
    file_search_store_name: str,
    section_names: tuple[str, ...] = DTC_SECTION_NAMES,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> dict[str, object]:
    evidence_by_section: dict[str, str] = {}
    for section_name in section_names:
        evidence_by_section[section_name] = retrieve_section_context(
            file_search_store_name=file_search_store_name,
            section_name=section_name,
            source_filename=source_filename,
            model_name=model_name,
        )

    combined_evidence = "\n\n".join(
        f"{section}:\n{evidence}"
        for section, evidence in evidence_by_section.items()
        if evidence
    )
    insights = summarize_dtc_insights(
        grounded_context=combined_evidence,
        model_name=model_name,
    )
    return {
        "insights": insights,
        "evidence_by_section": evidence_by_section,
    }
