from __future__ import annotations

from langchain_core.tools import tool

from src.services.file_search_retrieval import retrieve_subsection_evidence


@tool
def retrieve_subsection_evidence_tool(
    file_search_store_name: str,
    subsection_key: str,
    display_title: str,
    source_filename: str,
    model_name: str | None = None,
) -> str:
    """Retrieve neutral EARLY/MIDDLE/LATE excerpts for a dynamic subsection."""
    return retrieve_subsection_evidence(
        file_search_store_name=file_search_store_name,
        subsection_key=subsection_key,
        display_title=display_title,
        source_filename=source_filename,
        model_name=model_name,
    )
