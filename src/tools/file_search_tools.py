from __future__ import annotations

from langchain_core.tools import tool

from src.services.file_search_retrieval import (
    retrieve_dtc_evidence,
    retrieve_ip_evidence,
    retrieve_subsection_evidence,
)


@tool
def retrieve_subsection_evidence_tool(
    file_search_store_name: str,
    subsection_key: str,
    display_title: str,
    source_filename: str,
    model_name: str | None = None,
) -> str:
    """Grounded excerpt for one PDF subsection via Gemini File Search."""
    return retrieve_subsection_evidence(
        file_search_store_name=file_search_store_name,
        subsection_key=subsection_key,
        display_title=display_title,
        source_filename=source_filename,
        model_name=model_name,
    )


@tool
def retrieve_dtc_evidence_tool(
    file_search_store_name: str,
    section_name: str,
    source_filename: str,
    model_name: str | None = None,
) -> str:
    """DTC / fan-engagement evidence for a named report section."""
    return retrieve_dtc_evidence(
        file_search_store_name=file_search_store_name,
        section_name=section_name,
        source_filename=source_filename,
        model_name=model_name,
    )


@tool
def retrieve_ip_evidence_tool(
    file_search_store_name: str,
    section_name: str,
    source_filename: str,
    model_name: str | None = None,
) -> str:
    """IP / catalog-export evidence for a named report section."""
    return retrieve_ip_evidence(
        file_search_store_name=file_search_store_name,
        section_name=section_name,
        source_filename=source_filename,
        model_name=model_name,
    )
