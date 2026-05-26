from __future__ import annotations

import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_FILE_SEARCH_TOP_K = 12
DEFAULT_FILE_SEARCH_TOP_K_CATEGORY = 24

MAX_RETRIES_ON_RATE_LIMIT = 3
DEFAULT_RETRY_DELAY_S = 6.0
RETRY_DELAY_BUFFER_S = 1.0

NO_EVIDENCE_MAX_ATTEMPTS = 2
NO_EVIDENCE_SHORT_THRESHOLD = 200
NO_EVIDENCE_ESCAPE_SENTENCES = (
    "no dtc-relevant content in this section.",
    "no ip-relevant content in this section.",
)


class RetrievalError(RuntimeError):
    pass


def _parse_retry_delay_seconds(message: str) -> float:
    match = re.search(
        r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE
    )
    if match:
        return float(match.group(1))
    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
        message,
    )
    if match:
        return float(match.group(1))
    return DEFAULT_RETRY_DELAY_S


def _is_rate_limit_error(exc: Exception) -> bool:
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return "resource_exhausted" in text or "429" in text


def _generate_with_backoff(
    client: "genai.Client",
    *,
    model: str,
    contents: object,
    config: object,
) -> object:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            is_rate_limit = _is_rate_limit_error(exc)
            if not is_rate_limit:
                raise
            last_exc = exc
            if attempt + 1 >= MAX_RETRIES_ON_RATE_LIMIT:
                break
            delay = _parse_retry_delay_seconds(str(exc)) + RETRY_DELAY_BUFFER_S
            time.sleep(delay)
    raise RetrievalError(
        "Gemini rate limit not cleared after "
        f"{MAX_RETRIES_ON_RATE_LIMIT} attempts. Last error: {last_exc}"
    )


def _escape_metadata_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _looks_like_no_evidence(text: str) -> bool:
    if not text or not text.strip():
        return True
    stripped = text.strip()
    normalized = stripped.lower().rstrip(".").strip()
    escape_normalized = {
        sentence.lower().rstrip(".").strip()
        for sentence in NO_EVIDENCE_ESCAPE_SENTENCES
    }
    if normalized in escape_normalized:
        return True
    if len(stripped) >= NO_EVIDENCE_SHORT_THRESHOLD:
        return False
    phrases = (
        "unable to find",
        "could not find",
        "cannot find",
        "can't find",
        "not found in the document",
        "section was not found",
        "no evidence",
        "i am sorry",
        "i'm sorry",
        "not able to find",
    )
    lowered = stripped.lower()
    return any(phrase in lowered for phrase in phrases)


def _retrieve_with_no_evidence_retry(
    client: "genai.Client",
    *,
    model: str,
    contents: object,
    config: object,
    section_name: str,
    label: str,
) -> str:
    last_text = ""
    for attempt in range(NO_EVIDENCE_MAX_ATTEMPTS):
        response = _generate_with_backoff(
            client,
            model=model,
            contents=contents,
            config=config,
        )
        raw_text = (response.text or "").strip()
        looks_like_empty = _looks_like_no_evidence(raw_text)
        if not looks_like_empty:
            return raw_text
        last_text = raw_text
    raise RetrievalError(
        f"No {label} evidence retrieved for section '{section_name}' after "
        f"{NO_EVIDENCE_MAX_ATTEMPTS} attempts. Last response: "
        f"{last_text[:200]}"
    )


def retrieve_section_context(
    file_search_store_name: str,
    section_name: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)

    metadata_filter = (
        f'source_filename = "{source_filename}" '
        f'AND section = "{section_name}"'
    )

    prompt = (
        f"Section: {section_name}\n\n"
        "Return up to 9 short excerpts or data points from this section.\n"
        "Bucket the output to ensure coverage:\n"
        "- EARLY: 2-3 items from the start of the section.\n"
        "- MIDDLE: 2-3 items from the middle of the section.\n"
        "- LATE: 2-3 items from the end of the section.\n"
        "Each item on its own line (bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name=section_name,
        label="generic",
    )


def retrieve_subsection_evidence(
    file_search_store_name: str,
    subsection_key: str,
    display_title: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    """Neutral EARLY/MIDDLE/LATE excerpts for a dynamic subsection upload."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)
    sf = _escape_metadata_filter_value(source_filename)
    sk = _escape_metadata_filter_value(subsection_key)
    metadata_filter = (
        f'source_filename = "{sf}" AND subsection_key = "{sk}"'
    )

    prompt = (
        f"Section: {display_title}\n\n"
        "Return up to 9 short excerpts or data points from this section.\n"
        "Bucket the output to ensure coverage:\n"
        "- EARLY: 2-3 items from the start of the section.\n"
        "- MIDDLE: 2-3 items from the middle of the section.\n"
        "- LATE: 2-3 items from the end of the section.\n"
        "Each item on its own line (bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name=display_title,
        label="generic",
    )


def retrieve_dtc_evidence(
    file_search_store_name: str,
    section_name: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)

    metadata_filter = (
        f'source_filename = "{source_filename}" '
        f'AND section = "{section_name}"'
    )

    prompt = (
        f"Section: {section_name}\n\n"
        "Find all excerpts and data points in this section that relate to "
        "Direct-to-Consumer (DTC) fan and engagement themes. Themes include "
        "but are not limited to:\n"
        "- Superfans and fan tiers / spending behavior\n"
        "- Gaming and in-game music integrations (Roblox, Fortnite, etc.)\n"
        "- Podcasts and audio formats beyond music\n"
        "- Short-form video (TikTok, Reels, Shorts) discovery and behavior\n"
        "- Social and community platforms (Discord, Twitch, WhatsApp, "
        "Reddit) used for fan engagement\n"
        "- Live, ticketing, merch, and direct artist-to-fan monetization\n"
        "- Platform-specific listening and engagement patterns\n\n"
        "Return up to 9 short excerpts. Each item on its own line "
        "(bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
        "If this section contains no DTC-relevant content, reply exactly: "
        "'No DTC-relevant content in this section.'\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name=section_name,
        label="DTC",
    )


def retrieve_ip_evidence(
    file_search_store_name: str,
    section_name: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)

    metadata_filter = (
        f'source_filename = "{source_filename}" '
        f'AND section = "{section_name}"'
    )

    prompt = (
        f"Section: {section_name}\n\n"
        "Find all excerpts and data points in this section that relate to "
        "IP, catalog, and export themes. Themes include but are not "
        "limited to:\n"
        "- Catalog dynamics: current vs catalog age splits, deep catalog "
        "share, era-based growth (e.g., music from specific decades or "
        "release windows)\n"
        "- Genre-level catalog vs current breakdowns\n"
        "- Export power and country export rankings\n"
        "- Local-vs-foreign streaming shares within markets\n"
        "- Cross-border genre flows and regional over- or under-indexing\n"
        "- Catalog-driven streaming volume tied to artists, films, or "
        "documentaries\n"
        "- Territory-level chart performance and share by format\n\n"
        "Return up to 9 short excerpts. Each item on its own line "
        "(bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
        "If this section contains no IP-relevant content, reply exactly: "
        "'No IP-relevant content in this section.'\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name=section_name,
        label="IP",
    )


def retrieve_dtc_evidence_by_category(
    file_search_store_name: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)
    sf = _escape_metadata_filter_value(source_filename)
    metadata_filter = (
        f'source_filename = "{sf}" AND category = "{CATEGORY_DTC}"'
    )

    prompt = (
        "All indexed chunks for this upload are labeled category DTC / fan "
        "engagement (subsection titles may differ).\n\n"
        "Find all excerpts and data points that relate to "
        "Direct-to-Consumer (DTC) fan and engagement themes. Themes include "
        "but are not limited to:\n"
        "- Superfans and fan tiers / spending behavior\n"
        "- Gaming and in-game music integrations (Roblox, Fortnite, etc.)\n"
        "- Podcasts and audio formats beyond music\n"
        "- Short-form video (TikTok, Reels, Shorts) discovery and behavior\n"
        "- Social and community platforms (Discord, Twitch, WhatsApp, "
        "Reddit) used for fan engagement\n"
        "- Live, ticketing, merch, and direct artist-to-fan monetization\n"
        "- Platform-specific listening and engagement patterns\n\n"
        "Return up to 9 short excerpts. Each item on its own line "
        "(bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
        "If this material contains no DTC-relevant content, reply exactly: "
        "'No DTC-relevant content in this section.'\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K_CATEGORY,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name="DTC",
        label="DTC",
    )


def retrieve_ip_evidence_by_category(
    file_search_store_name: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)
    sf = _escape_metadata_filter_value(source_filename)
    metadata_filter = (
        f'source_filename = "{sf}" AND category = "{CATEGORY_IP}"'
    )

    prompt = (
        "All indexed chunks for this upload are labeled category IP / catalog "
        "(subsection titles may differ).\n\n"
        "Find all excerpts and data points that relate to "
        "IP, catalog, and export themes. Themes include but are not "
        "limited to:\n"
        "- Catalog dynamics: current vs catalog age splits, deep catalog "
        "share, era-based growth (e.g., music from specific decades or "
        "release windows)\n"
        "- Genre-level catalog vs current breakdowns\n"
        "- Export power and country export rankings\n"
        "- Local-vs-foreign streaming shares within markets\n"
        "- Cross-border genre flows and regional over- or under-indexing\n"
        "- Catalog-driven streaming volume tied to artists, films, or "
        "documentaries\n"
        "- Territory-level chart performance and share by format\n\n"
        "Return up to 9 short excerpts. Each item on its own line "
        "(bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
        "If this material contains no IP-relevant content, reply exactly: "
        "'No IP-relevant content in this section.'\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K_CATEGORY,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name="IP",
        label="IP",
    )


def retrieve_other_section_evidence(
    file_search_store_name: str,
    subsection_key: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_GEMINI_MODEL
    )
    client = genai.Client(api_key=api_key)
    sf = _escape_metadata_filter_value(source_filename)
    sk = _escape_metadata_filter_value(subsection_key)
    metadata_filter = (
        f'source_filename = "{sf}" AND category = "{CATEGORY_OTHER}" '
        f'AND subsection_key = "{sk}"'
    )

    prompt = (
        "Return up to 9 short excerpts or data points from this subsection.\n"
        "Bucket the output to ensure coverage:\n"
        "- EARLY: 2-3 items from the start.\n"
        "- MIDDLE: 2-3 items from the middle.\n"
        "- LATE: 2-3 items from the end.\n"
        "Each item on its own line (bulleted is fine).\n"
        "Be exact with text inside tables and charts; copy verbatim.\n"
        "If a table or chart item is ambiguous, skip it rather than guess.\n"
        "Prefer direct quotes; close paraphrases acceptable.\n"
        "Do not add new facts.\n"
    )

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=DEFAULT_FILE_SEARCH_TOP_K,
    )
    return _retrieve_with_no_evidence_retry(
        client,
        model=resolved_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[types.Tool(file_search=file_search)],
        ),
        section_name=subsection_key,
        label="generic",
    )
