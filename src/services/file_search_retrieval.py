from __future__ import annotations

import os
import re
import sys
import time
from typing import NamedTuple

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.config.models import resolve_model

DEFAULT_FILE_SEARCH_TOP_K = 12

MAX_RETRIES_ON_RATE_LIMIT = 3
DEFAULT_RETRY_DELAY_S = 6.0
RETRY_DELAY_BUFFER_S = 1.0

NO_EVIDENCE_MAX_ATTEMPTS = 2
NO_EVIDENCE_SHORT_THRESHOLD = 200
NO_EVIDENCE_ESCAPE_SENTENCES = (
    "no dtc-relevant content in this section.",
    "no ip-relevant content in this section.",
)


QUERY_CHUNKS_TOP_K = 12
QUERY_CHUNKS_MAX_OUTPUT_TOKENS = 1024
QUERY_CHUNKS_MAX_ATTEMPTS = 4
QUERY_CHUNKS_BACKOFF_S = 2.0


class RetrievalError(RuntimeError):
    pass


class RetrievedChunk(NamedTuple):
    text: str


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


def _is_transient_unavailable(exc: Exception) -> bool:
    if getattr(exc, "code", None) == 503:
        return True
    text = str(exc).lower()
    return "unavailable" in text or "deadline expired" in text


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
) -> tuple[object, float]:
    """Return the response and the successful call's latency in ms.

    Backoff sleep is not included. A 429 retries; any other error raises.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT):
        try:
            t_call = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            latency_ms = round((time.perf_counter() - t_call) * 1000, 2)
            return response, latency_ms
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
) -> tuple[str, object, float]:
    """Return excerpt text, the response, and that call's latency in ms.

    A short or empty reply is retried once. Latency and the response are
    from the call whose text is returned. Backoff sleep is not included.
    """
    last_text = ""
    for attempt in range(NO_EVIDENCE_MAX_ATTEMPTS):
        response, latency_ms = _generate_with_backoff(
            client,
            model=model,
            contents=contents,
            config=config,
        )
        raw_text = (response.text or "").strip()
        looks_like_empty = _looks_like_no_evidence(raw_text)
        if not looks_like_empty:
            return raw_text, response, latency_ms
        last_text = raw_text
    raise RetrievalError(
        f"No {label} evidence retrieved for section '{section_name}' after "
        f"{NO_EVIDENCE_MAX_ATTEMPTS} attempts. Last response: "
        f"{last_text[:200]}"
    )


def retrieve_subsection_evidence(
    file_search_store_name: str,
    subsection_key: str,
    display_title: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> str:
    """Neutral EARLY/MIDDLE/LATE excerpts for a dynamic subsection upload."""
    text, _, _ = retrieve_subsection_evidence_with_usage(
        file_search_store_name=file_search_store_name,
        subsection_key=subsection_key,
        display_title=display_title,
        source_filename=source_filename,
        model_name=model_name,
    )
    return text


def retrieve_subsection_evidence_with_usage(
    file_search_store_name: str,
    subsection_key: str,
    display_title: str,
    source_filename: str | None = None,
    model_name: str | None = None,
) -> tuple[str, object | None, float]:
    """Same retrieval as retrieve_subsection_evidence, plus usage and latency.

    Returns the excerpt string, that call's usage_metadata, and latency in
    ms. Latency is the successful generate_content call only. Backoff sleep
    is not included. The filter is always source_filename AND subsection_key.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    if not source_filename:
        raise RetrievalError(
            "Missing source filename for metadata-scoped retrieval. "
            "Upload/reselect the PDF and try again."
        )

    resolved_model = resolve_model(None, model_name)
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
    text, response, latency_ms = _retrieve_with_no_evidence_retry(
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
    return text, getattr(response, "usage_metadata", None), latency_ms


def retrieve_query_chunks(
    file_search_store_name: str,
    query: str,
    subsection_key: str | None = None,
    model_name: str | None = None,
    source_filename: str | None = None,
) -> list[RetrievedChunk]:
    """Raw File Search chunks for one question; see the _with_usage variant."""
    chunks, _, _ = retrieve_query_chunks_with_usage(
        file_search_store_name=file_search_store_name,
        query=query,
        subsection_key=subsection_key,
        model_name=model_name,
        source_filename=source_filename,
    )
    return chunks


def retrieve_query_chunks_with_usage(
    file_search_store_name: str,
    query: str,
    subsection_key: str | None = None,
    model_name: str | None = None,
    source_filename: str | None = None,
) -> tuple[list[RetrievedChunk], object | None, float]:
    """Raw File Search chunks, usage, and the successful call's latency in ms.

    A 503 or deadline timeout waits 2s, 4s, then 8s and retries. A 429 raises
    immediately. Returns [] when a call succeeds but no chunk text comes back.
    Usage and latency come only from the successful attempt. Backoff sleep is
    not included.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    resolved_model = resolve_model(None, model_name)
    parts: list[str] = []
    if source_filename:
        sf = _escape_metadata_filter_value(source_filename)
        parts.append(f'source_filename = "{sf}"')
    if subsection_key:
        sk = _escape_metadata_filter_value(subsection_key)
        parts.append(f'subsection_key = "{sk}"')
    metadata_filter = " AND ".join(parts) or None

    file_search = types.FileSearch(
        file_search_store_names=[file_search_store_name],
        metadata_filter=metadata_filter,
        top_k=QUERY_CHUNKS_TOP_K,
    )
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        # Only the grounding chunks are used; this caps the discarded answer.
        max_output_tokens=QUERY_CHUNKS_MAX_OUTPUT_TOKENS,
        tools=[types.Tool(file_search=file_search)],
    )
    last_exc: Exception | None = None
    retrieval_ms = 0.0
    for attempt in range(QUERY_CHUNKS_MAX_ATTEMPTS):
        try:
            t_call = time.perf_counter()
            response = client.models.generate_content(
                model=resolved_model,
                contents=query,
                config=config,
            )
            retrieval_ms = round((time.perf_counter() - t_call) * 1000, 2)
            break
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay_seconds(str(exc))
                raise RetrievalError(
                    f"Gemini rate limit (429); retry in {delay:.0f}s. {exc}"
                ) from exc
            if (
                not _is_transient_unavailable(exc)
                or attempt + 1 >= QUERY_CHUNKS_MAX_ATTEMPTS
            ):
                status = getattr(exc, "code", None) or type(exc).__name__
                raise RetrievalError(
                    f"File Search call failed ({status}): {exc}"
                ) from exc
            delay = QUERY_CHUNKS_BACKOFF_S * (2 ** attempt)
            print(
                f"File Search unavailable, waiting {delay:.0f}s "
                f"(attempt {attempt + 1} of {QUERY_CHUNKS_MAX_ATTEMPTS})",
                file=sys.stderr,
            )
            time.sleep(delay)
    else:
        raise RetrievalError(f"File Search call failed: {last_exc}")

    chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    for candidate in response.candidates or []:
        metadata = candidate.grounding_metadata
        for grounding_chunk in (metadata.grounding_chunks if metadata else None) or []:
            context = grounding_chunk.retrieved_context
            text = (context.text or "").strip() if context else ""
            if text and text not in seen:
                seen.add(text)
                chunks.append(RetrievedChunk(text=text))
    return chunks, response.usage_metadata, retrieval_ms
