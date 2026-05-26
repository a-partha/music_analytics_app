from __future__ import annotations

import json
import os
import re
from typing import NamedTuple

import fitz
from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_VISION_MODEL = "gemini-2.5-flash"
DEFAULT_RENDER_DPI = 110

SECTION_NAMES = (
    "Midyear Metrics",
    "Streaming Atlas",
    "Import / Export",
    "Engagement Horizon",
    "Artist Spectrum",
    "Future in Focus",
    "Midyear Charts",
)


class SectionDetectionError(RuntimeError):
    pass


class DynamicSectionSlice(NamedTuple):
    subsection_key: str
    display_title: str
    pdf_bytes: bytes


def _render_pages_to_png(
    pdf_bytes: bytes, dpi: int = DEFAULT_RENDER_DPI
) -> list[bytes]:
    pages: list[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def _slugify_title(name: str) -> str:
    base = (
        name.lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )
    return re.sub(r"[^a-z0-9_]+", "", base).strip("_")[:80] or "section"


def _unique_subsection_keys(display_titles: list[str]) -> list[str]:
    used: dict[str, int] = {}
    keys: list[str] = []
    for title in display_titles:
        stem = _slugify_title(title)
        n = used.get(stem, 0) + 1
        used[stem] = n
        keys.append(stem if n == 1 else f"{stem}_{n}")
    return keys


def _build_discovery_vision_prompt() -> str:
    return (
        "You are scanning a music industry or market analytics PDF report "
        "page by page.\n"
        "The pages are provided in order, starting at index 0.\n\n"
        "Task: list every major BODY section of the report. For each "
        "section, give the section title EXACTLY as shown on the page "
        "(verbatim spelling) and the 0-indexed page number where that "
        "section BEGINS.\n\n"
        "Rules:\n"
        "- A section begins on the first page where its title appears as a "
        "prominent heading (including ALL CAPS or minor punctuation variants).\n"
        "- Ignore tables of contents, running headers, navigation strips, "
        "or footers unless that is the only place the title appears.\n"
        "- Include all substantive chapters/sections you see; do not invent "
        "names.\n"
        "- Do not duplicate the same start_page.\n\n"
        "Return STRICT JSON only:\n"
        '{"sections": [{"display_title": "<string>", "start_page": '
        "<integer>}, ...]}\n"
        "Include at least one section. Order by start_page ascending.\n"
        "Do not include any text outside the JSON.\n"
    )


def _parse_discovery_response(raw_text: str) -> list[tuple[str, int]]:
    if not raw_text:
        raise SectionDetectionError("Vision model returned empty response.")
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SectionDetectionError(
            f"Vision response was not valid JSON: {exc}\n"
            f"Raw: {raw_text[:300]}"
        )
    entries = payload.get("sections")
    if not isinstance(entries, list):
        raise SectionDetectionError(
            "Vision response missing 'sections' list."
        )
    seen_pages: set[int] = set()
    ordered: list[tuple[str, int]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        title = item.get("display_title")
        start = item.get("start_page")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(start, int):
            continue
        title_clean = title.strip()
        if start in seen_pages:
            continue
        seen_pages.add(start)
        ordered.append((title_clean, start))
    ordered.sort(key=lambda x: x[1])
    if not ordered:
        raise SectionDetectionError(
            "Vision did not return any usable sections with titles and "
            "start pages."
        )
    return ordered


def _detect_discovered_sections(
    pdf_bytes: bytes,
    model_name: str | None,
) -> list[tuple[str, int]]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    page_pngs = _render_pages_to_png(pdf_bytes)
    if not page_pngs:
        raise SectionDetectionError("PDF has no pages.")

    client = genai.Client(api_key=api_key)
    resolved_model = model_name or os.getenv(
        "GEMINI_MODEL", DEFAULT_VISION_MODEL
    )

    parts: list[object] = [_build_discovery_vision_prompt()]
    for page_idx, png in enumerate(page_pngs):
        parts.append(f"Page {page_idx}:")
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    response = client.models.generate_content(
        model=resolved_model,
        contents=parts,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _parse_discovery_response(response.text or "")


def _page_ranges_for_starts(
    ordered: list[tuple[str, int]],
    total_pages: int,
) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    for idx, (title, start) in enumerate(ordered):
        if idx + 1 < len(ordered):
            end = ordered[idx + 1][1] - 1
        else:
            end = total_pages - 1
        if start < 0 or end >= total_pages or end < start:
            raise SectionDetectionError(
                f"Invalid range for '{title}': start={start}, end={end}, "
                f"total_pages={total_pages}."
            )
        ranges.append((title, start, end))
    return ranges


def _slice_pdf(pdf_bytes: bytes, start: int, end: int) -> bytes:
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out = fitz.open()
        try:
            out.insert_pdf(src, from_page=start, to_page=end)
            return out.tobytes()
        finally:
            out.close()
    finally:
        src.close()


def split_pdf_into_dynamic_slices(
    pdf_bytes: bytes,
    model_name: str | None = None,
) -> list[DynamicSectionSlice]:
    discovered = _detect_discovered_sections(pdf_bytes, model_name)

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = src.page_count
    finally:
        src.close()

    triples = _page_ranges_for_starts(discovered, total_pages)
    titles = [t for t, _, _ in triples]
    keys = _unique_subsection_keys(titles)

    slices: list[DynamicSectionSlice] = []
    for (title, start, end), subsection_key in zip(triples, keys):
        blob = _slice_pdf(pdf_bytes, start, end)
        slices.append(
            DynamicSectionSlice(
                subsection_key=subsection_key,
                display_title=title,
                pdf_bytes=blob,
            )
        )
    return slices
