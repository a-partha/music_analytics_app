from __future__ import annotations

import json
import os
import re

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


def _build_vision_prompt() -> str:
    section_lines = "\n".join(f"- {name}" for name in SECTION_NAMES)
    return (
        "You are scanning a music industry report (Luminate Midyear) "
        "page by page.\n"
        "The pages are provided in order, starting at index 0.\n\n"
        "Your task: for each of these 7 section headers, identify the "
        "0-indexed page number where that section BEGINS.\n\n"
        f"Section headers to find:\n{section_lines}\n\n"
        "Rules:\n"
        "- A section begins on the first page where its header (or close "
        "visual variant such as ALL CAPS or slash spacing) appears as a "
        "prominent title.\n"
        "- Ignore mentions inside tables of contents, navigation strips, "
        "or footers.\n"
        "- If you cannot confidently locate a section, use null for its "
        "start_page.\n\n"
        "Return STRICT JSON only, with this exact shape:\n"
        '{"sections": [{"section_name": "<one of the names above, '
        'verbatim>", "start_page": <int or null>}, ...]}\n'
        "Include exactly 7 entries, one per section, in the order listed "
        "above.\n"
        "Do not include any text outside the JSON.\n"
    )


def _parse_vision_response(raw_text: str) -> dict[str, int]:
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

    starts: dict[str, int] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = item.get("section_name")
        start = item.get("start_page")
        if name in SECTION_NAMES and isinstance(start, int):
            starts[name] = start

    missing = [name for name in SECTION_NAMES if name not in starts]
    if missing:
        raise SectionDetectionError(
            f"Vision could not locate sections: {missing}"
        )
    return starts


def _detect_section_start_pages(
    pdf_bytes: bytes,
    model_name: str | None,
) -> dict[str, int]:
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

    parts: list[object] = [_build_vision_prompt()]
    for page_idx, png in enumerate(page_pngs):
        parts.append(f"Page {page_idx}:")
        parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))

    response = client.models.generate_content(
        model=resolved_model,
        contents=parts,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return _parse_vision_response(response.text or "")


def _build_page_ranges(
    starts_by_section: dict[str, int],
    total_pages: int,
) -> dict[str, tuple[int, int]]:
    ordered = sorted(starts_by_section.items(), key=lambda kv: kv[1])
    ranges: dict[str, tuple[int, int]] = {}
    for idx, (name, start) in enumerate(ordered):
        if idx + 1 < len(ordered):
            end = ordered[idx + 1][1] - 1
        else:
            end = total_pages - 1
        if start < 0 or end >= total_pages or end < start:
            raise SectionDetectionError(
                f"Invalid range for '{name}': start={start}, end={end}, "
                f"total_pages={total_pages}."
            )
        ranges[name] = (start, end)
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


def split_pdf_into_sections(
    pdf_bytes: bytes,
    model_name: str | None = None,
) -> dict[str, bytes]:
    starts = _detect_section_start_pages(pdf_bytes, model_name)

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total_pages = src.page_count
    finally:
        src.close()

    ranges = _build_page_ranges(starts, total_pages)
    return {
        name: _slice_pdf(pdf_bytes, start, end)
        for name, (start, end) in ranges.items()
    }
