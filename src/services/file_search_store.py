from __future__ import annotations

import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from google import genai

from src.services.section_cache import (
    SectionCacheEntry,
    get_cached_entry,
    hash_pdf_bytes,
    put_cache_entry,
)
from src.services.section_categories import (
    CATEGORY_DTC,
    CATEGORY_IP,
    CATEGORY_OTHER,
)
from src.services.section_splitter import split_pdf_into_dynamic_slices
from src.services.section_splitter_legacy import (
    split_pdf_into_sections as split_pdf_into_sections_legacy,
)

DEFAULT_STORE_DISPLAY_NAME = "luminate_store"
POLL_INTERVAL_SECONDS = 5
DEFAULT_TIMEOUT_SECONDS = 300
SECTION_SPLITTER_MODE_ENV = "SECTION_SPLITTER_MODE"
CACHE_VERSION_TAG = "sec_v4"

# Must match analysis_pipeline.DTC_SECTION_NAMES / IP_SECTION_NAMES for legacy uploads.
_LEGACY_DTC_SECTION_NAMES = ("Engagement Horizon",)
_LEGACY_IP_SECTION_NAMES = ("Import / Export", "Streaming Atlas")


def _splitter_mode() -> str:
    raw = os.getenv(SECTION_SPLITTER_MODE_ENV, "dynamic").strip().lower()
    return raw if raw in ("legacy", "dynamic") else "dynamic"


def _cache_key(pdf_hash: str) -> str:
    return f"{pdf_hash}:{_splitter_mode()}:{CACHE_VERSION_TAG}"


def _slugify_section(name: str) -> str:
    return (
        name.lower()
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _legacy_manifest(section_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in section_names:
        if name in _LEGACY_DTC_SECTION_NAMES:
            cat = CATEGORY_DTC
        elif name in _LEGACY_IP_SECTION_NAMES:
            cat = CATEGORY_IP
        else:
            cat = CATEGORY_OTHER
        rows.append(
            {
                "subsection_key": _slugify_section(name),
                "display_title": name,
                "category": cat,
            }
        )
    return rows


def _load_env() -> None:
    try:
        load_dotenv()
    except Exception:
        pass


def _get_client() -> genai.Client:
    _load_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _get_store_by_display_name(
    client: genai.Client, display_name: str
) -> object | None:
    for store in client.file_search_stores.list():
        if store.display_name == display_name:
            return store
    return None


def _get_or_create_store(
    client: genai.Client, display_name: str
) -> object:
    existing = _get_store_by_display_name(client, display_name)
    if existing:
        return existing
    return client.file_search_stores.create(config={"display_name": display_name})


def _get_document_by_display_name(
    client: genai.Client, store_name: str, display_name: str
) -> object | None:
    for document in client.file_search_stores.documents.list(parent=store_name):
        if getattr(document, "display_name", None) == display_name:
            return document
    return None


def _wait_for_operation(
    client: genai.Client, operation: object, timeout_seconds: int
) -> object:
    start = time.time()
    while not operation.done:
        if time.time() - start > timeout_seconds:
            raise TimeoutError("Timed out waiting for file upload to finish.")
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = client.operations.get(operation)

    if getattr(operation, "error", None):
        raise RuntimeError(f"Upload failed: {operation.error}")
    return operation


def _wait_for_store_ready(
    client: genai.Client, store_name: str, timeout_seconds: int
) -> None:
    start = time.time()
    while True:
        store = client.file_search_stores.get(name=store_name)
        pending = int(getattr(store, "pending_documents_count", "0") or 0)
        active = int(getattr(store, "active_documents_count", "0") or 0)
        if pending == 0 and active > 0:
            return
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                "File Search indexing is still pending. Try again in a minute."
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def ensure_document_in_store_from_path(
    file_path: str | Path,
    display_name: str | None = None,
    store_display_name: str = DEFAULT_STORE_DISPLAY_NAME,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, str, bool]:
    client = _get_client()
    store = _get_or_create_store(client, store_display_name)

    path = Path(file_path)
    final_display_name = display_name or path.name

    existing = _get_document_by_display_name(client, store.name, final_display_name)
    if existing:
        _wait_for_store_ready(client, store.name, timeout_seconds)
        return store.name, existing.name, True

    operation = client.file_search_stores.upload_to_file_search_store(
        file=str(path),
        file_search_store_name=store.name,
        config={
            "display_name": final_display_name,
            "custom_metadata": [
                {
                    "key": "source_filename",
                    "string_value": final_display_name,
                }
            ],
        },
    )
    _wait_for_operation(client, operation, timeout_seconds)

    uploaded = _get_document_by_display_name(client, store.name, final_display_name)
    if not uploaded:
        raise RuntimeError("Upload finished but document was not found in store.")

    _wait_for_store_ready(client, store.name, timeout_seconds)
    return store.name, uploaded.name, False


def ensure_document_in_store_from_bytes(
    file_bytes: bytes,
    display_name: str,
    store_display_name: str = DEFAULT_STORE_DISPLAY_NAME,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, str, bool]:
    suffix = Path(display_name).suffix or ".pdf"
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        return ensure_document_in_store_from_path(
            temp_path,
            display_name=display_name,
            store_display_name=store_display_name,
            timeout_seconds=timeout_seconds,
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _truncate_metadata(s: str, max_len: int = 240) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def ensure_sections_in_store_from_bytes(
    file_bytes: bytes,
    display_name: str,
    store_display_name: str = DEFAULT_STORE_DISPLAY_NAME,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    model_name: str | None = None,
) -> tuple[str, dict[str, str], bool, list[dict[str, str]], str]:
    """
    Upload per-section PDFs and cache the mapping.

    Returns:
        store_name, section_doc_names (subsection slug key -> Gemini doc name),
        reused flag, manifest rows, splitter_mode (legacy|dynamic).
    """
    pdf_hash = hash_pdf_bytes(file_bytes)
    cache_key = _cache_key(pdf_hash)
    mode = _splitter_mode()
    cached = get_cached_entry(cache_key)
    if cached and cached.get("splitter_mode", mode) == mode:
        manifest = cached.get("manifest") or []
        return (
            cached["store_name"],
            dict(cached["section_doc_names"]),
            True,
            manifest,
            mode,
        )

    client = _get_client()
    store = _get_or_create_store(client, store_display_name)
    base_stem = Path(display_name).stem
    section_doc_names: dict[str, str] = {}
    manifest: list[dict[str, str]] = []

    if mode == "legacy":
        section_pdfs = split_pdf_into_sections_legacy(
            file_bytes, model_name=model_name
        )
        manifest = _legacy_manifest(list(section_pdfs.keys()))
        for section_name, section_bytes in section_pdfs.items():
            section_slug = _slugify_section(section_name)
            section_display = f"{base_stem}__{section_slug}.pdf"

            existing = _get_document_by_display_name(
                client, store.name, section_display
            )
            if existing:
                section_doc_names[section_slug] = existing.name
                continue

            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(section_bytes)
                tmp_path = tmp.name
            try:
                operation = client.file_search_stores.upload_to_file_search_store(
                    file=tmp_path,
                    file_search_store_name=store.name,
                    config={
                        "display_name": section_display,
                        "custom_metadata": [
                            {
                                "key": "source_filename",
                                "string_value": display_name,
                            },
                            {
                                "key": "section",
                                "string_value": section_name,
                            },
                        ],
                    },
                )
                _wait_for_operation(client, operation, timeout_seconds)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            uploaded = _get_document_by_display_name(
                client, store.name, section_display
            )
            if not uploaded:
                raise RuntimeError(
                    "Section upload finished but document not found: "
                    f"{section_display}"
                )
            section_doc_names[section_slug] = uploaded.name
    else:
        slices = split_pdf_into_dynamic_slices(
            file_bytes, model_name=model_name
        )
        for sl in slices:
            manifest.append(
                {
                    "subsection_key": sl.subsection_key,
                    "display_title": sl.display_title,
                }
            )
            section_display = f"{base_stem}__{sl.subsection_key}.pdf"
            existing = _get_document_by_display_name(
                client, store.name, section_display
            )
            if existing:
                section_doc_names[sl.subsection_key] = existing.name
                continue

            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(sl.pdf_bytes)
                tmp_path = tmp.name
            try:
                operation = client.file_search_stores.upload_to_file_search_store(
                    file=tmp_path,
                    file_search_store_name=store.name,
                    config={
                        "display_name": section_display,
                        "custom_metadata": [
                            {
                                "key": "source_filename",
                                "string_value": display_name,
                            },
                            {
                                "key": "subsection_key",
                                "string_value": sl.subsection_key,
                            },
                            {
                                "key": "subsection_display",
                                "string_value": _truncate_metadata(
                                    sl.display_title
                                ),
                            },
                        ],
                    },
                )
                _wait_for_operation(client, operation, timeout_seconds)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            uploaded = _get_document_by_display_name(
                client, store.name, section_display
            )
            if not uploaded:
                raise RuntimeError(
                    "Section upload finished but document not found: "
                    f"{section_display}"
                )
            section_doc_names[sl.subsection_key] = uploaded.name

    _wait_for_store_ready(client, store.name, timeout_seconds)

    cache_entry: SectionCacheEntry = {
        "store_name": store.name,
        "section_doc_names": section_doc_names,
        "manifest": manifest,
        "splitter_mode": mode,
    }
    put_cache_entry(cache_key, cache_entry)
    return store.name, section_doc_names, False, manifest, mode
