from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from google import genai

CACHE_DIR_DEFAULT = Path.home() / ".music_analytics_app"
CACHE_FILE_DEFAULT = "section_cache.json"


class SectionCacheEntry(TypedDict, total=False):
    store_name: str
    section_doc_names: dict[str, str]
    manifest: list[dict[str, str]]
    splitter_mode: str


def _cache_path() -> Path:
    custom = os.getenv("SECTION_CACHE_PATH")
    if custom:
        return Path(custom)
    return CACHE_DIR_DEFAULT / CACHE_FILE_DEFAULT


def hash_pdf_bytes(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _load_index() -> dict[str, SectionCacheEntry]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_index(index: dict[str, SectionCacheEntry]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)


def _get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _entry_is_valid_remote(entry: SectionCacheEntry) -> bool:
    try:
        client = _get_client()
        store = client.file_search_stores.get(name=entry["store_name"])
        if not store:
            return False
        existing_doc_names = {
            getattr(doc, "name", None)
            for doc in client.file_search_stores.documents.list(
                parent=entry["store_name"]
            )
        }
        for doc_name in entry["section_doc_names"].values():
            if doc_name not in existing_doc_names:
                return False
        return True
    except Exception:
        return False


def get_cached_entry(pdf_hash: str) -> SectionCacheEntry | None:
    index = _load_index()
    entry = index.get(pdf_hash)
    if entry is None:
        return None
    if not _entry_is_valid_remote(entry):
        index.pop(pdf_hash, None)
        _save_index(index)
        return None
    return entry


def put_cache_entry(pdf_hash: str, entry: SectionCacheEntry) -> None:
    index = _load_index()
    index[pdf_hash] = entry
    _save_index(index)
