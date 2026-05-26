"""Disk cache for neutral subsection evidence + synthesis summaries (always on)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv

from src.services.file_search_retrieval import DEFAULT_GEMINI_MODEL
from src.services.langchain_llm import DEFAULT_GEMINI_ANALYSIS_MODEL
from src.services.section_cache import CACHE_DIR_DEFAULT

SCHEMA_TOKEN = "neutral_cache_v1"
NEUTRAL_CACHE_FILENAME = "neutral_summary_cache.json"

_file_lock = threading.Lock()


class NeutralCacheRow(TypedDict):
    evidence: str
    summary: str


def _neutral_cache_path() -> Path:
    custom = os.getenv("SECTION_CACHE_PATH")
    if custom:
        return Path(custom).parent / NEUTRAL_CACHE_FILENAME
    return CACHE_DIR_DEFAULT / NEUTRAL_CACHE_FILENAME


def resolved_gemini_model(explicit: str | None) -> str:
    load_dotenv()
    return explicit or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def resolved_synthesis_model(explicit: str | None) -> str:
    load_dotenv()
    return explicit or os.getenv("GEMINI_ANALYSIS_MODEL") or os.getenv("GEMINI_SYNTHESIS_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_ANALYSIS_MODEL


def make_neutral_cache_key(
    pdf_sha256: str,
    subsection_key: str,
    gemini_model_id: str,
    synthesis_model_id: str,
) -> str:
    payload = "|".join(
        (
            SCHEMA_TOKEN,
            pdf_sha256,
            subsection_key,
            gemini_model_id,
            synthesis_model_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_store() -> dict[str, Any]:
    path = _neutral_cache_path()
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


def _save_store(store: dict[str, Any]) -> None:
    path = _neutral_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    tmp.replace(path)


def get_neutral_row(
    pdf_sha256: str,
    subsection_key: str,
    gemini_model_id: str,
    synthesis_model_id: str,
) -> NeutralCacheRow | None:
    key = make_neutral_cache_key(
        pdf_sha256, subsection_key, gemini_model_id, synthesis_model_id
    )
    with _file_lock:
        store = _load_store()
        row = store.get(key)
    if not isinstance(row, dict):
        return None
    ev = row.get("evidence")
    sm = row.get("summary")
    if not isinstance(ev, str) or not isinstance(sm, str):
        return None
    if not ev.strip():
        return None
    return {"evidence": ev, "summary": sm}


def put_neutral_row(
    pdf_sha256: str,
    subsection_key: str,
    gemini_model_id: str,
    synthesis_model_id: str,
    evidence: str,
    summary: str,
) -> None:
    if not evidence.strip():
        return
    key = make_neutral_cache_key(
        pdf_sha256, subsection_key, gemini_model_id, synthesis_model_id
    )
    with _file_lock:
        store = _load_store()
        store[key] = {"evidence": evidence, "summary": summary}
        _save_store(store)
