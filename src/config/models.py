from __future__ import annotations

import os

from dotenv import load_dotenv

FALLBACK_MODEL_ENV = "FALLBACK_MODEL"


def _env_model(name: str) -> str | None:
    return (os.getenv(name) or "").strip() or None


def resolve_model(env_var: str | None, explicit: str | None = None) -> str:
    """
    Resolve a model name from the environment. No model names live in code.

    Priority: explicit > env_var > FALLBACK_MODEL. Blank values count as unset.
    """
    if explicit:
        return explicit
    load_dotenv()
    model = (_env_model(env_var) if env_var else None) or _env_model(
        FALLBACK_MODEL_ENV
    )
    if model:
        return model
    names = f"{env_var} or {FALLBACK_MODEL_ENV}" if env_var else FALLBACK_MODEL_ENV
    raise RuntimeError(
        f"No model configured. Set {names} in .env or Streamlit secrets."
    )


def fallback_model_is_set() -> bool:
    load_dotenv()
    return _env_model(FALLBACK_MODEL_ENV) is not None
