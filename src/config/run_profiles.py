from __future__ import annotations

import os
from enum import Enum


class RunProfile(str, Enum):
    FULL = "full"
    DEV_ONE_PER_CATEGORY = "dev_one_per_category"


_DEV_ENV_VALUES = frozenset({"dev_one_per_category", "dev", "1", "true", "yes", "on"})


def _coerce_profile(value: RunProfile | str | None) -> RunProfile | None:
    if value is None:
        return None
    if isinstance(value, RunProfile):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in _DEV_ENV_VALUES:
        return RunProfile.DEV_ONE_PER_CATEGORY
    if raw in (RunProfile.FULL.value, "full", "0", "false", "no", "off"):
        return RunProfile.FULL
    return RunProfile.FULL


def resolve_run_profile(
    explicit: RunProfile | str | None = None,
    *,
    dev_mode_flag: bool = False,
    use_env: bool = True,
) -> RunProfile:
    """
    Resolve analysis run profile.

    Priority: explicit profile > dev_mode_flag > env (if use_env).

    Streamlit should pass explicit RunProfile.FULL or DEV so the checkbox
    is authoritative and env vars do not override an unchecked box.
    """
    coerced = _coerce_profile(explicit)
    if coerced is not None:
        return coerced

    if dev_mode_flag:
        return RunProfile.DEV_ONE_PER_CATEGORY

    if not use_env:
        return RunProfile.FULL

    env = (
        os.getenv("PIPELINE_RUN_PROFILE", "")
        or os.getenv("ANALYSIS_DEV_MODE", "")
    ).strip().lower()
    if env in _DEV_ENV_VALUES:
        return RunProfile.DEV_ONE_PER_CATEGORY
    return RunProfile.FULL


def profile_from_state_value(raw: str | None) -> RunProfile:
    """Read profile already stored on graph state (no env lookup)."""
    return _coerce_profile(raw) or RunProfile.FULL
