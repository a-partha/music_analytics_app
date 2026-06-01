"""Run profile resolution (no network)."""

from __future__ import annotations

import pytest

from src.config.run_profiles import RunProfile, resolve_run_profile


def test_explicit_full_ignores_dev_mode_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_RUN_PROFILE", "dev_one_per_category")
    assert (
        resolve_run_profile(RunProfile.FULL, dev_mode_flag=True, use_env=False)
        == RunProfile.FULL
    )


def test_explicit_dev_from_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIPELINE_RUN_PROFILE", raising=False)
    monkeypatch.delenv("ANALYSIS_DEV_MODE", raising=False)
    assert resolve_run_profile("dev_one_per_category", use_env=False) == (
        RunProfile.DEV_ONE_PER_CATEGORY
    )


def test_env_dev_when_no_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYSIS_DEV_MODE", "1")
    assert resolve_run_profile(None, dev_mode_flag=False) == (
        RunProfile.DEV_ONE_PER_CATEGORY
    )


def test_env_ignored_when_use_env_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYSIS_DEV_MODE", "1")
    assert resolve_run_profile(None, dev_mode_flag=False, use_env=False) == (
        RunProfile.FULL
    )
