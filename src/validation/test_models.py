"""Model name resolution from env vars (no network)."""

from __future__ import annotations

import pytest

from src.config import models
from src.config.models import resolve_model


@pytest.fixture(autouse=True)
def _ignore_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "load_dotenv", lambda *args, **kwargs: False)


def test_explicit_then_stage_var_win_over_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANALYSIS_MODEL", "stage-model")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model")
    assert resolve_model("ANALYSIS_MODEL", "explicit-model") == "explicit-model"
    assert resolve_model("ANALYSIS_MODEL") == "stage-model"


def test_fallback_used_when_stage_var_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYSIS_MODEL", "  ")
    monkeypatch.setenv("FALLBACK_MODEL", "fallback-model")
    assert resolve_model("ANALYSIS_MODEL") == "fallback-model"
    assert resolve_model(None) == "fallback-model"


def test_error_names_both_vars_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANALYSIS_MODEL", raising=False)
    monkeypatch.delenv("FALLBACK_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="ANALYSIS_MODEL or FALLBACK_MODEL"):
        resolve_model("ANALYSIS_MODEL")
