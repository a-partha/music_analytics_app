from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fallback_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALLBACK_MODEL", "test-fallback-model")
