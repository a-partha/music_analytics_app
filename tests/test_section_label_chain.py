"""Tests for section label JSON parsing (no Ollama)."""

from __future__ import annotations

import pytest

from src.chains.section_label_chain import parse_section_label_json
from src.services.section_categories import CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER


def test_parse_plain_json_array() -> None:
    raw = '[{"subsection_key": "foo", "label": "DTC"}, {"subsection_key": "bar", "label": "IP"}]'
    assert parse_section_label_json(raw) == {
        "foo": CATEGORY_DTC,
        "bar": CATEGORY_IP,
    }


def test_parse_fenced_json() -> None:
    raw = 'Here:\n```json\n[{"subsection_key": "x", "label": "OTHER"}]\n```'
    assert parse_section_label_json(raw) == {"x": CATEGORY_OTHER}


def test_parse_wrapped_labels_key() -> None:
    raw = '{"labels": [{"subsection_key": "a", "label": "IP"}]}'
    assert parse_section_label_json(raw) == {"a": CATEGORY_IP}


def test_parse_invalid_json_raises() -> None:
    with pytest.raises(Exception):
        parse_section_label_json("not json")


def test_parse_invalid_label_raises() -> None:
    raw = '[{"subsection_key": "foo", "label": "MAYBE_DTC"}]'
    with pytest.raises(ValueError):
        parse_section_label_json(raw)
