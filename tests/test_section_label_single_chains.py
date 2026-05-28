from __future__ import annotations

import pytest

from src.chains.section_label_single_chains import (
    parse_label_validator_json,
    parse_single_label_json,
)
from src.services.section_categories import CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER


def test_parse_single_label_json_plain_object() -> None:
    raw = '{"label": "DTC"}'
    assert parse_single_label_json(
        raw, allowed=frozenset({CATEGORY_DTC, CATEGORY_OTHER})
    ) == CATEGORY_DTC


def test_parse_single_label_json_fenced_object() -> None:
    raw = '```json\n{"label": "IP"}\n```'
    assert parse_single_label_json(
        raw,
        allowed=frozenset({CATEGORY_DTC, CATEGORY_IP, CATEGORY_OTHER}),
    ) == CATEGORY_IP


def test_parse_single_label_json_rejects_outside_allowed() -> None:
    with pytest.raises(ValueError):
        parse_single_label_json(
            '{"label": "IP"}',
            allowed=frozenset({CATEGORY_DTC, CATEGORY_OTHER}),
        )


def test_parse_single_label_json_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        parse_single_label_json(
            '{"label": "MAYBE"}',
            allowed=frozenset({CATEGORY_DTC, CATEGORY_OTHER}),
        )


def test_parse_label_validator_json_plain() -> None:
    parsed = parse_label_validator_json(
        '{"decision":"ACCEPT","reason":"Looks correct."}'
    )
    assert parsed["decision"] == "ACCEPT"
    assert parsed["reason"] == "Looks correct."


def test_parse_label_validator_json_rejects_bad_decision() -> None:
    with pytest.raises(ValueError):
        parse_label_validator_json('{"decision":"MAYBE"}')

