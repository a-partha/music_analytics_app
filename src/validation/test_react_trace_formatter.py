from __future__ import annotations

from src.agents.analysis.trace_formatter import (
    format_trace_rows_plain_english,
    renumber_trace_steps_for_display,
)


def test_formats_pending_tool_and_observation_lines() -> None:
    trace = [
        {
            "step": 1,
            "kind": "tool",
            "tool": "list_pending_subsections",
            "args": {},
        },
        {
            "step": 1,
            "kind": "observation",
            "tool": "list_pending_subsections",
            "text": (
                '[{"subsection_key":"a","display_title":"A"},'
                '{"subsection_key":"b","display_title":"B"}]'
            ),
        },
    ]
    text = format_trace_rows_plain_english(trace)
    assert "Checked which sections are still pending." in text
    assert "Pending list received (2): a, b." in text


def test_formats_fetch_and_judge_accept_reject_finish_runtime() -> None:
    trace = [
        {
            "step": 2,
            "kind": "tool",
            "tool": "fetch_and_summarize_subsection",
            "args": {"subsection_key": "engage_1"},
        },
        {
            "step": 2,
            "kind": "observation",
            "tool": "fetch_and_summarize_subsection",
            "text": '{"subsection_key":"engage_1","summary":"..."}',
        },
        {
            "step": 3,
            "kind": "observation",
            "tool": "judge_section_dtc",
            "text": (
                '{"label":"DTC","validator_decision":"ACCEPT",'
                '"validator_reason":"","accepted_for_target":true}'
            ),
        },
        {
            "step": 4,
            "kind": "observation",
            "tool": "judge_section_dtc",
            "text": (
                '{"label":"DTC","validator_decision":"REJECT",'
                '"validator_reason":"weak thematic match",'
                '"accepted_for_target":false}'
            ),
        },
        {
            "step": 5,
            "kind": "observation",
            "tool": "finish",
            "text": "Cannot finish yet. Missing targets: DTC.",
        },
        {
            "step": 6,
            "kind": "observation",
            "tool": "runtime",
            "text": "Recursion limit reached.",
        },
    ]
    text = format_trace_rows_plain_english(trace)
    assert "Fetched and summarized section engage_1." in text
    assert "Summary returned and stored." in text
    assert "Initial label DTC accepted by validator; counts toward target." in text
    assert (
        "Initial label DTC rejected by validator "
        "(reason: weak thematic match)." in text
    )
    assert "Finish check: Cannot finish yet. Missing targets: DTC." in text
    assert "Runtime note: Recursion limit reached." in text


def test_returns_waiting_message_for_empty_trace() -> None:
    assert format_trace_rows_plain_english([]) == "Waiting for agent activity..."


def test_renumber_trace_steps_fixes_step_one_spam() -> None:
    spam = [
        {"step": 1, "kind": "tool", "tool": "list_pending_subsections", "args": {}},
        {
            "step": 1,
            "kind": "observation",
            "tool": "list_pending_subsections",
            "text": "[]",
        },
        {
            "step": 1,
            "kind": "tool",
            "tool": "fetch_and_summarize_subsection",
            "args": {"subsection_key": "a"},
        },
        {
            "step": 1,
            "kind": "observation",
            "tool": "fetch_and_summarize_subsection",
            "text": "{}",
        },
        {
            "step": 1,
            "kind": "tool",
            "tool": "judge_section_dtc",
            "args": {"subsection_key": "a"},
        },
        {
            "step": 1,
            "kind": "observation",
            "tool": "judge_section_dtc",
            "text": '{"label":"DTC","validator_decision":"ACCEPT"}',
        },
    ]
    renumbered = renumber_trace_steps_for_display(spam)
    assert [row["step"] for row in renumbered if row["kind"] == "tool"] == [1, 2, 3]
    assert [
        row["step"] for row in renumbered if row["kind"] == "observation"
    ] == [1, 2, 3]

    text = format_trace_rows_plain_english(spam)
    assert "Step 1:" in text
    assert "Step 2:" in text
    assert "Step 3:" in text
    assert text.count("Step 1:") == 2  # tool + observation for first turn


def test_renumber_resets_after_cannot_finish_yet() -> None:
    rows = [
        {
            "step": 99,
            "kind": "tool",
            "tool": "fetch_and_summarize_subsection",
            "args": {"subsection_key": "introduction"},
        },
        {
            "step": 99,
            "kind": "observation",
            "tool": "fetch_and_summarize_subsection",
            "text": "{}",
        },
        {
            "step": 99,
            "kind": "tool",
            "tool": "judge_section_dtc",
            "args": {"subsection_key": "introduction"},
        },
        {
            "step": 99,
            "kind": "observation",
            "tool": "judge_section_dtc",
            "text": '{"label":"OTHER","validator_decision":"ACCEPT"}',
        },
        {"step": 99, "kind": "tool", "tool": "finish", "args": {}},
        {
            "step": 99,
            "kind": "observation",
            "tool": "finish",
            "text": "Cannot finish yet. Missing targets: DTC.",
        },
        {
            "step": 99,
            "kind": "tool",
            "tool": "fetch_and_summarize_subsection",
            "args": {"subsection_key": "welcome_to_transmedia"},
        },
        {
            "step": 99,
            "kind": "observation",
            "tool": "fetch_and_summarize_subsection",
            "text": "{}",
        },
        {"step": 99, "kind": "tool", "tool": "finish", "args": {}},
        {
            "step": 99,
            "kind": "observation",
            "tool": "finish",
            "text": "FINISH_OK",
        },
    ]
    renumbered = renumber_trace_steps_for_display(rows)
    steps = [row["step"] for row in renumbered]
    # First section cycle: 1,1,2,2,3,3 then reset; second cycle: 1,1,2,2
    assert steps == [1, 1, 2, 2, 3, 3, 1, 1, 2, 2]

    text = format_trace_rows_plain_english(rows)
    assert "Step 3: Finish check: Cannot finish yet. Missing targets: DTC." in text
    assert "-----\nStep 1: Fetched and summarized section welcome_to_transmedia." in text
    assert "Step 2: Finish check passed." in text
    assert not text.endswith("-----")


def test_renumber_does_not_reset_after_finish_ok() -> None:
    rows = [
        {"step": 1, "kind": "tool", "tool": "finish", "args": {}},
        {
            "step": 1,
            "kind": "observation",
            "tool": "finish",
            "text": "FINISH_OK",
        },
        {
            "step": 1,
            "kind": "tool",
            "tool": "list_pending_subsections",
            "args": {},
        },
    ]
    renumbered = renumber_trace_steps_for_display(rows)
    assert [row["step"] for row in renumbered] == [1, 1, 2]