from __future__ import annotations

from src.agents.analysis.trace_formatter import format_trace_rows_plain_english


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

