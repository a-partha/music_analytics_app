from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from src.agents.analysis import react_agents as ra
from src.config.analysis_mode import AnalysisMode


def test_toolset_is_mode_specific() -> None:
    dtc_names = [tool.name for tool in ra._toolset_for_mode(AnalysisMode.DTC_ONLY)]
    ip_names = [tool.name for tool in ra._toolset_for_mode(AnalysisMode.IP_ONLY)]
    both_names = [tool.name for tool in ra._toolset_for_mode(AnalysisMode.BOTH)]

    assert "judge_section_dtc" in dtc_names
    assert "judge_section_ip" not in dtc_names
    assert "judge_section_ip" in ip_names
    assert "judge_section_dtc" not in ip_names
    assert "judge_section_dtc_ip" in both_names


def test_trace_from_messages_parses_thought_tools_observations() -> None:
    messages = [
        AIMessage(
            content="I should inspect pending sections first.",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "list_pending_subsections",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="[{\"subsection_key\": \"a\"}]",
            name="list_pending_subsections",
            tool_call_id="call-1",
        ),
    ]

    trace = ra.trace_from_messages(messages)
    assert any(item.get("kind") == "thought" for item in trace)
    assert any(
        item.get("kind") == "tool"
        and item.get("tool") == "list_pending_subsections"
        for item in trace
    )
    assert any(item.get("kind") == "observation" for item in trace)


def test_trace_from_messages_advances_step_across_interleaved_turns() -> None:
    """Regression: silent tool-call turns must not collapse to Step 1 spam."""
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "list_pending_subsections",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='[{"subsection_key":"introduction"}]',
            name="list_pending_subsections",
            tool_call_id="call-1",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-2",
                    "name": "fetch_and_summarize_subsection",
                    "args": {"subsection_key": "introduction"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"subsection_key":"introduction","summary":"..."}',
            name="fetch_and_summarize_subsection",
            tool_call_id="call-2",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-3",
                    "name": "judge_section_dtc",
                    "args": {"subsection_key": "introduction"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content='{"label":"OTHER","validator_decision":"ACCEPT"}',
            name="judge_section_dtc",
            tool_call_id="call-3",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-4",
                    "name": "finish",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Cannot finish yet. Missing targets: DTC.",
            name="finish",
            tool_call_id="call-4",
        ),
    ]

    trace = ra.trace_from_messages(messages)
    tool_steps = [
        item["step"]
        for item in trace
        if item.get("kind") == "tool"
    ]
    assert tool_steps == [1, 2, 3, 4]
    assert not all(step == 1 for step in tool_steps)

    observation_steps = [
        item["step"]
        for item in trace
        if item.get("kind") == "observation"
    ]
    assert observation_steps == [1, 2, 3, 4]


def test_trace_from_messages_resets_after_cannot_finish_yet() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "fetch_and_summarize_subsection",
                    "args": {"subsection_key": "introduction"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="{}",
            name="fetch_and_summarize_subsection",
            tool_call_id="call-1",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-2",
                    "name": "finish",
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="Cannot finish yet. Missing targets: DTC.",
            name="finish",
            tool_call_id="call-2",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-3",
                    "name": "fetch_and_summarize_subsection",
                    "args": {"subsection_key": "welcome_to_transmedia"},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="{}",
            name="fetch_and_summarize_subsection",
            tool_call_id="call-3",
        ),
    ]

    trace = ra.trace_from_messages(messages)
    tool_steps = [item["step"] for item in trace if item.get("kind") == "tool"]
    assert tool_steps == [1, 2, 1]


def test_run_react_agent_passes_recursion_limit(monkeypatch) -> None:
    class _FakeAgent:
        def __init__(self) -> None:
            self.config = None

        def stream(self, payload, config=None, stream_mode=None):
            self.config = config
            yield {"messages": [AIMessage(content="thinking...")]}

        def invoke(self, payload, config=None):
            self.config = config
            return {"messages": [AIMessage(content="done")]}

    fake_agent = _FakeAgent()
    monkeypatch.setattr(ra, "_build_agent", lambda **kwargs: fake_agent)

    _, trace = ra.run_react_agent(
        mode=AnalysisMode.BOTH,
        model_name="test-model",
        recursion_limit=11,
    )
    assert fake_agent.config == {"recursion_limit": 11}
    assert any(item.get("kind") == "thought" for item in trace)


def test_run_react_agent_emits_live_trace(monkeypatch) -> None:
    class _FakeAgent:
        def stream(self, payload, config=None, stream_mode=None):
            yield {"messages": [AIMessage(content="step 1")]}
            yield {"messages": [AIMessage(content="step 1"), AIMessage(content="step 2")]}

        def invoke(self, payload, config=None):
            return {"messages": [AIMessage(content="done")]}

    monkeypatch.setattr(ra, "_build_agent", lambda **kwargs: _FakeAgent())
    emitted: list[list[dict[str, object]]] = []
    _, trace = ra.run_react_agent(
        mode=AnalysisMode.DTC_ONLY,
        model_name="test-model",
        recursion_limit=22,
        on_trace_update=lambda rows: emitted.append(rows),
    )
    assert len(emitted) >= 1
    assert len(trace) >= 2

