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

