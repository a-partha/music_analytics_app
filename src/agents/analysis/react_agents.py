from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain.agents import create_agent

from src.agents.analysis.react_tools import (
    fetch_and_summarize_subsection,
    finish,
    judge_section_dtc,
    judge_section_dtc_ip,
    judge_section_ip,
    list_pending_subsections,
)
from src.config.analysis_mode import AnalysisMode
from src.services.langchain_llm import get_langchain_gemini_model


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        return "\n".join(chunks).strip()
    return str(content or "").strip()


def _toolset_for_mode(mode: AnalysisMode) -> list[Any]:
    common = [list_pending_subsections, fetch_and_summarize_subsection]
    if mode == AnalysisMode.DTC_ONLY:
        return [*common, judge_section_dtc, finish]
    if mode == AnalysisMode.IP_ONLY:
        return [*common, judge_section_ip, finish]
    return [*common, judge_section_dtc_ip, finish]


def _prompt_for_mode(mode: AnalysisMode) -> str:
    if mode == AnalysisMode.DTC_ONLY:
        label_rule = (
            "Mode target: collect exactly one ACCEPTED DTC hit.\n"
            "Use judge_section_dtc, which can only emit DTC or OTHER."
        )
    elif mode == AnalysisMode.IP_ONLY:
        label_rule = (
            "Mode target: collect exactly one ACCEPTED IP hit.\n"
            "Use judge_section_ip, which can only emit IP or OTHER."
        )
    else:
        label_rule = (
            "Mode target: collect one ACCEPTED DTC and one ACCEPTED IP hit.\n"
            "Use judge_section_dtc_ip, which can emit DTC, IP, or OTHER."
        )
    return (
        "You are an analysis ReAct agent for one uploaded report.\n"
        "Goal: keep scanning subsections until mode target categories are "
        "satisfied.\n"
        f"{label_rule}\n"
        "Always use tools. Never invent subsection_key values.\n"
        "Workflow:\n"
        "1) Call list_pending_subsections.\n"
        "2) For each pending subsection_key, call "
        "fetch_and_summarize_subsection then the judge tool.\n"
        "3) The judge tool already includes a validator pass. Use its "
        "`accepted_for_target` field to understand whether this subsection "
        "counts toward the mode target.\n"
        "4) Call finish after each judged subsection. If finish says missing "
        "targets remain, continue with next pending subsection.\n"
    )


def _build_agent(
    *,
    mode: AnalysisMode,
    model_name: str | None,
):
    model = get_langchain_gemini_model(model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=_toolset_for_mode(mode),
        system_prompt=_prompt_for_mode(mode),
        name=f"analysis_react_{mode.value}",
    )


def trace_from_messages(
    messages: list[BaseMessage],
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    step = 0
    for message in messages:
        if isinstance(message, AIMessage):
            text = _content_to_text(message.content)
            if text:
                step += 1
                trace.append(
                    {
                        "step": step,
                        "kind": "thought",
                        "text": text,
                    }
                )
            for tool_call in list(message.tool_calls or []):
                name = str(tool_call.get("name") or "")
                args = tool_call.get("args")
                trace.append(
                    {
                        "step": max(step, 1),
                        "kind": "tool",
                        "tool": name,
                        "args": args,
                    }
                )
            continue

        if isinstance(message, ToolMessage):
            trace.append(
                {
                    "step": max(step, 1),
                    "kind": "observation",
                    "tool": str(message.name or ""),
                    "text": _content_to_text(message.content)[:1400],
                }
            )
    return trace


def run_react_agent(
    *,
    mode: AnalysisMode,
    model_name: str | None,
    recursion_limit: int,
    on_trace_update: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[BaseMessage], list[dict[str, Any]]]:
    agent = _build_agent(mode=mode, model_name=model_name)
    payload = {
        "messages": [
            (
                "user",
                "Process every pending subsection and complete labeling.",
            )
        ]
    }
    config = {"recursion_limit": recursion_limit}

    def _emit(trace_rows: list[dict[str, Any]]) -> None:
        if on_trace_update is None:
            return
        try:
            on_trace_update(list(trace_rows))
        except Exception:
            # UI callbacks should not break agent execution.
            return

    messages: list[BaseMessage] = []
    last_len = -1
    for snapshot in agent.stream(payload, config=config, stream_mode="values"):
        if not isinstance(snapshot, dict):
            continue
        messages = list(snapshot.get("messages") or [])
        trace = trace_from_messages(messages)
        if len(trace) != last_len:
            _emit(trace)
            last_len = len(trace)

    if not messages:
        result = agent.invoke(payload, config=config)
        messages = list(result.get("messages") or [])
        _emit(trace_from_messages(messages))

    return messages, trace_from_messages(messages)

