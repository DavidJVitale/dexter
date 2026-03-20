from __future__ import annotations

import json

from api.services.calculator_tool import CalculatorTool
from api.services.liquid_tool_runner import LiquidToolRunner
from api.services.tool_runtime import ToolRegistry


def test_liquid_runner_uses_assistant_and_tool_messages_for_second_pass() -> None:
    registry = ToolRegistry([CalculatorTool()])
    captured_messages: list[list[dict[str, str]]] = []
    model_outputs = iter(
        [
            '<|tool_call_start|>[{"name":"calculator","arguments":{"expression":"sqrt(924)"}}]<|tool_call_end|>',
            "The square root of 924 is approximately 30.4.",
        ]
    )
    runner = LiquidToolRunner(registry=registry, max_steps=3)

    outcome = runner.run(
        user_text="What is the square root of 924?",
        system_prompt="You are Dexter.",
        model_generate=lambda messages: captured_messages.append(messages) or next(model_outputs),
    )

    assert outcome.final_response == "The square root of 924 is approximately 30.4."
    assert len(outcome.steps) == 1
    assert captured_messages[1][-2]["role"] == "assistant"
    assert "<|tool_call_start|>" in captured_messages[1][-2]["content"]
    assert captured_messages[1][-1]["role"] == "tool"
    tool_payload = json.loads(captured_messages[1][-1]["content"])
    assert tool_payload["result"] == outcome.steps[0].result.content["result"]


def test_liquid_runner_falls_back_when_second_pass_is_orphan_tool_token() -> None:
    registry = ToolRegistry([CalculatorTool()])
    model_outputs = iter(
        [
            '<|tool_call_start|>[{"name":"calculator","arguments":{"expression":"sqrt(924)"}}]<|tool_call_end|>',
            "<|tool_call_end|>",
        ]
    )
    runner = LiquidToolRunner(registry=registry, max_steps=3)

    outcome = runner.run(
        user_text="What is the square root of 924?",
        system_prompt="You are Dexter.",
        model_generate=lambda _messages: next(model_outputs),
    )

    assert "30.397" in outcome.final_response
    assert outcome.traces[-1]["type"] == "llm_final_response"
    assert outcome.traces[-1]["fallback"] is True
