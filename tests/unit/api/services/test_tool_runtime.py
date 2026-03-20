from __future__ import annotations

from api.services.calculator_tool import CalculatorTool
from api.services.tool_runtime import ToolLoopRunner, ToolRegistry


def test_tool_loop_executes_calculator_then_returns_final_response() -> None:
    registry = ToolRegistry([CalculatorTool()])
    prompts: list[str] = []
    model_outputs = iter(
        [
            '{"type":"tool_call","tool_name":"calculator","arguments":{"expression":"((12 + 8) * (7 - 3)) / 5"}}',
            '{"type":"final_response","content":"The answer is 16."}',
        ]
    )

    runner = ToolLoopRunner(registry=registry, max_steps=3)

    outcome = runner.run(
        user_text="What is ((12 + 8) * (7 - 3)) / 5?",
        system_prompt="You are Dexter.",
        model_generate=lambda prompt: prompts.append(prompt) or next(model_outputs),
    )

    assert outcome.final_response == "The answer is 16."
    assert len(outcome.steps) == 1
    assert outcome.steps[0].tool_name == "calculator"
    assert outcome.steps[0].result.ok is True
    assert [trace["type"] for trace in outcome.traces] == [
        "llm_model_output",
        "tool_call_start",
        "tool_call_result",
        "tool_call_end",
        "llm_model_output",
        "llm_final_response",
    ]
    assert prompts[-1]
    assert '"tool_name": "calculator"' in prompts[-1]
    assert '"result": 16.0' in prompts[-1] or '"result": 16' in prompts[-1]


def test_tool_loop_falls_back_to_final_response_when_model_returns_plain_text() -> None:
    registry = ToolRegistry([CalculatorTool()])
    runner = ToolLoopRunner(registry=registry, max_steps=3)

    outcome = runner.run(
        user_text="Hello there",
        system_prompt="You are Dexter.",
        model_generate=lambda _prompt: "Hello back.",
    )

    assert outcome.final_response == "Hello back."
    assert outcome.steps == []


def test_tool_loop_parses_liquid_native_tool_call_tokens() -> None:
    registry = ToolRegistry([CalculatorTool()])
    prompts: list[str] = []
    model_outputs = iter(
        [
            '<|tool_call_start|>[{"name": "calculator", "arguments": {"expression": "479 + 6220"}}]<|tool_call_end|>',
            '{"type":"final_response","content":"The answer is 6699."}',
        ]
    )
    runner = ToolLoopRunner(registry=registry, max_steps=3)

    outcome = runner.run(
        user_text="What is 479 plus 6220?",
        system_prompt="You are Dexter.",
        model_generate=lambda prompt: prompts.append(prompt) or next(model_outputs),
    )

    assert outcome.final_response == "The answer is 6699."
    assert len(outcome.steps) == 1
    assert outcome.steps[0].tool_name == "calculator"
    assert outcome.steps[0].arguments == {"expression": "479 + 6220"}
    assert outcome.steps[0].result.ok is True


def test_tool_loop_forces_final_response_after_successful_tool_result() -> None:
    registry = ToolRegistry([CalculatorTool()])
    model_outputs = iter(
        [
            '<|tool_call_start|>[{"name": "calculator", "arguments": {"expression": "sqrt(479)"}}]<|tool_call_end|>',
            '<|tool_call_start|>[{"name": "calculator", "arguments": {"expression": "sqrt(479)"}}]<|tool_call_end|>[calculator(expression="sqrt(479)")]<|tool_call_end|>',
        ]
    )
    runner = ToolLoopRunner(registry=registry, max_steps=3)

    outcome = runner.run(
        user_text="What is the square root of 479?",
        system_prompt="You are Dexter.",
        model_generate=lambda _prompt: next(model_outputs),
    )

    assert len(outcome.steps) == 1
    assert outcome.steps[0].tool_name == "calculator"
    assert "21.886" in outcome.final_response
