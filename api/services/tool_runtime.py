from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    safety_mode: str
    enabled: bool = True


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: dict[str, Any] | list[Any] | str
    error: str | None

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "content": self.content,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToolExecutionStep:
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResult


@dataclass(frozen=True)
class ToolLoopOutcome:
    final_response: str
    steps: list[ToolExecutionStep]
    traces: list[dict[str, Any]]


class ToolRegistry:
    def __init__(self, tools: list[Any]) -> None:
        self._tools = {tool.spec.name: tool for tool in tools if getattr(tool.spec, "enabled", False)}

    def list_tools(self, _context: dict[str, Any] | None = None) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(ok=False, content="", error=f"Unknown tool: {tool_name}")
        return tool.execute(arguments, context)


class ToolLoopRunner:
    def __init__(self, registry: ToolRegistry, max_steps: int = 3) -> None:
        self._registry = registry
        self._max_steps = max_steps

    def run(
        self,
        *,
        user_text: str,
        system_prompt: str,
        model_generate: Any,
    ) -> ToolLoopOutcome:
        steps: list[ToolExecutionStep] = []
        traces: list[dict[str, Any]] = []
        working_memory: dict[str, Any] = {
            "user_request": user_text,
            "steps": [],
        }
        tool_specs = self._registry.list_tools({"user_request": user_text})
        require_final_response = False

        for _ in range(self._max_steps + 1):
            prompt = self._build_prompt(
                system_prompt=system_prompt,
                user_text=user_text,
                tool_specs=tool_specs,
                working_memory=working_memory,
                require_final_response=require_final_response,
            )
            raw_output = str(model_generate(prompt) or "").strip()
            traces.append(
                {
                    "type": "llm_model_output",
                    "raw_output": raw_output,
                }
            )
            parsed = self._parse_response(raw_output)

            if parsed["type"] == "final_response":
                final_response = str(parsed["content"]).strip()
                traces.append(
                    {
                        "type": "llm_final_response",
                        "content": final_response,
                    }
                )
                return ToolLoopOutcome(
                    final_response=final_response,
                    steps=steps,
                    traces=traces,
                )

            arguments = parsed.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            tool_name = str(parsed.get("tool_name", "")).strip()

            if require_final_response:
                final_response = self._fallback_final_response(user_text, steps)
                traces.append(
                    {
                        "type": "llm_final_response",
                        "content": final_response,
                        "fallback": True,
                    }
                )
                return ToolLoopOutcome(
                    final_response=final_response,
                    steps=steps,
                    traces=traces,
                )

            traces.append(
                {
                    "type": "tool_call_start",
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
            )
            result = self._registry.execute(
                tool_name,
                arguments,
                {"user_request": user_text, "steps": steps},
            )
            step = ToolExecutionStep(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
            )
            steps.append(step)
            traces.append(
                {
                    "type": "tool_call_result",
                    "tool_name": tool_name,
                    "ok": result.ok,
                    "content": result.content,
                    "error": result.error,
                }
            )
            traces.append(
                {
                    "type": "tool_call_end",
                    "tool_name": tool_name,
                    "ok": result.ok,
                }
            )
            working_memory["steps"].append(
                {
                    "tool_name": step.tool_name,
                    "arguments": step.arguments,
                    "result": step.result.as_prompt_payload(),
                }
            )
            if result.ok:
                require_final_response = True

        final_response = self._fallback_final_response(user_text, steps)
        traces.append(
            {
                "type": "llm_final_response",
                "content": final_response,
                "fallback": True,
            }
        )
        return ToolLoopOutcome(
            final_response=final_response,
            steps=steps,
            traces=traces,
        )

    def _build_prompt(
        self,
        *,
        system_prompt: str,
        user_text: str,
        tool_specs: list[ToolSpec],
        working_memory: dict[str, Any],
        require_final_response: bool,
    ) -> str:
        tools_payload = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "safety_mode": spec.safety_mode,
            }
            for spec in tool_specs
        ]
        additional_instruction = (
            "A tool result is already available. Do not call another tool. "
            "Respond with a final_response in plain language using the tool result.\n\n"
            if require_final_response
            else ""
        )
        return (
            f"{system_prompt}\n\n"
            "You may either answer directly or request exactly one tool call.\n"
            "If you need a tool, respond with JSON only using:\n"
            '{"type":"tool_call","tool_name":"...","arguments":{...}}\n'
            "If you do not need a tool, respond with JSON only using:\n"
            '{"type":"final_response","content":"..."}\n'
            "Use the calculator for arithmetic or formula evaluation.\n"
            "Never invent tools.\n\n"
            f"{additional_instruction}"
            f"Available tools:\n{json.dumps(tools_payload, indent=2)}\n\n"
            f"Working memory:\n{json.dumps(working_memory, indent=2)}\n\n"
            f"User request:\n{user_text}\n"
        )

    def _parse_response(self, raw_output: str) -> dict[str, Any]:
        native_tool_call = self._parse_native_tool_call(raw_output)
        if native_tool_call:
            return native_tool_call

        candidate = self._extract_json_candidate(raw_output)
        if candidate:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and parsed.get("type") in {"tool_call", "final_response"}:
                    return parsed
            except json.JSONDecodeError:
                pass
        return {
            "type": "final_response",
            "content": raw_output,
        }

    def _parse_native_tool_call(self, raw_output: str) -> dict[str, Any] | None:
        match = re.search(
            r"<\|tool_call_start\|>\s*(\[.*?\])\s*<\|tool_call_end\|>",
            raw_output,
            flags=re.DOTALL,
        )
        if not match:
            return None

        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, list) or not parsed:
            return None

        first_call = parsed[0]
        if not isinstance(first_call, dict):
            return None

        tool_name = first_call.get("name")
        arguments = first_call.get("arguments", {})
        if not isinstance(tool_name, str):
            return None
        if not isinstance(arguments, dict):
            arguments = {}

        return {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
        }

    def _extract_json_candidate(self, raw_output: str) -> str | None:
        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, flags=re.DOTALL)
        if fenced_match:
            return fenced_match.group(1)

        stripped = raw_output.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
        return None

    def _fallback_final_response(self, user_text: str, steps: list[ToolExecutionStep]) -> str:
        if steps:
            last_step = steps[-1]
            if last_step.result.ok and isinstance(last_step.result.content, dict):
                result = last_step.result.content.get("result")
                if result is not None:
                    return f"The answer is {result}."
            if last_step.result.error:
                return f"I couldn't complete that calculation: {last_step.result.error}"
        return f"I couldn't complete the request: {user_text}"
