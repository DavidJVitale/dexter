from __future__ import annotations

import json
import re
from typing import Any

from .tool_runtime import ToolExecutionStep, ToolLoopOutcome, ToolRegistry


class LiquidToolRunner:
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
        tools_payload = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            }
            for spec in self._registry.list_tools({"user_request": user_text})
        ]

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "List of tools: "
                    f"{json.dumps(tools_payload)}\n"
                    "Output function calls as JSON."
                ),
            },
            {"role": "user", "content": user_text},
        ]

        # This flow intentionally follows Liquid's documented tool-use pattern:
        # first pass generates a native tool call wrapped in Liquid-specific
        # special tokens, the external tool executes outside the model, then the
        # raw assistant tool-call output plus a `tool` role result message are
        # appended for the second pass. This is knowingly model-specific glue.
        #
        # In the future, if we want stronger model agnosticism, we should
        # abstract these quirks behind a provider adapter layer rather than let
        # the core assistant runtime understand Liquid token conventions,
        # Liquid-specific role sequencing, or Liquid-specific "gotchas" around
        # function-call formatting. Other models may prefer different message
        # roles, different special tokens, different schemas, or no special
        # tokens at all, so this code should be treated as a localized adapter,
        # not the universal tool-calling architecture.
        for _ in range(self._max_steps):
            raw_output = str(model_generate(messages) or "").strip()
            traces.append({"type": "llm_model_output", "raw_output": raw_output})

            parsed_tool_call = self._parse_native_tool_call(raw_output)
            if not parsed_tool_call:
                final_response = self._sanitize_final_response(raw_output)
                if not final_response and steps:
                    final_response = self._fallback_final_response(steps)
                    traces.append(
                        {
                            "type": "llm_final_response",
                            "content": final_response,
                            "fallback": True,
                        }
                    )
                else:
                    traces.append(
                        {
                            "type": "llm_final_response",
                            "content": final_response,
                        }
                    )
                return ToolLoopOutcome(final_response=final_response, steps=steps, traces=traces)

            tool_name = parsed_tool_call["tool_name"]
            arguments = parsed_tool_call["arguments"]
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
            step = ToolExecutionStep(tool_name=tool_name, arguments=arguments, result=result)
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

            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "tool", "content": json.dumps(result.content)})

        final_response = self._fallback_final_response(steps)
        traces.append(
            {
                "type": "llm_final_response",
                "content": final_response,
                "fallback": True,
            }
        )
        return ToolLoopOutcome(final_response=final_response, steps=steps, traces=traces)

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
        return {"tool_name": tool_name, "arguments": arguments}

    def _sanitize_final_response(self, raw_output: str) -> str:
        cleaned = raw_output or ""
        cleaned = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.replace("<|tool_call_start|>", "")
        cleaned = cleaned.replace("<|tool_call_end|>", "")
        cleaned = cleaned.replace("<|tool_response_start|>", "")
        cleaned = cleaned.replace("<|tool_response_end|>", "")
        cleaned = re.sub(r"\[[A-Za-z_][A-Za-z0-9_]*\(.*?\)\]", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _fallback_final_response(self, steps: list[ToolExecutionStep]) -> str:
        if steps:
            last_step = steps[-1]
            if last_step.result.ok and isinstance(last_step.result.content, dict):
                result = last_step.result.content.get("result")
                if result is not None:
                    return f"The answer is {result}."
            if last_step.result.error:
                return f"I couldn't complete that calculation: {last_step.result.error}"
        return "I couldn't complete that request."
