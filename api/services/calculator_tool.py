from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Any

from .tool_runtime import ToolResult, ToolSpec


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
}


@dataclass(frozen=True)
class CalculatorTool:
    spec: ToolSpec = ToolSpec(
        name="calculator",
        description=(
            "Evaluate deterministic arithmetic and math expressions. "
            "Use for calculations, formulas, parentheses, and standard math functions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        safety_mode="read",
        enabled=True,
    )

    def execute(self, arguments: dict[str, Any], _context: dict[str, Any] | None = None) -> ToolResult:
        expression = str((arguments or {}).get("expression", "")).strip()
        if not expression:
            return ToolResult(ok=False, content="", error="Missing required argument: expression")

        try:
            normalized = self._normalize_expression(expression)
            result = self._evaluate(normalized)
        except Exception as exc:
            return ToolResult(ok=False, content="", error=str(exc))

        return ToolResult(
            ok=True,
            content={
                "expression": normalized,
                "result": result,
            },
            error=None,
        )

    def _normalize_expression(self, expression: str) -> str:
        return (
            expression.replace("^", "**")
            .replace("×", "*")
            .replace("÷", "/")
            .strip()
        )

    def _evaluate(self, expression: str) -> float | int:
        node = ast.parse(expression, mode="eval")
        return self._eval_node(node.body)

    def _eval_node(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Unsupported constant in expression")

        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise ValueError(f"Unsupported identifier: {node.id}")

        if isinstance(node, ast.BinOp):
            operator_fn = _BINARY_OPERATORS.get(type(node.op))
            if not operator_fn:
                raise ValueError("Unsupported binary operator")
            return operator_fn(self._eval_node(node.left), self._eval_node(node.right))

        if isinstance(node, ast.UnaryOp):
            operator_fn = _UNARY_OPERATORS.get(type(node.op))
            if not operator_fn:
                raise ValueError("Unsupported unary operator")
            return operator_fn(self._eval_node(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported function call")
            function_name = node.func.id
            function = _FUNCTIONS.get(function_name)
            if not function:
                raise ValueError(f"Unsupported function: {function_name}")
            if node.keywords:
                raise ValueError("Keyword arguments are not supported")
            args = [self._eval_node(arg) for arg in node.args]
            return function(*args)

        raise ValueError(f"Unsupported expression node: {type(node).__name__}")
