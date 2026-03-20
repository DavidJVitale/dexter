from __future__ import annotations

import math

import pytest

from api.services.calculator_tool import CalculatorTool


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(12 + 8) * (7 - 3) / 5", 16),
        ("2^5 + 1", 33),
        ("sqrt(81) + abs(-4)", 13),
        ("sin(pi / 2) + log10(1000)", 4),
        ("round((3.14159 * 2), 3)", 6.283),
    ],
)
def test_calculator_tool_evaluates_supported_math(expression: str, expected: float) -> None:
    tool = CalculatorTool()

    result = tool.execute({"expression": expression})

    assert result.ok is True
    assert result.error is None
    assert math.isclose(float(result.content["result"]), expected, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('pwd')",
        "open('secret.txt')",
        "[1, 2, 3]",
        "lambda x: x + 1",
    ],
)
def test_calculator_tool_rejects_unsafe_or_unsupported_syntax(expression: str) -> None:
    tool = CalculatorTool()

    result = tool.execute({"expression": expression})

    assert result.ok is False
    assert result.error

