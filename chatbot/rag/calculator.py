"""
Deterministic numerical calculator.

The LLM must not perform arithmetic when a formula can be applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re


class CalculationError(ValueError):
    pass


@dataclass
class CalculationTrace:
    operation: str
    formula: str
    inputs: dict
    result: float
    verified: bool
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "formula": self.formula,
            "inputs": self.inputs,
            "result": self.result,
            "verified": self.verified,
            "error": self.error,
        }


def percentage_change(old_value: float, new_value: float) -> float:
    if old_value == 0:
        raise CalculationError("Cannot divide by zero when computing percentage change.")
    return ((new_value - old_value) / old_value) * 100


def difference(new_value: float, old_value: float) -> float:
    return new_value - old_value


def ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        raise CalculationError("Cannot divide by zero when computing a ratio.")
    return numerator / denominator


def average(values: list) -> float:
    if not values:
        raise CalculationError("Cannot compute an average with no values.")
    return sum(values) / len(values)


def margin(profit: float, revenue: float) -> float:
    if revenue == 0:
        raise CalculationError("Cannot divide by zero when computing a margin.")
    return (profit / revenue) * 100


def verify_trace(trace: CalculationTrace) -> CalculationTrace:
    try:
        if trace.operation in {"percentage_change", "growth", "growth_rate"}:
            old_value = float(trace.inputs["old_value"])
            new_value = float(trace.inputs["new_value"])
            expected = percentage_change(old_value, new_value)
        elif trace.operation == "difference":
            expected = difference(
                float(trace.inputs["new_value"]),
                float(trace.inputs["old_value"]),
            )
        elif trace.operation == "ratio":
            expected = ratio(
                float(trace.inputs["numerator"]),
                float(trace.inputs["denominator"]),
            )
        elif trace.operation == "average":
            expected = average(list(trace.inputs["values"]))
        elif trace.operation == "margin":
            expected = margin(
                float(trace.inputs["profit"]),
                float(trace.inputs["revenue"]),
            )
        else:
            trace.verified = False
            trace.error = f"Unsupported operation: {trace.operation}"
            return trace

        if abs(expected - float(trace.result)) > 1e-6:
            trace.verified = False
            trace.error = "Calculated result does not match the formula."
            return trace

        if trace.result != trace.result:
            trace.verified = False
            trace.error = "Result is not a valid number."
            return trace

        trace.verified = True
        trace.error = ""
        return trace
    except (KeyError, TypeError, ValueError, CalculationError) as exc:
        trace.verified = False
        trace.error = str(exc)
        return trace


def compute_percentage_change(old_value: float, new_value: float) -> CalculationTrace:
    result = percentage_change(old_value, new_value)
    trace = CalculationTrace(
        operation="percentage_change",
        formula="((new_value - old_value) / old_value) * 100",
        inputs={"old_value": old_value, "new_value": new_value},
        result=round(result, 6),
        verified=False,
    )
    return verify_trace(trace)


_FROM_TO = re.compile(
    r"(?:from\s+)?(?P<old>-?\d+(?:\.\d+)?)\s+(?:to|and)\s+(?P<new>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_explicit_pair(question: str) -> Optional[tuple]:
    match = _FROM_TO.search(question or "")
    if not match:
        return None
    return float(match.group("old")), float(match.group("new"))


def try_explicit_percentage(question: str) -> Optional[CalculationTrace]:
    text = (question or "").lower()
    if not any(
        token in text
        for token in (
            "percent",
            "percentage",
            "growth",
            "increase",
            "decrease",
            "change",
        )
    ):
        return None
    pair = extract_explicit_pair(question)
    if pair is None:
        return None
    old_value, new_value = pair
    return compute_percentage_change(old_value, new_value)
