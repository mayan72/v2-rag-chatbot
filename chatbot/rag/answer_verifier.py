"""
Check that the generated answer stays inside evidence + calculator output.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from rag.calculator import CalculationTrace


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _parse(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def allowed_numbers(
    evidence_text: str,
    trace: Optional[CalculationTrace] = None,
    extra: Optional[Iterable[float]] = None,
) -> List[float]:
    values = []
    for raw in NUMBER_RE.findall(evidence_text or ""):
        number = _parse(raw)
        if number is not None:
            values.append(number)
    if trace is not None:
        values.append(float(trace.result))
        for item in trace.inputs.values():
            if isinstance(item, (int, float)):
                values.append(float(item))
    if extra:
        values.extend(float(item) for item in extra)
    return values


def verify_answer(
    answer: str,
    evidence_text: str,
    trace: Optional[CalculationTrace] = None,
) -> dict:
    abstain_prefix = "i don't have enough"
    couldnt = "i couldn't find"
    lowered = (answer or "").strip().lower()
    if lowered.startswith(abstain_prefix) or couldnt in lowered:
        return {"verified": True, "reason": "abstention"}

    allowed = allowed_numbers(evidence_text, trace)
    unsupported = []
    for raw in NUMBER_RE.findall(answer or ""):
        number = _parse(raw)
        if number is None:
            continue
        if abs(number) in {2020, 2021, 2022, 2023, 2024, 2025, 2026, 2019}:
            continue
        if not any(abs(number - item) <= 0.05 or abs(number - round(item, 2)) <= 0.05 for item in allowed):
            # allow integer rounding of calculated percentages
            if trace is not None and abs(number - round(trace.result, 0)) <= 0.51:
                continue
            unsupported.append(number)

    if unsupported:
        return {
            "verified": False,
            "reason": f"unsupported numbers: {unsupported}",
        }
    return {"verified": True, "reason": "grounded"}
