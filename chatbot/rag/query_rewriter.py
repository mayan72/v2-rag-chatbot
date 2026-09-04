"""
Rewrite the user question from clarification + memory slots.
Prefer templates over sending the full conversation to the LLM.
"""

from __future__ import annotations

import re

from rag.conversation_memory import ConversationState
from rag.query_analyzer import QueryAnalysis, METRIC_ALIASES


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_metric_from_text(text: str) -> str:
    lowered = (text or "").lower()
    for alias, canonical in METRIC_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical
    return ""


def looks_like_clarification(text: str, memory: ConversationState) -> bool:
    if not memory or not memory.pending_clarification:
        return False
    stripped = _clean(text)
    if not stripped:
        return False
    if stripped.endswith("?") and len(stripped.split()) > 6:
        return False
    return True


def apply_clarification(original: str, clarification: str, analysis: QueryAnalysis) -> str:
    original = _clean(original)
    clarification = _clean(clarification)
    metric = analysis.metric or extract_metric_from_text(clarification) or extract_metric_from_text(original)
    start = analysis.start_period
    end = analysis.end_period

    years = re.findall(r"\b((?:19|20)\d{2})\b", clarification)
    if len(years) >= 2:
        start, end = years[0], years[1]
    elif len(years) == 1 and start and not end:
        end = years[0]
    elif len(years) == 1 and not start:
        start = years[0]

    if metric and "growth" in (original + " " + clarification).lower():
        if start and end:
            return f"What was the {metric} growth from {start} to {end}?"
        return f"What was the {metric} growth?"

    if original and clarification:
        if original.lower() in clarification.lower():
            return clarification
        return f"{original.rstrip('?')} {clarification}".strip() + (
            "?" if "?" in original else ""
        )
    return clarification or original


def rewrite(
    question: str,
    analysis: QueryAnalysis,
    memory: ConversationState,
) -> str:
    if memory and memory.pending_clarification:
        original = memory.original_query or question
        resolved = apply_clarification(original, question, analysis)
        return _clean(resolved)

    if analysis.is_follow_up and memory:
        metric = analysis.metric or memory.last_metric
        start = analysis.start_period or memory.last_start_period
        end = analysis.end_period or memory.last_end_period
        if metric and start and end and (
            "growth" in question.lower() or "increase" in question.lower()
        ):
            return f"What was the {metric} growth from {start} to {end}?"
        if metric and end and not start:
            return f"What was the {metric} in {end}?"
        if metric and start and not analysis.end_period:
            return f"What was the {metric} in {start}?"

    if analysis.metric and analysis.start_period and analysis.end_period and analysis.requires_calculation:
        return (
            f"What was the {analysis.metric} growth from "
            f"{analysis.start_period} to {analysis.end_period}?"
        )

    return _clean(question)
