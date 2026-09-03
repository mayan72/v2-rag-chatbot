"""
Clarification policy and option parsing.
"""

from __future__ import annotations

from typing import List

MAX_CLARIFICATION_TURNS = 2
from rag.query_analyzer import QueryAnalysis


def should_ask(analysis: QueryAnalysis, memory: ConversationState) -> bool:
    if memory.clarification_turns >= MAX_CLARIFICATION_TURNS:
        return False
    return bool(analysis.is_ambiguous and analysis.clarification_question)


def should_ask_period(
    analysis: QueryAnalysis,
    memory: ConversationState,
    period_options: List[str],
) -> bool:
    if memory.clarification_turns >= MAX_CLARIFICATION_TURNS:
        return False
    if not analysis.needs_period:
        return False
    if analysis.start_period and analysis.end_period:
        return False
    return len(period_options) >= 2


def period_question(options: List[str]) -> str:
    if not options:
        return "Which time period do you mean?"
    rendered = ", ".join(options[:-1])
    if len(options) == 1:
        return f"Do you mean {options[0]}?"
    return f"Which period do you mean: {rendered}, or {options[-1]}?"


def parse_options_from_question(question: str) -> List[str]:
    text = (question or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace("Do you mean", "").replace("?", "")
    text = text.replace(" or ", ", ")
    parts = [part.strip(" .") for part in text.split(",")]
    return [part for part in parts if part]
