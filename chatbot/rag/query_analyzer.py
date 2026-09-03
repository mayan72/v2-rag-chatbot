"""
Rule-based query analysis for clarification-driven RAG.

Reuses QueryIntentDetector for structured/table intent.
Does not guess missing metrics or periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import re

from rag.query_intent import QueryIntentDetector


METRIC_ALIASES = {
    "revenue": "revenue",
    "sales": "revenue",
    "turnover": "revenue",
    "profit": "profit",
    "income": "profit",
    "earnings": "profit",
    "employee": "employee",
    "employees": "employee",
    "headcount": "employee",
    "quantity": "quantity",
    "volume": "quantity",
    "production": "production",
    "milk": "milk",
    "cost": "cost",
    "price": "price",
    "margin": "margin",
}

GROWTH_RE = re.compile(
    r"\b(growth|increase|decrease|change|yoy|year over year)\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(
    r"\b(percent(?:age)?(?:\s+(?:increase|decrease|change|growth))?)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
FOLLOW_UP_RE = re.compile(
    r"^(and|what about|how about|also|same for)\b",
    re.IGNORECASE,
)


@dataclass
class QueryAnalysis:
    intent: str = "factual"
    is_ambiguous: bool = False
    requires_calculation: bool = False
    numerical: bool = False
    metric: str = ""
    start_period: str = ""
    end_period: str = ""
    required_values: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    clarification_question: str = ""
    clarification_options: List[str] = field(default_factory=list)
    needs_period: bool = False
    is_follow_up: bool = False
    structured: bool = False
    operation: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "is_ambiguous": self.is_ambiguous,
            "requires_calculation": self.requires_calculation,
            "numerical": self.numerical,
            "metric": self.metric,
            "start_period": self.start_period,
            "end_period": self.end_period,
            "required_values": self.required_values,
            "missing_information": self.missing_information,
            "clarification_question": self.clarification_question,
            "clarification_options": self.clarification_options,
            "needs_period": self.needs_period,
            "is_follow_up": self.is_follow_up,
            "structured": self.structured,
            "operation": self.operation,
            "reason": self.reason,
        }


class QueryAnalyzer:
    def __init__(self):
        self.intent_detector = QueryIntentDetector()

    def analyze(self, question: str, memory=None) -> QueryAnalysis:
        text = (question or "").strip()
        analysis = QueryAnalysis()
        if not text:
            analysis.is_ambiguous = True
            analysis.missing_information = ["question"]
            analysis.clarification_question = "What would you like to know?"
            return analysis

        intent = self.intent_detector.detect(text)
        analysis.structured = bool(intent.structured)
        analysis.operation = intent.operation
        analysis.is_follow_up = bool(
            FOLLOW_UP_RE.search(text)
            or (memory and memory.pending_clarification)
        )

        metric = self._extract_metric(text)
        years = YEAR_RE.findall(text)
        is_growth = bool(GROWTH_RE.search(text) or PERCENT_RE.search(text))
        explicit_numbers = bool(
            re.search(r"from\s+-?\d+(?:\.\d+)?\s+to\s+-?\d+(?:\.\d+)?", text, re.I)
        )

        if memory:
            if not metric and memory.last_metric:
                metric = memory.last_metric
                analysis.is_follow_up = True
            if len(years) == 1 and memory.last_start_period and not memory.last_end_period:
                years = [memory.last_start_period, years[0]]
            elif len(years) == 1 and memory.last_end_period == years[0] and memory.last_start_period:
                years = [memory.last_start_period, years[0]]
            elif not years and memory.last_start_period and memory.last_end_period:
                if is_growth or analysis.is_follow_up:
                    years = [memory.last_start_period, memory.last_end_period]

        analysis.metric = metric
        if len(years) >= 2:
            analysis.start_period = years[0]
            analysis.end_period = years[1]
        elif len(years) == 1:
            analysis.start_period = years[0]

        if is_growth or explicit_numbers:
            analysis.numerical = True
            analysis.requires_calculation = True
            analysis.intent = "numerical"
        elif intent.structured:
            analysis.numerical = True
            analysis.intent = "numerical"
        else:
            analysis.intent = "factual"

        if is_growth and not explicit_numbers:
            if not analysis.metric:
                analysis.is_ambiguous = True
                analysis.missing_information.append("metric")
                analysis.clarification_options = [
                    "revenue growth",
                    "profit growth",
                    "employee growth",
                ]
                analysis.clarification_question = (
                    "Do you mean revenue growth, profit growth, or employee growth?"
                )
                analysis.reason = "ambiguous_metric"
            elif not analysis.start_period or not analysis.end_period:
                analysis.needs_period = True
                analysis.missing_information.append("period")
                analysis.reason = "missing_period"

            if analysis.metric and analysis.start_period and analysis.end_period:
                analysis.required_values = [
                    f"{analysis.start_period} {analysis.metric}",
                    f"{analysis.end_period} {analysis.metric}",
                ]
            elif analysis.metric:
                analysis.required_values = [
                    f"{analysis.metric} start value",
                    f"{analysis.metric} end value",
                ]

        if PERCENT_RE.search(text) and not analysis.metric and not explicit_numbers and not is_growth:
            if "percentage of" not in text.lower():
                analysis.is_ambiguous = True
                analysis.numerical = True
                analysis.intent = "numerical"
                analysis.missing_information.append("metric")
                analysis.clarification_question = (
                    "What was the percentage of? For example revenue, profit, or another metric?"
                )
                analysis.reason = "ambiguous_percentage"

        return analysis

    def _extract_metric(self, text: str) -> str:
        lowered = text.lower()
        for alias, canonical in METRIC_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return canonical
        return ""
