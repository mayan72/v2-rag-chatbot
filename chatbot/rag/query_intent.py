"""
Deterministic intent detection for structured-data questions.

Purpose
-------
Identify whether a user question requires structured computation.

This module does NOT:
- select a table
- select columns
- resolve filter values
- execute calculations
- use an LLM

It only identifies the likely analytical intent.

Examples
--------
"What is the total revenue?"
    -> sum

"How many orders are there?"
    -> count

"What is the average price?"
    -> avg

"What is the correlation between quantity and revenue?"
    -> correlation

"Show revenue by region."
    -> group_sum

"What are the top 5 products by revenue?"
    -> top_n

"What is the percentage of total revenue from North?"
    -> percentage

"What happened to aluminium prices?"
    -> semantic / unknown

The output is deliberately conservative. If the question does not
contain sufficient structured intent evidence, it should not force
an analytical operation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Supported intents
# ----------------------------------------------------------------------

STRUCTURED_OPERATIONS = {
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "distinct_count",
    "median",
    "std",
    "variance",
    "correlation",
    "group_count",
    "group_sum",
    "group_avg",
    "group_min",
    "group_max",
    "top_n",
    "bottom_n",
    "percentage",
    "ratio",
    "compare",
    "trend",
}


# ----------------------------------------------------------------------
# Intent result
# ----------------------------------------------------------------------


@dataclass
class QueryIntent:
    """
    Result of deterministic intent detection.
    """

    intent: str

    operation: Optional[str] = None

    confidence: float = 0.0

    structured: bool = False

    group_requested: bool = False

    comparison_requested: bool = False

    ranking_requested: bool = False

    temporal_requested: bool = False

    percentage_requested: bool = False

    correlation_requested: bool = False

    limit: Optional[int] = None

    matched_terms: List[str] = field(
        default_factory=list
    )

    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "operation": self.operation,
            "confidence": self.confidence,
            "structured": self.structured,
            "group_requested": self.group_requested,
            "comparison_requested": self.comparison_requested,
            "ranking_requested": self.ranking_requested,
            "temporal_requested": self.temporal_requested,
            "percentage_requested": self.percentage_requested,
            "correlation_requested": self.correlation_requested,
            "limit": self.limit,
            "matched_terms": self.matched_terms,
            "reason": self.reason,
        }


# ----------------------------------------------------------------------
# Keyword definitions
# ----------------------------------------------------------------------

# Exact / phrase-based expressions are intentionally explicit.
#
# We don't use fuzzy matching here because intent detection should
# be conservative. Fuzzy matching belongs later in column/value
# resolution.

OPERATION_PATTERNS: Dict[str, List[str]] = {
    "count": [
        r"\bhow many\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\btotal number\b",
        r"\bno\.?\s+of\b",
        r"\bnumber\b",
    ],

    "sum": [
        r"\btotal\b",
        r"\bsum\b",
        r"\bsummed\b",
        r"\baggregate\b",
        r"\badd up\b",
        r"\bcombined\b",
        r"\bgrand total\b",
    ],

    "avg": [
        r"\baverage\b",
        r"\bavg\b",
        r"\bmean\b",
        r"\baverage value\b",
        r"\baverage price\b",
        r"\baverage revenue\b",
    ],

    "min": [
        r"\bminimum\b",
        r"\bmin\b",
        r"\blowest\b",
        r"\bsmallest\b",
        r"\bleast\b",
    ],

    "max": [
        r"\bmaximum\b",
        r"\bmax\b",
        r"\bhighest\b",
        r"\blargest\b",
        r"\bgreatest\b",
        r"\bmost\b",
    ],

    "distinct_count": [
        r"\bdistinct\b",
        r"\bunique\b",
        r"\bunique count\b",
        r"\bdistinct count\b",
        r"\bhow many different\b",
        r"\bhow many unique\b",
    ],

    "median": [
        r"\bmedian\b",
    ],

    "std": [
        r"\bstandard deviation\b",
        r"\bstd deviation\b",
        r"\bstd\.?\b",
        r"\bvolatility\b",
    ],

    "variance": [
        r"\bvariance\b",
    ],

    "correlation": [
        r"\bcorrelation\b",
        r"\bcorrelated\b",
        r"\bcorrelate\b",
        r"\brelationship between\b",
        r"\brelationship of\b",
        r"\bassociation between\b",
        r"\bhow related\b",
        r"\brelationship\b",
    ],

    "percentage": [
        r"\bpercentage\b",
        r"\bpercent\b",
        r"\b%\b",
        r"\bproportion\b",
        r"\bshare of\b",
        r"\bpercent of\b",
        r"\bpercentage of\b",
    ],

    "ratio": [
        r"\bratio\b",
        r"\brate relative to\b",
        r"\brelative to\b",
        r"\bdivided by\b",
        r"\bper\b",
    ],

    "compare": [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bvs\b",
        r"\bdifference between\b",
        r"\bdifference in\b",
        r"\bhigher than\b",
        r"\blower than\b",
    ],

    "trend": [
        r"\btrend\b",
        r"\bover time\b",
        r"\bchange over time\b",
        r"\bchanged over time\b",
        r"\bmonthly trend\b",
        r"\byearly trend\b",
        r"\bdaily trend\b",
        r"\bweekly trend\b",
        r"\btime series\b",
    ],
}


GROUP_PATTERNS = [
    r"\bby\s+",
    r"\bper\s+",
    r"\bfor\s+each\b",
    r"\beach\s+",
    r"\bbreakdown by\b",
    r"\bbreakdown of\b",
    r"\bgrouped by\b",
    r"\bgroup by\b",
    r"\bacross\b",
]


RANKING_PATTERNS = [
    r"\btop\b",
    r"\bbottom\b",
    r"\brank\b",
    r"\branking\b",
    r"\branked\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\blargest\b",
    r"\bsmallest\b",
]


TEMPORAL_PATTERNS = [
    r"\bover time\b",
    r"\bby month\b",
    r"\bmonthly\b",
    r"\bby year\b",
    r"\byearly\b",
    r"\bannually\b",
    r"\bby quarter\b",
    r"\bquarterly\b",
    r"\bby week\b",
    r"\bweekly\b",
    r"\bby day\b",
    r"\bdaily\b",
    r"\bbetween\b",
    r"\bfrom\b",
    r"\buntil\b",
    r"\bbefore\b",
    r"\bafter\b",
]


# ----------------------------------------------------------------------
# QueryIntentDetector
# ----------------------------------------------------------------------


class QueryIntentDetector:
    """
    Deterministic structured-query intent detector.

    The detector is deliberately conservative.

    It should answer:

        "Does this question require structured computation?"

    It should NOT answer:

        "Which Revenue column should I use?"

    That is the planner's responsibility.
    """

    def detect(
        self,
        question: str,
    ) -> QueryIntent:
        if not question or not question.strip():
            return QueryIntent(
                intent="unknown",
                structured=False,
                confidence=0.0,
                reason="empty question",
            )

        normalized = self._normalize(
            question
        )

        operation_matches = (
            self._find_operation_matches(
                normalized
            )
        )

        group_requested = self._contains_any(
            normalized,
            GROUP_PATTERNS,
        )

        ranking_requested = self._contains_any(
            normalized,
            RANKING_PATTERNS,
        )

        temporal_requested = self._contains_any(
            normalized,
            TEMPORAL_PATTERNS,
        )

        percentage_requested = (
            "percentage"
            in operation_matches
            or self._contains_any(
                normalized,
                OPERATION_PATTERNS[
                    "percentage"
                ],
            )
        )

        correlation_requested = (
            "correlation"
            in operation_matches
        )

        comparison_requested = (
            "compare"
            in operation_matches
        )

        limit = self._extract_limit(
            normalized
        )

        operation = self._resolve_operation(
            normalized=normalized,
            operation_matches=operation_matches,
            group_requested=group_requested,
            ranking_requested=ranking_requested,
            temporal_requested=temporal_requested,
        )

        # --------------------------------------------------------------
        # No analytical operation
        # --------------------------------------------------------------

        if operation is None:

            if group_requested and self._looks_like_data_question(
                normalized
            ):
                return QueryIntent(
                    intent="structured",
                    operation=None,
                    structured=True,
                    group_requested=True,
                    confidence=0.55,
                    matched_terms=self._flatten_matches(
                        operation_matches
                    ),
                    reason=(
                        "grouping language detected but "
                        "aggregation operation is unresolved"
                    ),
                )

            return QueryIntent(
                intent="semantic",
                structured=False,
                confidence=0.80,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "no deterministic structured "
                    "calculation intent detected"
                ),
            )

        # --------------------------------------------------------------
        # Correlation
        # --------------------------------------------------------------

        if operation == "correlation":

            return QueryIntent(
                intent="structured",
                operation="correlation",
                structured=True,
                correlation_requested=True,
                confidence=0.98,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "correlation intent detected; "
                    "requires two numeric columns"
                ),
            )

        # --------------------------------------------------------------
        # Ranking
        # --------------------------------------------------------------

        if ranking_requested:

            if self._contains_any(
                normalized,
                [
                    r"\btop\b",
                    r"\bbest\b",
                    r"\bhighest\b",
                    r"\blargest\b",
                ],
            ):
                ranking_operation = "top_n"

            else:
                ranking_operation = "bottom_n"

            return QueryIntent(
                intent="structured",
                operation=ranking_operation,
                structured=True,
                ranking_requested=True,
                limit=limit or 5,
                confidence=0.96,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "ranking intent detected"
                ),
            )

        # --------------------------------------------------------------
        # Grouped aggregation
        # --------------------------------------------------------------

        if group_requested:

            grouped_operation = (
                self._group_operation(
                    operation
                )
            )

            return QueryIntent(
                intent="structured",
                operation=grouped_operation,
                structured=True,
                group_requested=True,
                temporal_requested=temporal_requested,
                confidence=0.96,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "grouped structured aggregation "
                    "detected"
                ),
            )

        # --------------------------------------------------------------
        # Percentage
        # --------------------------------------------------------------

        if operation == "percentage":

            return QueryIntent(
                intent="structured",
                operation="percentage",
                structured=True,
                percentage_requested=True,
                confidence=0.97,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "percentage calculation detected"
                ),
            )

        # --------------------------------------------------------------
        # Ratio
        # --------------------------------------------------------------

        if operation == "ratio":

            return QueryIntent(
                intent="structured",
                operation="ratio",
                structured=True,
                confidence=0.97,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "ratio calculation detected"
                ),
            )

        # --------------------------------------------------------------
        # Comparison
        # --------------------------------------------------------------

        if operation == "compare":

            return QueryIntent(
                intent="structured",
                operation="compare",
                structured=True,
                comparison_requested=True,
                confidence=0.95,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "comparison intent detected"
                ),
            )

        # --------------------------------------------------------------
        # Temporal trend
        # --------------------------------------------------------------

        if operation == "trend":

            return QueryIntent(
                intent="structured",
                operation="trend",
                structured=True,
                temporal_requested=True,
                confidence=0.96,
                matched_terms=self._flatten_matches(
                    operation_matches
                ),
                reason=(
                    "time-series/trend intent detected"
                ),
            )

        # --------------------------------------------------------------
        # Standard aggregation
        # --------------------------------------------------------------

        return QueryIntent(
            intent="structured",
            operation=operation,
            structured=True,
            temporal_requested=temporal_requested,
            confidence=self._operation_confidence(
                operation
            ),
            matched_terms=self._flatten_matches(
                operation_matches
            ),
            reason=(
                f"{operation} aggregation intent detected"
            ),
        )

    # ------------------------------------------------------------------
    # OPERATION MATCHING
    # ------------------------------------------------------------------

    def _find_operation_matches(
        self,
        normalized: str,
    ) -> Dict[str, List[str]]:
        matches: Dict[str, List[str]] = {}

        for operation, patterns in (
            OPERATION_PATTERNS.items()
        ):

            found = []

            for pattern in patterns:

                match = re.search(
                    pattern,
                    normalized,
                    flags=re.IGNORECASE,
                )

                if match:
                    found.append(
                        match.group(0)
                    )

            if found:
                matches[
                    operation
                ] = found

        return matches

    # ------------------------------------------------------------------
    # OPERATION RESOLUTION
    # ------------------------------------------------------------------

    def _resolve_operation(
        self,
        normalized: str,
        operation_matches: Dict[
            str,
            List[str],
        ],
        group_requested: bool,
        ranking_requested: bool,
        temporal_requested: bool,
    ) -> Optional[str]:
        """
        Resolve conflicts between overlapping phrases.

        Example:
            "highest revenue"

        contains a max-like word but is normally a ranking request,
        not simply MAX(Revenue).

        Another example:
            "how many unique products"

        contains both count and distinct_count.
        distinct_count wins.
        """

        if not operation_matches:
            return None

        # --------------------------------------------------------------
        # Highest priority: correlation
        # --------------------------------------------------------------

        if "correlation" in operation_matches:
            return "correlation"

        # --------------------------------------------------------------
        # Distinct count beats normal count
        # --------------------------------------------------------------

        if "distinct_count" in operation_matches:
            return "distinct_count"

        # --------------------------------------------------------------
        # Explicit percentage
        # --------------------------------------------------------------

        if "percentage" in operation_matches:
            return "percentage"

        # --------------------------------------------------------------
        # Explicit ratio
        # --------------------------------------------------------------

        if "ratio" in operation_matches:
            return "ratio"

        # --------------------------------------------------------------
        # Explicit comparison
        # --------------------------------------------------------------

        if "compare" in operation_matches:
            return "compare"

        # --------------------------------------------------------------
        # Explicit trend
        # --------------------------------------------------------------

        if "trend" in operation_matches:
            return "trend"

        # --------------------------------------------------------------
        # Ranking should win over MAX/MIN
        # --------------------------------------------------------------

        if ranking_requested:

            if self._contains_any(
                normalized,
                [
                    r"\btop\b",
                    r"\bbest\b",
                    r"\bhighest\b",
                    r"\blargest\b",
                    r"\bgreatest\b",
                    r"\bmost\b",
                ],
            ):
                return "top_n"

            if self._contains_any(
                normalized,
                [
                    r"\bbottom\b",
                    r"\bworst\b",
                    r"\blowest\b",
                    r"\bsmallest\b",
                    r"\bleast\b",
                ],
            ):
                return "bottom_n"

        # --------------------------------------------------------------
        # Explicit median/std/variance
        # --------------------------------------------------------------

        for operation in (
            "median",
            "std",
            "variance",
        ):
            if operation in operation_matches:
                return operation

        # --------------------------------------------------------------
        # Normal aggregations
        # --------------------------------------------------------------

        # "How many" should beat "total" if both appear.
        if "count" in operation_matches:
            return "count"

        if "sum" in operation_matches:
            return "sum"

        if "avg" in operation_matches:
            return "avg"

        if "max" in operation_matches:
            return "max"

        if "min" in operation_matches:
            return "min"

        return None

    # ------------------------------------------------------------------
    # GROUP OPERATION
    # ------------------------------------------------------------------

    def _group_operation(
        self,
        operation: Optional[str],
    ) -> str:
        if operation == "count":
            return "group_count"

        if operation == "sum":
            return "group_sum"

        if operation == "avg":
            return "group_avg"

        if operation == "min":
            return "group_min"

        if operation == "max":
            return "group_max"

        if operation == "distinct_count":
            return "group_count"

        # If a question says "by region" but doesn't
        # explicitly say what aggregation to perform,
        # keep the operation unresolved rather than
        # silently assuming SUM.
        return "group_count"

    # ------------------------------------------------------------------
    # GROUP DETECTION
    # ------------------------------------------------------------------

    def _contains_any(
        self,
        text: str,
        patterns: List[str],
    ) -> bool:

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # LIMIT
    # ------------------------------------------------------------------

    def _extract_limit(
        self,
        normalized: str,
    ) -> Optional[int]:
        """
        Extract ranking limits.

        Examples:
            top 5
            top five
            first 10
            bottom 3
        """

        numeric_match = re.search(
            r"\b(?:top|bottom|first|last)\s+(\d{1,4})\b",
            normalized,
            flags=re.IGNORECASE,
        )

        if numeric_match:

            try:
                value = int(
                    numeric_match.group(1)
                )

                if 1 <= value <= 1000:
                    return value

            except ValueError:
                pass

        word_limits = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }

        word_match = re.search(
            r"\b(?:top|bottom|first|last)\s+"
            r"(one|two|three|four|five|six|seven|"
            r"eight|nine|ten)\b",
            normalized,
            flags=re.IGNORECASE,
        )

        if word_match:
            return word_limits.get(
                word_match.group(1).casefold()
            )

        return None

    # ------------------------------------------------------------------
    # DATA-QUESTION HEURISTIC
    # ------------------------------------------------------------------

    def _looks_like_data_question(
        self,
        normalized: str,
    ) -> bool:
        """
        Detect whether a question sounds like it is asking
        about uploaded data.

        This is deliberately weak. It is only used when grouping
        language exists but no aggregation operation was identified.
        """

        data_patterns = [
            r"\bwhat\b",
            r"\bwhich\b",
            r"\bshow\b",
            r"\blist\b",
            r"\bgive\b",
            r"\bdata\b",
            r"\breport\b",
            r"\bvalues?\b",
            r"\bfigures?\b",
        ]

        return self._contains_any(
            normalized,
            data_patterns,
        )

    # ------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------

    def _operation_confidence(
        self,
        operation: str,
    ) -> float:
        high_confidence = {
            "count": 0.97,
            "sum": 0.97,
            "avg": 0.97,
            "min": 0.96,
            "max": 0.96,
            "distinct_count": 0.98,
            "median": 0.98,
            "std": 0.98,
            "variance": 0.98,
        }

        return high_confidence.get(
            operation,
            0.90,
        )

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize(
        self,
        question: str,
    ) -> str:
        text = str(
            question
        ).casefold()

        # Normalize common mathematical symbols.
        text = text.replace(
            "%",
            " percent ",
        )

        text = text.replace(
            "&",
            " and ",
        )

        # Normalize punctuation while preserving
        # useful numeric characters.
        text = re.sub(
            r"[^a-z0-9%.\-]+",
            " ",
            text,
        )

        return re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

    # ------------------------------------------------------------------
    # MATCH FLATTENING
    # ------------------------------------------------------------------

    def _flatten_matches(
        self,
        matches: Dict[
            str,
            List[str],
        ],
    ) -> List[str]:

        flattened = []

        for values in matches.values():

            for value in values:

                if value not in flattened:
                    flattened.append(
                        value
                    )

        return flattened


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------


_default_detector = QueryIntentDetector()


def detect_query_intent(
    question: str,
) -> QueryIntent:
    """
    Convenience wrapper.
    """

    return _default_detector.detect(
        question
    )