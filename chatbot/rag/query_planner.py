"""
Production-oriented structured query planner.

Responsibilities
----------------
1. Detect structured intent using QueryIntentDetector.
2. Resolve the best logical table from its schema.
3. Resolve user-mentioned columns against the actual schema.
4. Resolve filters conservatively.
5. Support multi-column operations such as correlation.
6. Support grouping / ranking / comparison / percentage plans.
7. Use the LLM only as a candidate planner when deterministic
   intent/schema resolution is insufficient.
8. NEVER silently drop an invalid filter or column.
9. NEVER claim a plan is valid merely because an LLM produced it.

The planner does NOT execute calculations.

Pipeline
--------
Question
   ↓
QueryIntentDetector
   ↓
Schema/table resolution
   ↓
Column resolution
   ↓
Filter resolution
   ↓
Plan validation
   ↓
QueryPlan
   ↓
StructuredExecutor
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rag.query_intent import (
    QueryIntent,
    QueryIntentDetector,
)
from rag.text_normalize import (
    best_column_match,
    best_value_match,
    normalize_text,
)
from logger.console import qlog

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Supported operations
# ----------------------------------------------------------------------

ALLOWED_OPERATIONS = {
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


NUMERIC_OPERATIONS = {
    "sum",
    "avg",
    "min",
    "max",
    "median",
    "std",
    "variance",
    "correlation",
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


GROUP_OPERATIONS = {
    "group_count",
    "group_sum",
    "group_avg",
    "group_min",
    "group_max",
}


# ----------------------------------------------------------------------
# Query filter
# ----------------------------------------------------------------------


@dataclass
class QueryFilter:
    column: str
    op: str
    value: Any
    score: float = 1.0

    # Whether the value was actually verified against
    # the selected table.
    validated: bool = False

    # Original user text.
    requested_value: Any = None


# ----------------------------------------------------------------------
# Query plan
# ----------------------------------------------------------------------


@dataclass
class QueryPlan:
    mode: str

    operation: Optional[str] = None

    table_id: Optional[str] = None

    target_column: Optional[str] = None

    # Used for operations such as correlation.
    target_columns: List[str] = field(
        default_factory=list
    )

    group_by: List[str] = field(
        default_factory=list
    )

    filters: List[QueryFilter] = field(
        default_factory=list
    )

    sort_column: Optional[str] = None

    sort_direction: Optional[str] = None

    limit: Optional[int] = None

    confidence: float = 0.0

    reason: str = ""

    # Planner diagnostics.
    valid: bool = False

    validation_errors: List[str] = field(
        default_factory=list
    )

    intent: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "operation": self.operation,
            "table_id": self.table_id,
            "target_column": self.target_column,
            "target_columns": self.target_columns,
            "group_by": self.group_by,
            "filters": [
                {
                    "column": item.column,
                    "op": item.op,
                    "value": item.value,
                    "score": item.score,
                    "validated": item.validated,
                    "requested_value": item.requested_value,
                }
                for item in self.filters
            ],
            "sort_column": self.sort_column,
            "sort_direction": self.sort_direction,
            "limit": self.limit,
            "confidence": self.confidence,
            "reason": self.reason,
            "valid": self.valid,
            "validation_errors": self.validation_errors,
            "intent": self.intent,
        }


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------


class QueryPlanner:

    def __init__(
        self,
        intent_detector: Optional[
            QueryIntentDetector
        ] = None,
        column_min_score: float = 0.78,
        value_min_score: float = 0.88,
        ambiguity_margin: float = 0.05,
    ):
        self.intent_detector = (
            intent_detector
            or QueryIntentDetector()
        )

        self.column_min_score = float(
            column_min_score
        )

        self.value_min_score = float(
            value_min_score
        )

        self.ambiguity_margin = float(
            ambiguity_margin
        )

    # ==================================================================
    # PUBLIC
    # ==================================================================

    def plan(
        self,
        question: str,
        schemas: List[dict],
        llm: Any = None,
    ) -> QueryPlan:

        if not question or not question.strip():
            return QueryPlan(
                mode="semantic",
                confidence=0.0,
                reason="empty question",
                valid=False,
                validation_errors=[
                    "Question is empty."
                ],
            )

        if not schemas:
            return QueryPlan(
                mode="semantic",
                confidence=0.0,
                reason="no structured schemas available",
                valid=False,
                validation_errors=[
                    "No structured tables are available."
                ],
            )

        # --------------------------------------------------------------
        # STEP 1
        # --------------------------------------------------------------

        intent = self.intent_detector.detect(
            question
        )

        qlog(
            "INTENT",
            operation=intent.operation or "none",
            structured=intent.structured,
            intent=intent.intent,
            confidence=intent.confidence,
            reason=intent.reason,
        )

        # --------------------------------------------------------------
        # Semantic question
        # --------------------------------------------------------------

        if not intent.structured:

            return QueryPlan(
                mode="semantic",
                confidence=intent.confidence,
                reason=intent.reason,
                valid=True,
                intent=intent.intent,
            )

        # --------------------------------------------------------------
        # Structured question but unresolved operation
        # --------------------------------------------------------------

        if not intent.operation:

            return QueryPlan(
                mode="structured",
                operation=None,
                confidence=intent.confidence,
                reason=(
                    "Structured/data intent detected, "
                    "but calculation operation could not "
                    "be determined safely."
                ),
                valid=False,
                intent=intent.intent,
                validation_errors=[
                    "Structured operation is unresolved."
                ],
            )

        # --------------------------------------------------------------
        # Validate supported operation
        # --------------------------------------------------------------

        if intent.operation not in ALLOWED_OPERATIONS:

            return QueryPlan(
                mode="structured",
                operation=intent.operation,
                confidence=0.0,
                reason="unsupported structured operation",
                valid=False,
                intent=intent.intent,
                validation_errors=[
                    f"Unsupported operation: "
                    f"{intent.operation}"
                ],
            )

        # --------------------------------------------------------------
        # STEP 2 - table candidates
        # --------------------------------------------------------------

        table_candidates = (
            self._rank_tables(
                question=question,
                schemas=schemas,
                intent=intent,
            )
        )

        if not table_candidates:

            return QueryPlan(
                mode="structured",
                operation=intent.operation,
                confidence=0.0,
                reason="no compatible table found",
                valid=False,
                intent=intent.intent,
                validation_errors=[
                    "No compatible structured table was found."
                ],
            )

        # --------------------------------------------------------------
        # STEP 3 - attempt deterministic planning
        # --------------------------------------------------------------

        deterministic_plan = (
            self._build_deterministic_plan(
                question=question,
                intent=intent,
                table_candidates=table_candidates,
            )
        )

        if deterministic_plan is not None:

            if deterministic_plan.valid:

                qlog(
                    "PLAN",
                    source="deterministic",
                    valid=True,
                    operation=deterministic_plan.operation,
                    table=deterministic_plan.table_id,
                    column=deterministic_plan.target_column,
                )

                return deterministic_plan

            # IMPORTANT:
            # Do not immediately send the question to semantic RAG.
            #
            # It is a structured question. If it cannot be safely
            # planned, we must either use a validated LLM candidate
            # or return an explicit structured failure.

            if llm is None:

                return deterministic_plan

        # --------------------------------------------------------------
        # STEP 4 - optional LLM candidate planning
        # --------------------------------------------------------------

        if llm is not None:

            llm_plan = self._build_llm_candidate(
                question=question,
                intent=intent,
                schemas=schemas,
                llm=llm,
            )

            if llm_plan is not None:

                validated = (
                    self._validate_plan(
                        plan=llm_plan,
                        schemas=schemas,
                    )
                )

                if validated.valid:

                    # Never blindly trust LLM confidence.
                    validated.confidence = min(
                        validated.confidence,
                        0.90,
                    )

                    validated.reason = (
                        "validated LLM structured plan"
                    )

                    return validated

                logger.warning(
                    "Rejected invalid LLM structured plan | "
                    "errors=%s | plan=%s",
                    validated.validation_errors,
                    validated.to_dict(),
                )

        # --------------------------------------------------------------
        # STEP 5 - safe failure
        # --------------------------------------------------------------

        if deterministic_plan is not None:

            deterministic_plan.reason = (
                deterministic_plan.reason
                or "structured plan could not be validated"
            )

            return deterministic_plan

        return QueryPlan(
            mode="structured",
            operation=intent.operation,
            confidence=0.0,
            reason=(
                "Structured question could not be "
                "resolved safely."
            ),
            valid=False,
            intent=intent.intent,
            validation_errors=[
                "Unable to construct a validated structured plan."
            ],
        )

    # ==================================================================
    # TABLE RESOLUTION
    # ==================================================================

    def _rank_tables(
        self,
        question: str,
        schemas: List[dict],
        intent: QueryIntent,
    ) -> List[Tuple[dict, float]]:
        """
        Rank tables using schema compatibility.

        No answer is produced here.

        A table gets points for:
            - matching question terms against column names
            - having required numeric columns for numeric operations
            - having dimensions for grouping
            - having time columns for trends
        """

        ranked = []

        question_normalized = normalize_text(
            question
        )

        for schema in schemas:

            table_id = schema.get(
                "table_id"
            ) or schema.get(
                "document_id"
            )

            columns = [
                column
                for column in schema.get(
                    "columns",
                    [],
                )
                if not column.get(
                    "internal",
                    False,
                )
            ]

            if not columns:
                continue

            score = 0.0

            # ----------------------------------------------------------
            # Column semantic compatibility
            # ----------------------------------------------------------

            numeric_columns = [
                column
                for column in columns
                if self._is_numeric_column(
                    column
                )
            ]

            dimension_columns = [
                column
                for column in columns
                if self._is_dimension_column(
                    column
                )
            ]

            time_columns = [
                column
                for column in columns
                if self._is_time_column(
                    column
                )
            ]

            if intent.operation in NUMERIC_OPERATIONS:

                if numeric_columns:
                    score += 0.25

                else:
                    score -= 0.50

            if intent.operation in GROUP_OPERATIONS:

                if (
                    numeric_columns
                    and dimension_columns
                ):
                    score += 0.35

                elif dimension_columns:
                    score += 0.10

                else:
                    score -= 0.25

            if intent.operation == "correlation":

                if len(numeric_columns) >= 2:
                    score += 0.50
                else:
                    score -= 0.75

            if intent.temporal_requested:

                if time_columns:
                    score += 0.25

            # ----------------------------------------------------------
            # Question → column relevance
            # ----------------------------------------------------------

            column_names = [
                str(
                    column.get(
                        "name",
                        "",
                    )
                )
                for column in columns
            ]

            best_column_score = 0.0

            for name in column_names:

                candidate_score = (
                    self._column_question_score(
                        question_normalized,
                        name,
                    )
                )

                best_column_score = max(
                    best_column_score,
                    candidate_score,
                )

            score += (
                best_column_score * 0.40
            )

            # ----------------------------------------------------------
            # Table name / sheet name relevance
            # ----------------------------------------------------------

            table_text = " ".join(
                [
                    str(
                        schema.get(
                            "document_name",
                            "",
                        )
                    ),
                    str(
                        schema.get(
                            "sheet_name",
                            "",
                        )
                    ),
                    str(
                        table_id or ""
                    ),
                ]
            )

            table_score = (
                self._text_overlap_score(
                    question_normalized,
                    table_text,
                )
            )

            score += (
                table_score * 0.20
            )

            # ----------------------------------------------------------
            # Row count is only a tiny tie breaker.
            # ----------------------------------------------------------

            row_count = int(
                schema.get(
                    "row_count",
                    0,
                )
                or 0
            )

            if row_count > 0:
                score += min(
                    0.05,
                    row_count / 100000.0,
                )

            ranked.append(
                (
                    schema,
                    score,
                )
            )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked

    # ==================================================================
    # DETERMINISTIC PLAN
    # ==================================================================

    def _build_deterministic_plan(
        self,
        question: str,
        intent: QueryIntent,
        table_candidates: List[
            Tuple[dict, float]
        ],
    ) -> Optional[QueryPlan]:

        best_schema = table_candidates[0][0]

        table_score = table_candidates[0][1]

        second_score = (
            table_candidates[1][1]
            if len(table_candidates) > 1
            else None
        )

        # If two tables are almost equally good, don't guess.
        if (
            second_score is not None
            and abs(
                table_score
                - second_score
            )
            < self.ambiguity_margin
        ):

            return QueryPlan(
                mode="structured",
                operation=intent.operation,
                confidence=0.0,
                reason=(
                    "ambiguous table selection"
                ),
                valid=False,
                intent=intent.intent,
                validation_errors=[
                    "Multiple tables are similarly "
                    "compatible with the question."
                ],
            )

        columns = [
            column
            for column in best_schema.get(
                "columns",
                [],
            )
            if not column.get(
                "internal",
                False,
            )
        ]

        # --------------------------------------------------------------
        # Resolve operation-specific columns
        # --------------------------------------------------------------

        target_column = None
        target_columns = []
        group_by = []

        operation = intent.operation

        if operation == "correlation":

            target_columns = (
                self._resolve_correlation_columns(
                    question=question,
                    columns=columns,
                )
            )

            if len(target_columns) != 2:

                return QueryPlan(
                    mode="structured",
                    operation=operation,
                    table_id=(
                        best_schema.get(
                            "table_id"
                        )
                        or best_schema.get(
                            "document_id"
                        )
                    ),
                    confidence=0.0,
                    reason=(
                        "correlation requires "
                        "two resolvable numeric columns"
                    ),
                    valid=False,
                    intent=intent.intent,
                    validation_errors=[
                        "Could not safely resolve "
                        "two numeric columns for correlation."
                    ],
                )

        elif operation in GROUP_OPERATIONS:

            target_column = (
                self._resolve_target_column(
                    question=question,
                    columns=columns,
                    require_numeric=(
                        operation
                        != "group_count"
                    ),
                )
            )

            group_by = (
                self._resolve_group_columns(
                    question=question,
                    columns=columns,
                )
            )

            if not group_by:

                return QueryPlan(
                    mode="structured",
                    operation=operation,
                    table_id=(
                        best_schema.get(
                            "table_id"
                        )
                        or best_schema.get(
                            "document_id"
                        )
                    ),
                    target_column=target_column,
                    confidence=0.0,
                    reason=(
                        "grouped operation requires "
                        "a grouping column"
                    ),
                    valid=False,
                    intent=intent.intent,
                    validation_errors=[
                        "Could not resolve a grouping column."
                    ],
                )

            if operation != "group_count" and not target_column:

                return QueryPlan(
                    mode="structured",
                    operation=operation,
                    table_id=(
                        best_schema.get(
                            "table_id"
                        )
                        or best_schema.get(
                            "document_id"
                        )
                    ),
                    group_by=group_by,
                    confidence=0.0,
                    reason=(
                        "grouped numeric operation "
                        "requires a numeric target column"
                    ),
                    valid=False,
                    intent=intent.intent,
                    validation_errors=[
                        "Could not resolve a numeric "
                        "target column."
                    ],
                )

        elif operation in {
            "top_n",
            "bottom_n",
        }:

            target_column = (
                self._resolve_target_column(
                    question=question,
                    columns=columns,
                    require_numeric=True,
                )
            )

            group_by = (
                self._resolve_group_columns(
                    question=question,
                    columns=columns,
                )
            )

            if not target_column:

                return QueryPlan(
                    mode="structured",
                    operation=operation,
                    table_id=(
                        best_schema.get(
                            "table_id"
                        )
                        or best_schema.get(
                            "document_id"
                        )
                    ),
                    confidence=0.0,
                    reason=(
                        "ranking requires "
                        "a numeric target column"
                    ),
                    valid=False,
                    intent=intent.intent,
                    validation_errors=[
                        "Could not resolve ranking metric."
                    ],
                )

            # A ranking question normally has something
            # being ranked. If no explicit grouping field
            # is found, executor can rank rows.
            if not group_by:

                group_by = (
                    self._infer_ranking_dimension(
                        question=question,
                        columns=columns,
                        target_column=target_column,
                    )
                )

        else:

            target_column = (
                self._resolve_target_column(
                    question=question,
                    columns=columns,
                    require_numeric=(
                        operation
                        in NUMERIC_OPERATIONS
                    ),
                )
            )

            # count does not necessarily need a target.
            if (
                operation != "count"
                and not target_column
            ):

                return QueryPlan(
                    mode="structured",
                    operation=operation,
                    table_id=(
                        best_schema.get(
                            "table_id"
                        )
                        or best_schema.get(
                            "document_id"
                        )
                    ),
                    confidence=0.0,
                    reason=(
                        "target column could not "
                        "be resolved safely"
                    ),
                    valid=False,
                    intent=intent.intent,
                    validation_errors=[
                        "Could not resolve a target column."
                    ],
                )

        # --------------------------------------------------------------
        # Resolve filters
        # --------------------------------------------------------------

        filters = (
            self._extract_filters(
                question=question,
                columns=columns,
            )
        )

        # --------------------------------------------------------------
        # Ranking metadata
        # --------------------------------------------------------------

        limit = intent.limit

        sort_direction = None

        if operation == "top_n":
            sort_direction = "desc"

        elif operation == "bottom_n":
            sort_direction = "asc"

        # --------------------------------------------------------------
        # Build plan
        # --------------------------------------------------------------

        table_id = (
            best_schema.get(
                "table_id"
            )
            or best_schema.get(
                "document_id"
            )
        )

        plan = QueryPlan(
            mode="structured",
            operation=operation,
            table_id=table_id,
            target_column=target_column,
            target_columns=target_columns,
            group_by=group_by,
            filters=filters,
            sort_column=target_column,
            sort_direction=sort_direction,
            limit=limit,
            confidence=min(
                0.99,
                max(
                    0.70,
                    intent.confidence
                    * 0.85
                    + min(
                        0.15,
                        max(
                            0.0,
                            table_score,
                        ),
                    ),
                ),
            ),
            reason=(
                "deterministic structured plan"
            ),
            valid=True,
            intent=intent.intent,
        )

        # --------------------------------------------------------------
        # Strict final validation
        # --------------------------------------------------------------

        return self._validate_plan(
            plan=plan,
            schemas=[best_schema],
        )

    # ==================================================================
    # TARGET COLUMN
    # ==================================================================

    def _resolve_target_column(
        self,
        question: str,
        columns: List[dict],
        require_numeric: bool,
    ) -> Optional[str]:

        candidates = []

        for column in columns:

            name = str(
                column.get(
                    "name",
                    "",
                )
            )

            if not name:
                continue

            if require_numeric and not (
                self._is_numeric_column(
                    column
                )
            ):
                continue

            score = (
                self._column_question_score(
                    normalize_text(question),
                    name,
                )
            )

            # Measure columns receive a small boost
            # for numerical operations.
            if require_numeric and (
                column.get(
                    "semantic_type"
                )
                in {
                    "integer",
                    "numeric",
                }
                or column.get(
                    "role"
                )
                == "measure"
            ):
                score += 0.05

            candidates.append(
                (
                    name,
                    min(
                        1.0,
                        score,
                    ),
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        best_name, best_score = (
            candidates[0]
        )

        second_score = (
            candidates[1][1]
            if len(candidates) > 1
            else 0.0
        )

        # Don't guess between equally plausible measures.
        if best_score < self.column_min_score:
            return None

        if (
            second_score > 0
            and (
                best_score
                - second_score
            )
            < self.ambiguity_margin
        ):
            return None

        return best_name

    # ==================================================================
    # CORRELATION
    # ==================================================================

    def _resolve_correlation_columns(
        self,
        question: str,
        columns: List[dict],
    ) -> List[str]:

        numeric_columns = [
            column
            for column in columns
            if self._is_numeric_column(
                column
            )
        ]

        if len(numeric_columns) < 2:
            return []

        question_normalized = normalize_text(
            question
        )

        # --------------------------------------------------------------
        # Try to identify "between X and Y"
        # --------------------------------------------------------------

        # Stop the second column at a following "and", comma, or
        # end of question so extra clauses are not part of Y.
        # Example: "between Quantity sold and Revenue, and is..."
        between_match = re.search(
            r"\bbetween\s+(.+?)\s+\band\b\s+(.+?)(?=\s+and\b|\s*,|\s*\?|$)",
            question_normalized,
            flags=re.IGNORECASE,
        )

        if between_match:

            first_text = (
                between_match.group(1)
                .strip()
            )

            second_text = (
                between_match.group(2)
                .strip()
            )

            first = self._best_numeric_column(
                first_text,
                numeric_columns,
            )

            second = self._best_numeric_column(
                second_text,
                numeric_columns,
            )

            if first and second and first != second:
                return [
                    first,
                    second,
                ]

        # --------------------------------------------------------------
        # Try "X and Y"
        # --------------------------------------------------------------

        and_parts = re.findall(
            r"\b([a-z][a-z0-9 _-]{1,60})\b",
            question_normalized,
        )

        resolved = []

        for column in numeric_columns:

            name = str(
                column.get(
                    "name",
                    "",
                )
            )

            score = self._column_question_score(
                question_normalized,
                name,
            )

            if score >= self.column_min_score:
                resolved.append(
                    (
                        name,
                        score,
                    )
                )

        resolved.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # Require two sufficiently strong and
        # sufficiently distinct matches.
        if len(resolved) >= 2:

            first_name, first_score = (
                resolved[0]
            )

            second_name, second_score = (
                resolved[1]
            )

            if (
                first_score
                >= self.column_min_score
                and second_score
                >= self.column_min_score
                and first_name != second_name
            ):
                return [
                    first_name,
                    second_name,
                ]

        return []

    def _best_numeric_column(
        self,
        query: str,
        columns: List[dict],
    ) -> Optional[str]:

        best = None
        best_score = 0.0

        for column in columns:

            name = str(
                column.get(
                    "name",
                    "",
                )
            )

            score = self._column_question_score(
                normalize_text(query),
                name,
            )

            if score > best_score:

                best = name
                best_score = score

        if (
            best is None
            or best_score < self.column_min_score
        ):
            return None

        return best

    # ==================================================================
    # GROUP BY
    # ==================================================================

    def _resolve_group_columns(
        self,
        question: str,
        columns: List[dict],
    ) -> List[str]:

        normalized = normalize_text(
            question
        )

        # First try explicit "by X".
        explicit_matches = re.findall(
            r"\bby\s+([a-z0-9][a-z0-9 _-]{1,60})",
            normalized,
        )

        candidates = []

        for column in columns:

            if not (
                self._is_dimension_column(
                    column
                )
                or self._is_time_column(
                    column
                )
            ):
                continue

            name = str(
                column.get(
                    "name",
                    "",
                )
            )

            score = self._column_question_score(
                normalized,
                name,
            )

            for explicit in explicit_matches:

                explicit_score = (
                    self._text_similarity(
                        explicit,
                        name,
                    )
                )

                score = max(
                    score,
                    explicit_score + 0.15,
                )

            candidates.append(
                (
                    name,
                    min(
                        1.0,
                        score,
                    ),
                )
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        if not candidates:
            return []

        result = []

        for name, score in candidates:

            if score < self.column_min_score:
                continue

            if name not in result:
                result.append(name)

            # Avoid over-grouping unless the question
            # explicitly asks for multiple dimensions.
            if len(result) >= 2:
                break

        return result

    # ==================================================================
    # RANKING DIMENSION
    # ==================================================================

    def _infer_ranking_dimension(
        self,
        question: str,
        columns: List[dict],
        target_column: str,
    ) -> List[str]:

        candidates = []

        for column in columns:

            name = str(
                column.get(
                    "name",
                    "",
                )
            )

            if name == target_column:
                continue

            if not (
                self._is_dimension_column(
                    column
                )
                or self._is_time_column(
                    column
                )
            ):
                continue

            score = self._column_question_score(
                normalize_text(question),
                name,
            )

            candidates.append(
                (
                    name,
                    score,
                )
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        if (
            candidates
            and candidates[0][1]
            >= self.column_min_score
        ):
            return [
                candidates[0][0]
            ]

        return []

    # ==================================================================
    # FILTERS
    # ==================================================================

    def _extract_filters(
        self,
        question: str,
        columns: List[dict],
    ) -> List[QueryFilter]:

        filters = []

        normalized = normalize_text(
            question
        )

        # --------------------------------------------------------------
        # Numeric comparison filters
        # --------------------------------------------------------------

        numeric_patterns = [
            (
                "gte",
                r"([a-z0-9 _-]+?)\s*(?:>=|at least|greater than or equal to)\s*(-?\d+(?:\.\d+)?)",
            ),
            (
                "lte",
                r"([a-z0-9 _-]+?)\s*(?:<=|at most|less than or equal to)\s*(-?\d+(?:\.\d+)?)",
            ),
            (
                "gt",
                r"([a-z0-9 _-]+?)\s*(?:>|greater than|more than|above)\s*(-?\d+(?:\.\d+)?)",
            ),
            (
                "lt",
                r"([a-z0-9 _-]+?)\s*(?:<|less than|below|under)\s*(-?\d+(?:\.\d+)?)",
            ),
        ]

        for op, pattern in numeric_patterns:

            for match in re.finditer(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):

                column_text = (
                    match.group(1)
                    .strip()
                )

                raw_value = (
                    match.group(2)
                    .strip()
                )

                column = self._resolve_column_from_text(
                    column_text,
                    columns,
                )

                if not column:
                    continue

                filters.append(
                    QueryFilter(
                        column=column,
                        op=op,
                        value=self._parse_numeric(
                            raw_value
                        ),
                        score=1.0,
                        validated=True,
                        requested_value=raw_value,
                    )
                )

        # --------------------------------------------------------------
        # Equality filters:
        #
        # Examples:
        #   for North
        #   in India
        #   where category is Electronics
        #   region = North
        # --------------------------------------------------------------

        equality_patterns = [
            r"\bwhere\s+(.+?)\s*(?:is|=|equals?)\s+([a-z0-9 _./%-]+)",
            r"\b([a-z][a-z0-9 _-]{1,40})\s*=\s*([a-z0-9 _./%-]+)",
            r"\bfor\s+([a-z][a-z0-9 _-]{1,40})\s+in\s+([a-z0-9 _./%-]+)",
            r"\bin\s+([a-z][a-z0-9 _-]{1,40})\s+([a-z0-9 _./%-]+)",
        ]

        for pattern in equality_patterns:

            for match in re.finditer(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):

                column_text = (
                    match.group(1)
                    .strip()
                )

                value_text = (
                    match.group(2)
                    .strip()
                )

                column = (
                    self._resolve_column_from_text(
                        column_text,
                        columns,
                    )
                )

                if not column:
                    continue

                # Resolve the actual value only if
                # schema sample values support it.
                value_result = (
                    self._resolve_filter_value(
                        column=column,
                        requested_value=value_text,
                        columns=columns,
                    )
                )

                if value_result is None:
                    # IMPORTANT:
                    # Do not silently create a filter from
                    # an unknown value.
                    continue

                matched_value, score = (
                    value_result
                )

                filters.append(
                    QueryFilter(
                        column=column,
                        op="eq",
                        value=matched_value,
                        score=score,
                        validated=(
                            score
                            >= self.value_min_score
                        ),
                        requested_value=value_text,
                    )
                )

        return self._dedupe_filters(
            filters
        )

    def _resolve_filter_value(
        self,
        column: str,
        requested_value: str,
        columns: List[dict],
    ) -> Optional[
        Tuple[Any, float]
    ]:

        schema_column = None

        for item in columns:

            if str(
                item.get(
                    "name",
                    "",
                )
            ) == column:
                schema_column = item
                break

        if schema_column is None:
            return None

        samples = schema_column.get(
            "sample_values",
            [],
        )

        if not samples:
            return None

        result = best_value_match(
            requested_value,
            samples,
            min_score=self.value_min_score,
        )

        if result is None:
            return None

        return result

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_plan(
        self,
        plan: QueryPlan,
        schemas: List[dict],
    ) -> QueryPlan:

        errors = []

        if plan.operation not in ALLOWED_OPERATIONS:

            errors.append(
                f"Unsupported operation: "
                f"{plan.operation}"
            )

        schema = self._find_schema(
            plan.table_id,
            schemas,
        )

        if schema is None:

            errors.append(
                "Selected table does not exist."
            )

            plan.valid = False
            plan.validation_errors = errors
            return plan

        columns = {
            str(
                column.get(
                    "name",
                    "",
                )
            ): column
            for column in schema.get(
                "columns",
                []
            )
            if not column.get(
                "internal",
                False,
            )
        }

        # --------------------------------------------------------------
        # Target column
        # --------------------------------------------------------------

        if plan.target_column:

            if (
                plan.target_column
                not in columns
            ):

                errors.append(
                    f"Target column does not exist: "
                    f"{plan.target_column}"
                )

            elif (
                plan.operation
                in NUMERIC_OPERATIONS
                and not self._is_numeric_column(
                    columns[
                        plan.target_column
                    ]
                )
            ):

                errors.append(
                    f"Target column is not numeric: "
                    f"{plan.target_column}"
                )

        # --------------------------------------------------------------
        # Multiple target columns
        # --------------------------------------------------------------

        for column_name in (
            plan.target_columns
        ):

            if column_name not in columns:

                errors.append(
                    f"Target column does not exist: "
                    f"{column_name}"
                )

        if plan.operation == "correlation":

            if len(
                plan.target_columns
            ) != 2:

                errors.append(
                    "Correlation requires exactly "
                    "two target columns."
                )

            else:

                for column_name in (
                    plan.target_columns
                ):

                    if not self._is_numeric_column(
                        columns[
                            column_name
                        ]
                    ):

                        errors.append(
                            f"Correlation column "
                            f"is not numeric: "
                            f"{column_name}"
                        )

        # --------------------------------------------------------------
        # Group columns
        # --------------------------------------------------------------

        for group_column in (
            plan.group_by
        ):

            if group_column not in columns:

                errors.append(
                    f"Group-by column does not exist: "
                    f"{group_column}"
                )

        # --------------------------------------------------------------
        # Filters
        # --------------------------------------------------------------

        for query_filter in (
            plan.filters
        ):

            if (
                query_filter.column
                not in columns
            ):

                errors.append(
                    f"Filter column does not exist: "
                    f"{query_filter.column}"
                )

            if not query_filter.validated:

                errors.append(
                    f"Filter value was not safely "
                    f"validated for column "
                    f"{query_filter.column}: "
                    f"{query_filter.requested_value}"
                )

        # --------------------------------------------------------------
        # Operation-specific validation
        # --------------------------------------------------------------

        if plan.operation in GROUP_OPERATIONS:

            if not plan.group_by:

                errors.append(
                    "Grouped operation requires "
                    "at least one group-by column."
                )

            if (
                plan.operation
                != "group_count"
                and not plan.target_column
            ):

                errors.append(
                    "Grouped numeric operation requires "
                    "a target column."
                )

        if plan.operation in {
            "top_n",
            "bottom_n",
        }:

            if not plan.target_column:

                errors.append(
                    "Ranking operation requires "
                    "a target column."
                )

            if (
                plan.limit is None
                or plan.limit < 1
                or plan.limit > 1000
            ):

                errors.append(
                    "Ranking limit must be "
                    "between 1 and 1000."
                )

        # --------------------------------------------------------------
        # Final result
        # --------------------------------------------------------------

        plan.validation_errors = errors

        plan.valid = not errors

        if not plan.valid:
            plan.confidence = min(
                plan.confidence,
                0.30,
            )

        return plan

    # ==================================================================
    # LLM CANDIDATE
    # ==================================================================

    def _build_llm_candidate(
        self,
        question: str,
        intent: QueryIntent,
        schemas: List[dict],
        llm: Any,
    ) -> Optional[QueryPlan]:

        """
        Ask the LLM for a candidate plan.

        IMPORTANT:
        The output is NEVER trusted directly.

        It is parsed and then passed through _validate_plan().
        """

        schema_payload = []

        for schema in schemas:

            schema_payload.append(
                {
                    "table_id": (
                        schema.get(
                            "table_id"
                        )
                        or schema.get(
                            "document_id"
                        )
                    ),
                    "document_name": schema.get(
                        "document_name"
                    ),
                    "sheet_name": schema.get(
                        "sheet_name"
                    ),
                    "row_count": schema.get(
                        "row_count"
                    ),
                    "columns": [
                        {
                            "name": column.get(
                                "name"
                            ),
                            "semantic_type": column.get(
                                "semantic_type"
                            ),
                            "role": column.get(
                                "role"
                            ),
                            "sample_values": column.get(
                                "sample_values",
                                [],
                            )[:20],
                        }
                        for column in schema.get(
                            "columns",
                            []
                        )
                        if not column.get(
                            "internal",
                            False,
                        )
                    ],
                }
            )

        prompt = f"""
You are a structured-data query planner.

The user question is:

{question}

Deterministic intent:
{json.dumps(intent.to_dict(), indent=2)}

Available tables and schemas:

{json.dumps(schema_payload, indent=2, ensure_ascii=False)}

Return ONLY valid JSON.

Allowed operations:

{sorted(ALLOWED_OPERATIONS)}

JSON format:

{{
  "operation": "sum",
  "table_id": "exact_table_id",
  "target_column": "exact_column_name_or_null",
  "target_columns": [],
  "group_by": [],
  "filters": [],
  "limit": null,
  "sort_column": null,
  "sort_direction": null
}}

Filter format:

{{
  "column": "exact_column_name",
  "op": "eq|ne|gt|gte|lt|lte|contains",
  "value": "value"
}}

Rules:

1. Use ONLY table IDs and column names that exist in the supplied schema.
2. Never invent a column.
3. Never invent a table.
4. For correlation, target_columns MUST contain exactly two numeric columns.
5. For sum/avg/min/max/median/std/variance, target_column must be numeric.
6. For group operations, group_by must contain an actual dimension/time column.
7. Do not invent filter values.
8. If the question cannot be safely represented, return:
   {{
       "operation": null,
       "table_id": null,
       "target_column": null,
       "target_columns": [],
       "group_by": [],
       "filters": [],
       "limit": null,
       "sort_column": null,
       "sort_direction": null
   }}
"""

        try:

            response = self._call_llm(
                llm,
                prompt,
            )

            parsed = self._parse_json_response(
                response
            )

            if not parsed:
                return None

            operation = parsed.get(
                "operation"
            )

            if not operation:
                return None

            filters = []

            for item in (
                parsed.get(
                    "filters",
                    []
                )
                or []
            ):

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                column = item.get(
                    "column"
                )

                if not column:
                    continue

                filters.append(
                    QueryFilter(
                        column=str(
                            column
                        ),
                        op=str(
                            item.get(
                                "op",
                                "eq",
                            )
                        ),
                        value=item.get(
                            "value"
                        ),
                        score=0.0,
                        validated=False,
                        requested_value=item.get(
                            "value"
                        ),
                    )
                )

            return QueryPlan(
                mode="structured",
                operation=str(
                    operation
                ),
                table_id=(
                    str(
                        parsed.get(
                            "table_id"
                        )
                    )
                    if parsed.get(
                        "table_id"
                    )
                    else None
                ),
                target_column=(
                    str(
                        parsed.get(
                            "target_column"
                        )
                    )
                    if parsed.get(
                        "target_column"
                    )
                    else None
                ),
                target_columns=[
                    str(value)
                    for value in (
                        parsed.get(
                            "target_columns",
                            []
                        )
                        or []
                    )
                    if value
                ],
                group_by=[
                    str(value)
                    for value in (
                        parsed.get(
                            "group_by",
                            []
                        )
                        or []
                    )
                    if value
                ],
                filters=filters,
                limit=self._safe_int(
                    parsed.get(
                        "limit"
                    )
                ),
                sort_column=(
                    str(
                        parsed.get(
                            "sort_column"
                        )
                    )
                    if parsed.get(
                        "sort_column"
                    )
                    else None
                ),
                sort_direction=(
                    str(
                        parsed.get(
                            "sort_direction"
                        )
                    ).lower()
                    if parsed.get(
                        "sort_direction"
                    )
                    else None
                ),
                confidence=0.80,
                reason="llm candidate",
                valid=False,
                intent=intent.intent,
            )

        except Exception:

            logger.exception(
                "Failed to build LLM structured candidate"
            )

            return None

    # ==================================================================
    # LLM HELPERS
    # ==================================================================

    def _call_llm(
        self,
        llm: Any,
        prompt: str,
    ) -> str:

        if hasattr(
            llm,
            "invoke",
        ):

            response = llm.invoke(
                prompt
            )

            if hasattr(
                response,
                "content",
            ):
                return str(
                    response.content
                )

            return str(
                response
            )

        if callable(llm):

            response = llm(
                prompt
            )

            if hasattr(
                response,
                "content",
            ):
                return str(
                    response.content
                )

            return str(
                response
            )

        raise TypeError(
            "Unsupported LLM interface."
        )

    def _parse_json_response(
        self,
        response: str,
    ) -> Optional[dict]:

        text = str(
            response or ""
        ).strip()

        if not text:
            return None

        # Remove markdown code fences.
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        try:

            parsed = json.loads(
                text
            )

            return (
                parsed
                if isinstance(
                    parsed,
                    dict,
                )
                else None
            )

        except json.JSONDecodeError:

            # Attempt to recover the first JSON object.
            match = re.search(
                r"\{.*\}",
                text,
                flags=re.DOTALL,
            )

            if not match:
                return None

            try:

                parsed = json.loads(
                    match.group(0)
                )

                return (
                    parsed
                    if isinstance(
                        parsed,
                        dict,
                    )
                    else None
                )

            except json.JSONDecodeError:
                return None

    # ==================================================================
    # COLUMN MATCHING
    # ==================================================================

    def _resolve_column_from_text(
        self,
        query: str,
        columns: List[dict],
    ) -> Optional[str]:

        candidates = []

        for column in columns:

            name = str(
                column.get(
                    "name",
                    "",
                )
            )

            score = self._text_similarity(
                query,
                name,
            )

            if score >= self.column_min_score:
                candidates.append(
                    (
                        name,
                        score,
                    )
                )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        if not candidates:
            return None

        best_name, best_score = (
            candidates[0]
        )

        second_score = (
            candidates[1][1]
            if len(candidates) > 1
            else 0.0
        )

        if (
            second_score > 0
            and best_score
            - second_score
            < self.ambiguity_margin
        ):
            return None

        return best_name

    def _column_question_score(
        self,
        question: str,
        column_name: str,
    ) -> float:

        q = normalize_text(
            question
        )

        c = normalize_text(
            column_name
        )

        if not q or not c:
            return 0.0

        # Exact column phrase.
        if c in q:
            return 1.0

        # Strong word overlap.
        q_tokens = set(
            q.split()
        )

        c_tokens = set(
            c.split()
        )

        if not c_tokens:
            return 0.0

        overlap = (
            len(
                q_tokens
                & c_tokens
            )
            / len(c_tokens)
        )

        fuzzy = self._text_similarity(
            q,
            c,
        )

        return max(
            overlap,
            fuzzy,
        )

    def _text_similarity(
        self,
        first: str,
        second: str,
    ) -> float:

        result = best_column_match(
            first,
            [second],
            min_score=0.0,
        )

        if result is None:
            return 0.0

        return float(
            result[1]
        )

    # ==================================================================
    # FILTER HELPERS
    # ==================================================================

    def _dedupe_filters(
        self,
        filters: List[QueryFilter],
    ) -> List[QueryFilter]:

        result = []

        seen = set()

        for item in filters:

            key = (
                item.column,
                item.op,
                normalize_text(
                    str(
                        item.value
                    )
                ),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    # ==================================================================
    # SCHEMA HELPERS
    # ==================================================================

    def _find_schema(
        self,
        table_id: Optional[str],
        schemas: List[dict],
    ) -> Optional[dict]:

        if not table_id:
            return None

        for schema in schemas:

            candidate = (
                schema.get(
                    "table_id"
                )
                or schema.get(
                    "document_id"
                )
            )

            if candidate == table_id:
                return schema

        return None

    def _is_numeric_column(
        self,
        column: dict,
    ) -> bool:

        semantic_type = column.get(
            "semantic_type"
        )

        if semantic_type in {
            "integer",
            "numeric",
        }:
            return True

        role = column.get(
            "role"
        )

        dtype = str(
            column.get(
                "dtype",
                ""
            )
        ).casefold()

        numeric_dtype = any(
            token in dtype
            for token in (
                "int",
                "float",
                "double",
                "decimal",
                "number",
            )
        )

        if role == "measure":
            return numeric_dtype

        # Integer/float columns named like measures (Quantity,
        # Revenue) must stay usable even if schema typed them
        # as identifier because values were unique.
        if semantic_type == "identifier" and numeric_dtype:

            name_tokens = set(
                normalize_text(
                    column.get(
                        "name",
                        "",
                    )
                ).split()
            )

            id_name_tokens = {
                "id",
                "code",
                "key",
                "identifier",
                "sku",
                "isin",
                "cusip",
                "uuid",
            }

            if name_tokens & id_name_tokens:
                return False

            return True

        return False

    def _is_dimension_column(
        self,
        column: dict,
    ) -> bool:

        return (
            column.get(
                "semantic_type"
            )
            in {
                "categorical",
                "text",
            }
            or column.get(
                "role"
            )
            == "dimension"
        )

    def _is_time_column(
        self,
        column: dict,
    ) -> bool:

        return (
            column.get(
                "semantic_type"
            )
            in {
                "date",
                "datetime",
            }
            or column.get(
                "role"
            )
            == "time"
        )

    # ==================================================================
    # GENERAL SCORING
    # ==================================================================

    def _text_overlap_score(
        self,
        question: str,
        candidate: str,
    ) -> float:

        q_tokens = set(
            normalize_text(
                question
            ).split()
        )

        c_tokens = set(
            normalize_text(
                candidate
            ).split()
        )

        if not q_tokens or not c_tokens:
            return 0.0

        return (
            len(
                q_tokens
                & c_tokens
            )
            / len(c_tokens)
        )

    # ==================================================================
    # NUMERIC PARSING
    # ==================================================================

    def _parse_numeric(
        self,
        value: str,
    ) -> Any:

        text = str(
            value
        ).strip()

        try:

            number = float(
                text
            )

            if number.is_integer():
                return int(
                    number
                )

            return number

        except (
            TypeError,
            ValueError,
        ):
            return value

    @staticmethod
    def _safe_int(
        value: Any,
    ) -> Optional[int]:

        if value is None:
            return None

        try:

            number = int(
                float(
                    value
                )
            )

            if number < 1:
                return None

            return number

        except (
            TypeError,
            ValueError,
        ):
            return None