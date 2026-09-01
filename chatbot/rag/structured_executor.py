"""
Production-safe structured query executor.

The executor is the source of truth for structured data.

Rules
-----
1. Never use an LLM to calculate an answer.
2. Never generate SQL.
3. Never silently ignore an invalid filter.
4. Validate columns before execution.
5. Validate numeric columns before numeric operations.
6. Use pandas for deterministic calculations.
7. Return structured metadata together with the answer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import MAX_VALUE_MATCH_CANDIDATES
from rag.query_planner import NUMERIC_OPERATIONS, QueryFilter, QueryPlan
from rag.table_store import INTERNAL_COLUMNS, TableStore
from rag.text_normalize import (
    best_value_match,
    normalize_text,
)
from debug_trace import dbg

logger = logging.getLogger(__name__)


# ======================================================================
# RESULT
# ======================================================================


@dataclass
class StructuredResult:
    matched: bool

    answer: str = ""

    value: Any = None

    operation: str = ""

    row_count: int = 0

    filters: List[Dict[str, Any]] = field(
        default_factory=list
    )

    sources: List[Dict[str, Any]] = field(
        default_factory=list
    )

    table_id: str = ""

    document_name: str = ""

    # New structured result metadata.
    columns: List[str] = field(
        default_factory=list
    )

    group_by: List[str] = field(
        default_factory=list
    )

    result_rows: List[Dict[str, Any]] = field(
        default_factory=list
    )

    confidence: float = 0.0

    error: Optional[str] = None


# ======================================================================
# EXECUTOR
# ======================================================================


class StructuredExecutor:

    def __init__(
        self,
        table_store: Optional[TableStore] = None,
    ):
        self.table_store = (
            table_store or TableStore()
        )

    # ==================================================================
    # PUBLIC
    # ==================================================================

    def execute(
        self,
        plan: QueryPlan,
        schemas: List[dict],
    ) -> StructuredResult:

        dbg(
            "EXECUTOR_START",
            plan=plan.to_dict()
            if hasattr(plan, "to_dict")
            else str(plan),
        )

        # --------------------------------------------------------------
        # Never execute an invalid plan.
        # --------------------------------------------------------------

        if not plan.valid:

            return StructuredResult(
                matched=False,
                operation=(
                    plan.operation
                    or ""
                ),
                error=(
                    "; ".join(
                        plan.validation_errors
                    )
                    or "Invalid structured query plan."
                ),
            )

        if plan.mode != "structured":

            return StructuredResult(
                matched=False
            )

        if not plan.operation:

            return StructuredResult(
                matched=False,
                error=(
                    "Structured operation is missing."
                ),
            )

        # --------------------------------------------------------------
        # Select exact table.
        # --------------------------------------------------------------

        tables = self._select_tables(
            plan,
            schemas,
        )

        if not tables:

            return StructuredResult(
                matched=False,
                operation=plan.operation,
                error=(
                    "The requested structured data "
                    "could not be found."
                ),
            )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # If the planner selected a specific table, execute only
        # against that table.
        #
        # Do NOT aggregate multiple matching tables together.
        # Otherwise duplicate CSV/XLSX copies can double totals.
        # --------------------------------------------------------------

        if plan.table_id:

            schema, frame = tables[0]

            return self._execute_on_table(
                plan,
                schema,
                frame,
            )

        # --------------------------------------------------------------
        # No explicit table.
        #
        # Execute candidates independently and only accept an
        # unambiguous result.
        # --------------------------------------------------------------

        results = []

        for schema, frame in tables:

            result = self._execute_on_table(
                plan,
                schema,
                frame,
            )

            if result.matched:
                results.append(
                    result
                )

        if not results:

            return StructuredResult(
                matched=False,
                operation=plan.operation,
                error=(
                    "No table could safely answer "
                    "the structured question."
                ),
            )

        if len(results) == 1:
            return results[0]

        # Multiple valid tables are dangerous.
        # Do not choose based on row count.
        #
        # Your previous implementation did:
        #
        #     if result.row_count > best_result.row_count:
        #         best_result = result
        #
        # That can return a mathematically correct answer from
        # the WRONG table.
        #
        return StructuredResult(
            matched=False,
            operation=plan.operation,
            error=(
                "Multiple structured tables could answer "
                "the question and the result is ambiguous."
            ),
        )

    # ==================================================================
    # TABLE SELECTION
    # ==================================================================

    def _select_tables(
        self,
        plan: QueryPlan,
        schemas: List[dict],
    ) -> List[
        Tuple[dict, pd.DataFrame]
    ]:

        selected = []

        needed = set(
            item.column
            for item in plan.filters
        )

        if plan.target_column:
            needed.add(
                plan.target_column
            )

        for column in (
            plan.target_columns
        ):
            needed.add(column)

        for column in (
            plan.group_by
        ):
            needed.add(column)

        if plan.sort_column:
            needed.add(
                plan.sort_column
            )

        for schema in schemas:

            document_id = (
                schema.get(
                    "table_id"
                )
                or schema.get(
                    "document_id"
                )
            )

            if (
                plan.table_id
                and document_id
                != plan.table_id
            ):
                continue

            frame = (
                self.table_store.load_dataframe(
                    document_id
                )
            )

            if frame.empty:
                continue

            actual_columns = set(
                frame.columns
            )

            if needed and not needed.issubset(
                actual_columns
            ):
                continue

            selected.append(
                (
                    schema,
                    frame,
                )
            )

        return selected

    # ==================================================================
    # TABLE EXECUTION
    # ==================================================================

    def _execute_on_table(
        self,
        plan: QueryPlan,
        schema: dict,
        frame: pd.DataFrame,
    ) -> StructuredResult:

        # --------------------------------------------------------------
        # Validate actual DataFrame one more time.
        #
        # The planner validates against schema.json, but the executor
        # must validate against the actual data.
        # --------------------------------------------------------------

        validation_error = (
            self._validate_actual_frame(
                plan,
                frame,
            )
        )

        if validation_error:

            return StructuredResult(
                matched=False,
                operation=plan.operation,
                table_id=self._table_id(
                    schema
                ),
                document_name=schema.get(
                    "document_name",
                    "",
                ),
                error=validation_error,
            )

        filtered = frame.copy()

        applied = []

        # --------------------------------------------------------------
        # Apply filters.
        # --------------------------------------------------------------

        for query_filter in (
            plan.filters
        ):

            filtered, applied_filter = (
                self._apply_filter(
                    filtered,
                    query_filter,
                )
            )

            applied.append(
                applied_filter
            )

            # IMPORTANT:
            # If a requested filter cannot be validated,
            # don't silently continue with unfiltered data.
            if not applied_filter.get(
                "matched",
                False,
            ):

                return StructuredResult(
                    matched=False,
                    operation=plan.operation,
                    table_id=self._table_id(
                        schema
                    ),
                    document_name=schema.get(
                        "document_name",
                        "",
                    ),
                    filters=applied,
                    error=(
                        "A requested filter could not "
                        "be safely applied."
                    ),
                )

        dbg(
            "EXECUTOR_FILTERED",
            document_name=schema.get(
                "document_name"
            ),
            document_id=self._table_id(
                schema
            ),
            rows_before=len(frame),
            rows_after=len(filtered),
            operation=plan.operation,
            target_column=plan.target_column,
            target_columns=plan.target_columns,
            group_by=plan.group_by,
            applied=applied,
            sample_after=(
                filtered.head(3)
                .astype(str)
                .to_dict(
                    orient="records"
                )
            ),
        )

        # --------------------------------------------------------------
        # Execute operation.
        # --------------------------------------------------------------

        try:

            result = self._execute_operation(
                filtered,
                plan,
            )

        except Exception as exc:

            logger.exception(
                "Structured execution failed | "
                "operation=%s | table=%s",
                plan.operation,
                self._table_id(schema),
            )

            return StructuredResult(
                matched=False,
                operation=plan.operation,
                table_id=self._table_id(
                    schema
                ),
                document_name=schema.get(
                    "document_name",
                    "",
                ),
                filters=applied,
                error=(
                    f"Structured calculation failed: "
                    f"{exc}"
                ),
            )

        dbg(
            "EXECUTOR_VALUE",
            document_name=schema.get(
                "document_name"
            ),
            value=result.get(
                "value"
            ),
            row_count=int(
                len(filtered)
            ),
        )

        sources = self._sample_sources(
            filtered,
            schema,
        )

        return StructuredResult(
            matched=True,
            answer=result.get(
                "answer",
                "",
            ),
            value=result.get(
                "value"
            ),
            operation=plan.operation,
            row_count=int(
                len(filtered)
            ),
            filters=applied,
            sources=sources,
            table_id=self._table_id(
                schema
            ),
            document_name=schema.get(
                "document_name",
                "",
            ),
            columns=result.get(
                "columns",
                []
            ),
            group_by=plan.group_by,
            result_rows=result.get(
                "result_rows",
                []
            ),
            confidence=min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            plan,
                            "confidence",
                            0.0,
                        )
                        or 0.0
                    ),
                ),
            ),
        )

    # ==================================================================
    # OPERATION DISPATCH
    # ==================================================================

    def _execute_operation(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        operation = plan.operation

        if operation == "count":
            return self._execute_count(
                frame
            )

        if operation == "sum":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "sum",
            )

        if operation == "avg":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "avg",
            )

        if operation == "min":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "min",
            )

        if operation == "max":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "max",
            )

        if operation == "distinct_count":
            return self._execute_distinct_count(
                frame,
                plan.target_column,
            )

        if operation == "median":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "median",
            )

        if operation == "std":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "std",
            )

        if operation == "variance":
            return self._execute_single_numeric(
                frame,
                plan.target_column,
                "variance",
            )

        if operation == "correlation":
            return self._execute_correlation(
                frame,
                plan,
            )

        if operation in {
            "group_count",
            "group_sum",
            "group_avg",
            "group_min",
            "group_max",
        }:
            return self._execute_grouped(
                frame,
                plan,
            )

        if operation in {
            "top_n",
            "bottom_n",
        }:
            return self._execute_ranking(
                frame,
                plan,
            )

        if operation == "percentage":
            return self._execute_percentage(
                frame,
                plan,
            )

        if operation == "ratio":
            return self._execute_ratio(
                frame,
                plan,
            )

        if operation == "compare":
            return self._execute_compare(
                frame,
                plan,
            )

        if operation == "trend":
            return self._execute_trend(
                frame,
                plan,
            )

        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    # ==================================================================
    # BASIC AGGREGATIONS
    # ==================================================================

    def _execute_count(
        self,
        frame: pd.DataFrame,
    ) -> Dict[str, Any]:

        value = int(
            len(frame)
        )

        return {
            "value": value,
            "answer": str(
                value
            ),
            "columns": [],
            "result_rows": [],
        }

    def _execute_single_numeric(
        self,
        frame: pd.DataFrame,
        column: Optional[str],
        operation: str,
    ) -> Dict[str, Any]:

        if not column:
            raise ValueError(
                "Numeric target column is missing."
            )

        if column not in frame.columns:
            raise ValueError(
                f"Column does not exist: {column}"
            )

        numeric = self._numeric_series(
            frame[column],
            column,
        )

        if numeric.empty:
            raise ValueError(
                f"Column '{column}' contains "
                "no usable numeric values."
            )

        if operation == "sum":
            value = numeric.sum()

        elif operation == "avg":
            value = numeric.mean()

        elif operation == "min":
            value = numeric.min()

        elif operation == "max":
            value = numeric.max()

        elif operation == "median":
            value = numeric.median()

        elif operation == "std":
            value = numeric.std(
                ddof=1
            )

        elif operation == "variance":
            value = numeric.var(
                ddof=1
            )

        else:
            raise ValueError(
                f"Unsupported numeric operation: "
                f"{operation}"
            )

        value = self._safe_number(
            value
        )

        return {
            "value": value,
            "answer": self._format_number(
                value
            ),
            "columns": [
                column
            ],
            "result_rows": [],
        }

    def _execute_distinct_count(
        self,
        frame: pd.DataFrame,
        column: Optional[str],
    ) -> Dict[str, Any]:

        if not column:
            raise ValueError(
                "Distinct-count target column is missing."
            )

        if column not in frame.columns:
            raise ValueError(
                f"Column does not exist: {column}"
            )

        value = int(
            frame[column].nunique(
                dropna=True
            )
        )

        return {
            "value": value,
            "answer": str(
                value
            ),
            "columns": [
                column
            ],
            "result_rows": [],
        }

    # ==================================================================
    # CORRELATION
    # ==================================================================

    def _execute_correlation(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        columns = list(
            plan.target_columns
        )

        if len(columns) != 2:
            raise ValueError(
                "Correlation requires exactly "
                "two columns."
            )

        first, second = columns

        if first not in frame.columns:
            raise ValueError(
                f"Column does not exist: {first}"
            )

        if second not in frame.columns:
            raise ValueError(
                f"Column does not exist: {second}"
            )

        first_numeric = pd.to_numeric(
            frame[first],
            errors="coerce",
        )

        second_numeric = pd.to_numeric(
            frame[second],
            errors="coerce",
        )

        pair = pd.DataFrame(
            {
                first: first_numeric,
                second: second_numeric,
            }
        ).dropna()

        if len(pair) < 2:
            raise ValueError(
                "Not enough paired numeric observations "
                "to calculate correlation."
            )

        if (
            pair[first].nunique()
            < 2
            or pair[second].nunique()
            < 2
        ):
            raise ValueError(
                "Correlation cannot be calculated because "
                "one of the columns has no variance."
            )

        value = pair[first].corr(
            pair[second]
        )

        value = self._safe_number(
            value
        )

        return {
            "value": value,
            "answer": self._format_number(
                value,
                decimals=6,
            ),
            "columns": columns,
            "result_rows": [],
        }

    # ==================================================================
    # GROUPED OPERATIONS
    # ==================================================================

    def _execute_grouped(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        if not plan.group_by:
            raise ValueError(
                "Grouped operation requires group_by."
            )

        for column in plan.group_by:

            if column not in frame.columns:
                raise ValueError(
                    f"Group column does not exist: "
                    f"{column}"
                )

        operation = plan.operation

        if operation == "group_count":

            grouped = (
                frame
                .groupby(
                    plan.group_by,
                    dropna=False,
                )
                .size()
                .reset_index(
                    name="count"
                )
            )

            value_column = "count"

        else:

            if not plan.target_column:
                raise ValueError(
                    "Grouped numeric operation requires "
                    "a target column."
                )

            numeric = self._numeric_series(
                frame[
                    plan.target_column
                ],
                plan.target_column,
            )

            working = frame.copy()

            working[
                "__structured_numeric_target"
            ] = numeric

            grouped_obj = working.groupby(
                plan.group_by,
                dropna=False,
            )[
                "__structured_numeric_target"
            ]

            if operation == "group_sum":

                grouped = (
                    grouped_obj
                    .sum(
                        min_count=1
                    )
                    .reset_index(
                        name="value"
                    )
                )

            elif operation == "group_avg":

                grouped = (
                    grouped_obj
                    .mean()
                    .reset_index(
                        name="value"
                    )
                )

            elif operation == "group_min":

                grouped = (
                    grouped_obj
                    .min()
                    .reset_index(
                        name="value"
                    )
                )

            elif operation == "group_max":

                grouped = (
                    grouped_obj
                    .max()
                    .reset_index(
                        name="value"
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported grouped operation: "
                    f"{operation}"
                )

            value_column = "value"

        result_rows = (
            grouped
            .replace(
                {
                    float("nan"): None
                }
            )
            .to_dict(
                orient="records"
            )
        )

        result_rows = (
            self._clean_result_rows(
                result_rows
            )
        )

        return {
            "value": result_rows,
            "answer": self._format_table(
                result_rows
            ),
            "columns": (
                list(plan.group_by)
                + (
                    [value_column]
                    if value_column
                    else []
                )
            ),
            "result_rows": result_rows,
        }

    # ==================================================================
    # TOP / BOTTOM N
    # ==================================================================

    def _execute_ranking(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        if not plan.target_column:
            raise ValueError(
                "Ranking requires a target column."
            )

        if plan.target_column not in frame.columns:
            raise ValueError(
                f"Column does not exist: "
                f"{plan.target_column}"
            )

        limit = (
            plan.limit
            if plan.limit
            else 5
        )

        if limit < 1:
            raise ValueError(
                "Ranking limit must be greater than zero."
            )

        if limit > 1000:
            raise ValueError(
                "Ranking limit cannot exceed 1000."
            )

        numeric = self._numeric_series(
            frame[
                plan.target_column
            ],
            plan.target_column,
        )

        working = frame.copy()

        working[
            "__structured_numeric_target"
        ] = numeric

        working = working.dropna(
            subset=[
                "__structured_numeric_target"
            ]
        )

        # --------------------------------------------------------------
        # If group_by exists, aggregate first.
        # --------------------------------------------------------------

        if plan.group_by:

            for column in plan.group_by:

                if column not in frame.columns:
                    raise ValueError(
                        f"Group column does not exist: "
                        f"{column}"
                    )

            grouped = (
                working
                .groupby(
                    plan.group_by,
                    dropna=False,
                )[
                    "__structured_numeric_target"
                ]
                .sum()
                .reset_index(
                    name=plan.target_column
                )
            )

            ascending = (
                plan.operation
                == "bottom_n"
            )

            result = grouped.sort_values(
                by=plan.target_column,
                ascending=ascending,
                kind="stable",
            ).head(
                limit
            )

        else:

            ascending = (
                plan.operation
                == "bottom_n"
            )

            result = working.sort_values(
                by="__structured_numeric_target",
                ascending=ascending,
                kind="stable",
            ).head(
                limit
            )

            result[
                plan.target_column
            ] = result[
                "__structured_numeric_target"
            ]

        result = result.drop(
            columns=[
                "__structured_numeric_target"
            ],
            errors="ignore",
        )

        # Don't expose internal columns.
        result = result.drop(
            columns=[
                column
                for column in result.columns
                if column in INTERNAL_COLUMNS
            ],
            errors="ignore",
        )

        result_rows = (
            result
            .replace(
                {
                    float("nan"): None
                }
            )
            .to_dict(
                orient="records"
            )
        )

        result_rows = (
            self._clean_result_rows(
                result_rows
            )
        )

        return {
            "value": result_rows,
            "answer": self._format_table(
                result_rows
            ),
            "columns": list(
                result.columns
            ),
            "result_rows": result_rows,
        }

    # ==================================================================
    # PERCENTAGE
    # ==================================================================

    def _execute_percentage(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        if not plan.target_column:
            raise ValueError(
                "Percentage calculation requires "
                "a target numeric column."
            )

        numeric = self._numeric_series(
            frame[
                plan.target_column
            ],
            plan.target_column,
        )

        if numeric.empty:
            raise ValueError(
                "No numeric values available "
                "for percentage calculation."
            )

        # --------------------------------------------------------------
        # If filters exist, the numerator is the filtered data.
        #
        # But percentage requires access to the denominator.
        #
        # At this stage the executor receives the already-filtered
        # frame, so a true "percentage of total" query needs the
        # original frame.
        #
        # Therefore we intentionally fail rather than return a
        # potentially misleading percentage.
        # --------------------------------------------------------------

        if plan.filters:
            raise ValueError(
                "Percentage with filters requires the "
                "original unfiltered table and is not yet "
                "supported by this executor path."
            )

        total = numeric.sum()

        if total == 0:
            raise ValueError(
                "Cannot calculate percentage because "
                "the total is zero."
            )

        # Without a specific subgroup filter, percentage of total
        # is 100%.
        value = 100.0

        return {
            "value": value,
            "answer": self._format_number(
                value,
                decimals=4,
            ) + "%",
            "columns": [
                plan.target_column
            ],
            "result_rows": [],
        }

    # ==================================================================
    # RATIO
    # ==================================================================

    def _execute_ratio(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        if len(
            plan.target_columns
        ) != 2:

            raise ValueError(
                "Ratio requires exactly two "
                "target columns."
            )

        first, second = (
            plan.target_columns
        )

        left = self._numeric_series(
            frame[first],
            first,
        )

        right = self._numeric_series(
            frame[second],
            second,
        )

        pair = pd.DataFrame(
            {
                first: left,
                second: right,
            }
        ).dropna()

        pair = pair[
            pair[second] != 0
        ]

        if pair.empty:
            raise ValueError(
                "No valid non-zero denominator "
                "values for ratio."
            )

        ratio = (
            pair[first]
            / pair[second]
        )

        value = self._safe_number(
            ratio.mean()
        )

        return {
            "value": value,
            "answer": self._format_number(
                value,
                decimals=6,
            ),
            "columns": [
                first,
                second,
            ],
            "result_rows": [],
        }

    # ==================================================================
    # COMPARE
    # ==================================================================

    def _execute_compare(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        if len(
            plan.target_columns
        ) == 2:

            first, second = (
                plan.target_columns
            )

            first_numeric = (
                self._numeric_series(
                    frame[first],
                    first,
                )
            )

            second_numeric = (
                self._numeric_series(
                    frame[second],
                    second,
                )
            )

            first_value = (
                first_numeric.sum()
            )

            second_value = (
                second_numeric.sum()
            )

            difference = (
                first_value
                - second_value
            )

            return {
                "value": {
                    first: self._safe_number(
                        first_value
                    ),
                    second: self._safe_number(
                        second_value
                    ),
                    "difference": self._safe_number(
                        difference
                    ),
                },
                "answer": (
                    f"{first}: "
                    f"{self._format_number(first_value)}; "
                    f"{second}: "
                    f"{self._format_number(second_value)}; "
                    f"difference: "
                    f"{self._format_number(difference)}"
                ),
                "columns": [
                    first,
                    second,
                ],
                "result_rows": [],
            }

        if not plan.target_column:
            raise ValueError(
                "Comparison requires a target column."
            )

        numeric = self._numeric_series(
            frame[
                plan.target_column
            ],
            plan.target_column,
        )

        if numeric.empty:
            raise ValueError(
                "No numeric values available "
                "for comparison."
            )

        minimum = numeric.min()
        maximum = numeric.max()

        difference = (
            maximum
            - minimum
        )

        return {
            "value": {
                "min": self._safe_number(
                    minimum
                ),
                "max": self._safe_number(
                    maximum
                ),
                "difference": self._safe_number(
                    difference
                ),
            },
            "answer": (
                f"Minimum: "
                f"{self._format_number(minimum)}; "
                f"Maximum: "
                f"{self._format_number(maximum)}; "
                f"Difference: "
                f"{self._format_number(difference)}"
            ),
            "columns": [
                plan.target_column
            ],
            "result_rows": [],
        }

    # ==================================================================
    # TREND
    # ==================================================================

    def _execute_trend(
        self,
        frame: pd.DataFrame,
        plan: QueryPlan,
    ) -> Dict[str, Any]:

        if not plan.target_column:
            raise ValueError(
                "Trend requires a numeric target column."
            )

        if not plan.group_by:
            raise ValueError(
                "Trend requires a time/grouping column."
            )

        if len(
            plan.group_by
        ) != 1:

            raise ValueError(
                "Trend currently requires exactly "
                "one time/grouping column."
            )

        time_column = (
            plan.group_by[0]
        )

        if time_column not in frame.columns:
            raise ValueError(
                f"Time column does not exist: "
                f"{time_column}"
            )

        numeric = self._numeric_series(
            frame[
                plan.target_column
            ],
            plan.target_column,
        )

        dates = pd.to_datetime(
            frame[
                time_column
            ],
            errors="coerce",
        )

        working = pd.DataFrame(
            {
                time_column: dates,
                plan.target_column: numeric,
            }
        ).dropna()

        if working.empty:
            raise ValueError(
                "No valid time/numeric observations "
                "available for trend."
            )

        grouped = (
            working
            .groupby(
                time_column
            )[
                plan.target_column
            ]
            .sum()
            .reset_index()
            .sort_values(
                by=time_column
            )
        )

        result_rows = (
            grouped
            .assign(
                **{
                    time_column: grouped[
                        time_column
                    ].dt.strftime(
                        "%Y-%m-%d"
                    )
                }
            )
            .to_dict(
                orient="records"
            )
        )

        result_rows = (
            self._clean_result_rows(
                result_rows
            )
        )

        return {
            "value": result_rows,
            "answer": self._format_table(
                result_rows
            ),
            "columns": [
                time_column,
                plan.target_column,
            ],
            "result_rows": result_rows,
        }

    # ==================================================================
    # FILTERING
    # ==================================================================

    def _apply_filter(
        self,
        frame: pd.DataFrame,
        query_filter: QueryFilter,
    ) -> Tuple[
        pd.DataFrame,
        Dict[str, Any],
    ]:

        column = (
            query_filter.column
        )

        if column not in frame.columns:

            return (
                frame.iloc[0:0],
                {
                    "column": column,
                    "op": query_filter.op,
                    "value": query_filter.value,
                    "matched": False,
                    "reason": (
                        "column does not exist"
                    ),
                },
            )

        # --------------------------------------------------------------
        # Executor should never accept an unvalidated filter.
        # --------------------------------------------------------------

        if not query_filter.validated:

            return (
                frame.iloc[0:0],
                {
                    "column": column,
                    "op": query_filter.op,
                    "value": query_filter.value,
                    "matched": False,
                    "reason": (
                        "filter was not validated "
                        "by planner"
                    ),
                },
            )

        series = frame[column]

        op = query_filter.op

        raw_value = (
            query_filter.value
        )

        # --------------------------------------------------------------
        # Numeric comparisons
        # --------------------------------------------------------------

        if op in {
            "gt",
            "gte",
            "lt",
            "lte",
        }:

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            try:
                target = float(
                    raw_value
                )

            except (
                TypeError,
                ValueError,
            ):

                return (
                    frame.iloc[0:0],
                    {
                        "column": column,
                        "op": op,
                        "value": raw_value,
                        "matched": False,
                        "reason": (
                            "filter value is not numeric"
                        ),
                    },
                )

            if op == "gt":
                mask = numeric > target

            elif op == "gte":
                mask = numeric >= target

            elif op == "lt":
                mask = numeric < target

            else:
                mask = numeric <= target

            mask = mask.fillna(
                False
            )

            return (
                frame[mask],
                {
                    "column": column,
                    "op": op,
                    "value": raw_value,
                    "matched": True,
                    "rows_after": int(
                        mask.sum()
                    ),
                },
            )

        # --------------------------------------------------------------
        # Equality / contains
        # --------------------------------------------------------------

        normalized = (
            series
            .astype(str)
            .map(
                normalize_text
            )
        )

        wanted = normalize_text(
            raw_value
        )

        if op == "eq":

            mask = (
                normalized
                == wanted
            )

        elif op == "ne":

            mask = (
                normalized
                != wanted
            )

        elif op == "contains":

            mask = (
                normalized.str.contains(
                    wanted,
                    regex=False,
                    na=False,
                )
            )

        else:

            return (
                frame.iloc[0:0],
                {
                    "column": column,
                    "op": op,
                    "value": raw_value,
                    "matched": False,
                    "reason": (
                        f"unsupported filter operation: "
                        f"{op}"
                    ),
                },
            )

        mask = mask.fillna(
            False
        )

        return (
            frame[mask],
            {
                "column": column,
                "op": op,
                "requested_value": (
                    query_filter.requested_value
                ),
                "matched_value": raw_value,
                "matched": True,
                "rows_after": int(
                    mask.sum()
                ),
            },
        )

    # ==================================================================
    # ACTUAL DATA VALIDATION
    # ==================================================================

    def _validate_actual_frame(
        self,
        plan: QueryPlan,
        frame: pd.DataFrame,
    ) -> Optional[str]:

        required = set()

        if plan.target_column:
            required.add(
                plan.target_column
            )

        required.update(
            plan.target_columns
        )

        required.update(
            plan.group_by
        )

        required.update(
            item.column
            for item in plan.filters
        )

        if plan.sort_column:
            required.add(
                plan.sort_column
            )

        missing = [
            column
            for column in required
            if column not in frame.columns
        ]

        if missing:

            return (
                "Required columns are missing from "
                "the actual stored table: "
                + ", ".join(
                    missing
                )
            )

        # --------------------------------------------------------------
        # Numeric validation
        # --------------------------------------------------------------

        numeric_required = set()

        if plan.operation in NUMERIC_OPERATIONS:

            if plan.target_column:
                numeric_required.add(
                    plan.target_column
                )

        if plan.operation == "correlation":

            numeric_required.update(
                plan.target_columns
            )

        if plan.operation == "ratio":

            numeric_required.update(
                plan.target_columns
            )

        if plan.operation == "compare":

            numeric_required.update(
                plan.target_columns
            )

            if plan.target_column:
                numeric_required.add(
                    plan.target_column
                )

        for column in numeric_required:

            if column not in frame.columns:
                continue

            numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            usable = int(
                numeric.notna().sum()
            )

            if usable == 0:

                return (
                    f"Column '{column}' has no "
                    "usable numeric values."
                )

        return None

    # ==================================================================
    # SOURCES
    # ==================================================================

    def _sample_sources(
        self,
        frame: pd.DataFrame,
        schema: dict,
    ) -> List[
        Dict[str, Any]
    ]:

        samples = []

        preview_columns = [
            column["name"]
            for column in schema.get(
                "columns",
                [],
            )
            if column.get(
                "name"
            ) not in INTERNAL_COLUMNS
        ][:8]

        for _, row in (
            frame.head(5)
            .iterrows()
        ):

            item = {
                "document_name": schema.get(
                    "document_name"
                ),
                "document_id": self._table_id(
                    schema
                ),
                "row_number": int(
                    row.get(
                        "__row_number",
                        0,
                    )
                    or 0
                ),
            }

            for column in preview_columns:

                if column in row:

                    value = row[column]

                    item[column] = (
                        None
                        if pd.isna(
                            value
                        )
                        else str(
                            value
                        )
                    )

            samples.append(
                item
            )

        return samples

    # ==================================================================
    # NUMERIC HELPERS
    # ==================================================================

    def _numeric_series(
        self,
        series: pd.Series,
        column_name: str,
    ) -> pd.Series:

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        # Do not silently convert a completely non-numeric
        # column into zero.
        if numeric.notna().sum() == 0:

            raise ValueError(
                f"Column '{column_name}' does not "
                "contain usable numeric data."
            )

        return numeric.dropna()

    # ==================================================================
    # FORMATTERS
    # ==================================================================

    def _format_number(
        self,
        value: Any,
        decimals: int = 4,
    ) -> str:

        value = self._safe_number(
            value
        )

        if value is None:
            return ""

        if isinstance(
            value,
            int,
        ):
            return f"{value:,}"

        if isinstance(
            value,
            float,
        ):

            if value.is_integer():
                return (
                    f"{int(value):,}"
                )

            text = (
                f"{value:,.{decimals}f}"
                .rstrip("0")
                .rstrip(".")
            )

            return text

        return str(
            value
        )

    def _format_table(
        self,
        rows: List[
            Dict[str, Any]
        ],
    ) -> str:

        if not rows:
            return ""

        lines = []

        columns = list(
            rows[0].keys()
        )

        lines.append(
            " | ".join(
                columns
            )
        )

        lines.append(
            " | ".join(
                "---"
                for _ in columns
            )
        )

        for row in rows:

            values = []

            for column in columns:

                value = row.get(
                    column
                )

                if isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                ):

                    values.append(
                        self._format_number(
                            value
                        )
                    )

                else:

                    values.append(
                        ""
                        if value is None
                        else str(
                            value
                        )
                    )

            lines.append(
                " | ".join(
                    values
                )
            )

        return "\n".join(
            lines
        )

    # ==================================================================
    # CLEANING
    # ==================================================================

    def _clean_result_rows(
        self,
        rows: List[
            Dict[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        cleaned = []

        for row in rows:

            item = {}

            for key, value in row.items():

                if pd.isna(
                    value
                ):
                    item[key] = None
                    continue

                item[key] = self._safe_number(
                    value
                )

                if item[key] is value:
                    item[key] = value

            cleaned.append(
                item
            )

        return cleaned

    @staticmethod
    def _safe_number(
        value: Any,
    ) -> Any:

        if value is None:
            return None

        try:

            if pd.isna(
                value
            ):
                return None

        except Exception:
            pass

        if hasattr(
            value,
            "item",
        ):

            try:
                value = value.item()
            except Exception:
                pass

        if isinstance(
            value,
            float,
        ):

            if not math.isfinite(
                value
            ):
                return None

            if value.is_integer():
                return int(
                    value
                )

        return value

    # ==================================================================
    # TABLE ID
    # ==================================================================

    @staticmethod
    def _table_id(
        schema: dict,
    ) -> str:

        return str(
            schema.get(
                "table_id"
            )
            or schema.get(
                "document_id"
            )
            or ""
        )