"""
Generic schema profiler for structured CSV / Excel tables.

Purpose
-------
Analyze an already-ingested table schema and provide metadata that can
be safely consumed by the query planner and validator.

This module is intentionally generic:
- No commodity names are hardcoded.
- No business-specific column names are hardcoded.
- No LLM is used.
- No calculations are executed for answering user questions.
- Inference is conservative; uncertain classifications remain uncertain.

Supported semantic types:
    boolean
    integer
    numeric
    date
    datetime
    categorical
    text
    identifier

Column roles:
    measure
    dimension
    time
    identifier
    text
    boolean
    unknown
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from rag.table_store import INTERNAL_COLUMNS

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class ColumnProfile:
    """
    Profile for one column.
    """

    name: str

    dtype: str

    semantic_type: str

    role: str

    nullable: bool

    row_count: int

    non_null_count: int

    unique_count: int

    unique_ratio: float

    sample_values: List[str] = field(default_factory=list)

    min_value: Optional[Any] = None
    max_value: Optional[Any] = None

    mean_value: Optional[float] = None

    numeric_parse_ratio: float = 0.0
    date_parse_ratio: float = 0.0

    likely_measure: bool = False
    likely_dimension: bool = False
    likely_time: bool = False
    likely_identifier: bool = False

    unit: Optional[str] = None

    inference_confidence: float = 0.0

    internal: bool = False


@dataclass
class TableProfile:
    """
    Profile for one logical table.
    """

    table_id: str

    document_id: str

    document_name: str

    sheet_name: Optional[str]

    source_type: str

    row_count: int

    column_count: int

    columns: List[ColumnProfile] = field(
        default_factory=list
    )

    numeric_columns: List[str] = field(
        default_factory=list
    )

    categorical_columns: List[str] = field(
        default_factory=list
    )

    date_columns: List[str] = field(
        default_factory=list
    )

    datetime_columns: List[str] = field(
        default_factory=list
    )

    identifier_columns: List[str] = field(
        default_factory=list
    )

    text_columns: List[str] = field(
        default_factory=list
    )

    measure_columns: List[str] = field(
        default_factory=list
    )

    dimension_columns: List[str] = field(
        default_factory=list
    )

    time_columns: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Profiler
# ----------------------------------------------------------------------


class SchemaProfiler:
    """
    Generic profiler for structured tables.

    The profiler consumes a pandas DataFrame and optional schema metadata
    from TableStore.

    It does NOT decide what the user question means.

    Example:

        profiler = SchemaProfiler()

        profile = profiler.profile(
            df,
            schema,
        )

        print(profile.numeric_columns)
    """

    def __init__(
        self,
        max_sample_values: int = 50,
        categorical_max_unique: int = 100,
        categorical_max_ratio: float = 0.20,
    ):
        self.max_sample_values = max(
            1,
            int(max_sample_values),
        )

        self.categorical_max_unique = max(
            1,
            int(categorical_max_unique),
        )

        self.categorical_max_ratio = max(
            0.0,
            min(
                1.0,
                float(categorical_max_ratio),
            ),
        )

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def profile(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> TableProfile:
        """
        Build a complete table profile.
        """

        if df is None or df.empty:
            raise ValueError(
                "Cannot profile an empty DataFrame."
            )

        schema = schema or {}

        table_id = str(
            schema.get(
                "table_id"
            )
            or schema.get(
                "document_id"
            )
            or ""
        )

        document_id = str(
            schema.get(
                "document_id"
            )
            or table_id
        )

        document_name = str(
            schema.get(
                "document_name"
            )
            or ""
        )

        sheet_name = schema.get(
            "sheet_name"
        )

        source_type = str(
            schema.get(
                "source_type"
            )
            or ""
        )

        profiles: List[ColumnProfile] = []

        for column in df.columns:

            column_name = str(column)

            try:
                profile = self.profile_column(
                    series=df[column],
                    column_name=column_name,
                )

                profiles.append(
                    profile
                )

            except Exception:
                logger.exception(
                    "Failed to profile column | "
                    "table=%s | column=%s",
                    table_id,
                    column_name,
                )

                # Fail closed.
                profiles.append(
                    ColumnProfile(
                        name=column_name,
                        dtype=str(
                            df[column].dtype
                        ),
                        semantic_type="text",
                        role="unknown",
                        nullable=True,
                        row_count=len(df),
                        non_null_count=int(
                            df[column].notna().sum()
                        ),
                        unique_count=int(
                            df[column].nunique(
                                dropna=True
                            )
                        ),
                        unique_ratio=0.0,
                        inference_confidence=0.0,
                        internal=(
                            column_name
                            in INTERNAL_COLUMNS
                        ),
                    )
                )

        result = TableProfile(
            table_id=table_id,
            document_id=document_id,
            document_name=document_name,
            sheet_name=(
                str(sheet_name)
                if sheet_name is not None
                else None
            ),
            source_type=source_type,
            row_count=int(len(df)),
            column_count=len(profiles),
            columns=profiles,
        )

        self._build_column_indexes(
            result
        )

        return result

    def profile_column(
        self,
        series: pd.Series,
        column_name: Optional[str] = None,
    ) -> ColumnProfile:
        """
        Profile a single column.
        """

        name = str(
            column_name
            if column_name is not None
            else series.name
        )

        row_count = int(
            len(series)
        )

        non_null = series.dropna()

        non_null_count = int(
            len(non_null)
        )

        unique_count = int(
            non_null.nunique(
                dropna=True
            )
        )

        unique_ratio = (
            unique_count / non_null_count
            if non_null_count
            else 0.0
        )

        semantic_type = self._infer_semantic_type(
            series
        )

        role = self._infer_role(
            series=series,
            semantic_type=semantic_type,
            unique_ratio=unique_ratio,
        )

        numeric_ratio = (
            self._numeric_parse_ratio(
                non_null
            )
        )

        date_ratio = (
            self._date_parse_ratio(
                non_null
            )
        )

        samples = self._sample_values(
            non_null
        )

        min_value = None
        max_value = None
        mean_value = None

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric_values = (
            numeric_series.dropna()
        )

        if not numeric_values.empty:

            min_value = self._safe_scalar(
                numeric_values.min()
            )

            max_value = self._safe_scalar(
                numeric_values.max()
            )

            mean_value = self._safe_float(
                numeric_values.mean()
            )

        unit = self._infer_unit(
            name
        )

        confidence = (
            self._inference_confidence(
                semantic_type=semantic_type,
                role=role,
                numeric_ratio=numeric_ratio,
                date_ratio=date_ratio,
                unique_ratio=unique_ratio,
            )
        )

        return ColumnProfile(
            name=name,
            dtype=str(
                series.dtype
            ),
            semantic_type=semantic_type,
            role=role,
            nullable=bool(
                series.isna().any()
            ),
            row_count=row_count,
            non_null_count=non_null_count,
            unique_count=unique_count,
            unique_ratio=round(
                unique_ratio,
                4,
            ),
            sample_values=samples,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            numeric_parse_ratio=round(
                numeric_ratio,
                4,
            ),
            date_parse_ratio=round(
                date_ratio,
                4,
            ),
            likely_measure=(
                role == "measure"
            ),
            likely_dimension=(
                role == "dimension"
            ),
            likely_time=(
                role == "time"
            ),
            likely_identifier=(
                role == "identifier"
            ),
            unit=unit,
            inference_confidence=round(
                confidence,
                4,
            ),
            internal=(
                name in INTERNAL_COLUMNS
            ),
        )

    # ------------------------------------------------------------------
    # TYPE INFERENCE
    # ------------------------------------------------------------------

    def _infer_semantic_type(
        self,
        series: pd.Series,
    ) -> str:
        """
        Conservative semantic type inference.

        We don't try to guess domain-specific meanings here.
        """

        non_null = series.dropna()

        if non_null.empty:
            return "text"

        dtype = series.dtype

        if pd.api.types.is_bool_dtype(
            dtype
        ):
            return "boolean"

        if pd.api.types.is_integer_dtype(
            dtype
        ):
            if self._looks_like_identifier(
                series
            ):
                return "identifier"

            return "integer"

        if pd.api.types.is_float_dtype(
            dtype
        ):
            return "numeric"

        if pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            return "datetime"

        if pd.api.types.is_object_dtype(
            dtype
        ):

            date_ratio = (
                self._date_parse_ratio(
                    non_null
                )
            )

            if date_ratio >= 0.95:
                return "date"

            numeric_ratio = (
                self._numeric_parse_ratio(
                    non_null
                )
            )

            if numeric_ratio >= 0.98:

                if self._looks_like_identifier(
                    series
                ):
                    return "identifier"

                return "numeric"

            if self._looks_boolean(
                non_null
            ):
                return "boolean"

            if self._looks_like_identifier(
                series
            ):
                return "identifier"

            if self._looks_categorical(
                non_null
            ):
                return "categorical"

            return "text"

        return "text"

    # ------------------------------------------------------------------
    # ROLE INFERENCE
    # ------------------------------------------------------------------

    def _infer_role(
        self,
        series: pd.Series,
        semantic_type: str,
        unique_ratio: float,
    ) -> str:
        """
        Determine how a query planner can use the column.

        This is intentionally broader than semantic_type.

        Example:
            numeric -> measure
            categorical -> dimension
            date -> time
            identifier -> identifier
        """

        name = str(
            series.name or ""
        ).casefold()

        if semantic_type in {
            "integer",
            "numeric",
        }:

            if self._looks_like_identifier(
                series
            ):
                return "identifier"

            return "measure"

        if semantic_type in {
            "date",
            "datetime",
        }:
            return "time"

        if semantic_type == "categorical":
            return "dimension"

        if semantic_type == "identifier":
            return "identifier"

        if semantic_type == "boolean":
            return "boolean"

        if semantic_type == "text":

            # A text field whose name strongly indicates
            # descriptive content should remain text.
            text_tokens = {
                "description",
                "comment",
                "notes",
                "remarks",
                "summary",
                "text",
            }

            name_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    name,
                )
            )

            if name_tokens & text_tokens:
                return "text"

            # High-cardinality text can still be used
            # for exact filtering, but don't call it a
            # dimension automatically.
            if unique_ratio <= self.categorical_max_ratio:
                return "dimension"

            return "text"

        return "unknown"

    # ------------------------------------------------------------------
    # NUMERIC / DATE
    # ------------------------------------------------------------------

    def _numeric_parse_ratio(
        self,
        series: pd.Series,
    ) -> float:
        if series.empty:
            return 0.0

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        return float(
            numeric.notna().mean()
        )

    def _date_parse_ratio(
        self,
        series: pd.Series,
    ) -> float:
        if series.empty:
            return 0.0

        try:
            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            return float(
                parsed.notna().mean()
            )

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # CATEGORICAL
    # ------------------------------------------------------------------

    def _looks_categorical(
        self,
        series: pd.Series,
    ) -> bool:
        if series.empty:
            return False

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        row_count = len(series)

        unique_ratio = (
            unique_count / row_count
            if row_count
            else 0.0
        )

        return bool(
            unique_count
            <= self.categorical_max_unique
            or unique_ratio
            <= self.categorical_max_ratio
        )

    # ------------------------------------------------------------------
    # IDENTIFIER
    # ------------------------------------------------------------------

    def _looks_like_identifier(
        self,
        series: pd.Series,
    ) -> bool:
        """
        Conservative identifier detection.

        Numeric IDs must not accidentally become measures.
        """

        name = str(
            series.name or ""
        ).casefold()

        tokens = set(
            re.findall(
                r"[a-z0-9]+",
                name,
            )
        )

        identifier_tokens = {
            "id",
            "code",
            "key",
            "identifier",
            "sku",
            "isin",
            "cusip",
            "uuid",
            "account",
            "customerid",
            "productid",
            "orderid",
        }

        if tokens & identifier_tokens:
            return True

        non_null = series.dropna()

        if non_null.empty:
            return False

        unique_ratio = (
            non_null.nunique(
                dropna=True
            )
            / len(non_null)
        )

        # Only treat numeric columns as identifiers
        # through this weak heuristic when they are
        # almost completely unique.
        if (
            unique_ratio >= 0.995
            and pd.api.types.is_integer_dtype(
                series
            )
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # BOOLEAN
    # ------------------------------------------------------------------

    def _looks_boolean(
        self,
        series: pd.Series,
    ) -> bool:
        if series.empty:
            return False

        normalized = (
            series.astype(str)
            .str.casefold()
            .str.strip()
        )

        unique = set(
            normalized.dropna().unique()
        )

        boolean_values = {
            "true",
            "false",
            "yes",
            "no",
            "y",
            "n",
            "1",
            "0",
        }

        return bool(
            unique
            and unique.issubset(
                boolean_values
            )
        )

    # ------------------------------------------------------------------
    # SAMPLE VALUES
    # ------------------------------------------------------------------

    def _sample_values(
        self,
        series: pd.Series,
    ) -> List[str]:
        values = []

        try:

            unique_values = (
                series
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "nan": None,
                        "NaT": None,
                    }
                )
                .dropna()
                .unique()
            )

            for value in unique_values:

                text = str(
                    value
                ).strip()

                if not text:
                    continue

                values.append(text)

                if (
                    len(values)
                    >= self.max_sample_values
                ):
                    break

        except Exception:
            logger.exception(
                "Failed to sample column values"
            )

        return values

    # ------------------------------------------------------------------
    # UNIT INFERENCE
    # ------------------------------------------------------------------

    def _infer_unit(
        self,
        column_name: str,
    ) -> Optional[str]:
        """
        Conservative unit inference from column name.

        This is intentionally NOT commodity-specific.

        If no reliable unit is found, return None.

        Examples:
            Price (USD/t) -> USD/t
            Weight (kg)   -> kg
            Margin (%)    -> %
        """

        text = str(
            column_name
        ).strip()

        if not text:
            return None

        patterns = [
            r"\(([^()]{1,30})\)",
            r"\[([^\[\]]{1,30})\]",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
            )

            if not match:
                continue

            candidate = (
                match.group(1)
                .strip()
            )

            if self._looks_like_unit(
                candidate
            ):
                return candidate

        # Common explicit suffixes.
        suffix_patterns = [
            r"(?:^|[\s_-])usd/?t$",
            r"(?:^|[\s_-])usd/?kg$",
            r"(?:^|[\s_-])usd/?lb$",
            r"(?:^|[\s_-])eur/?t$",
            r"(?:^|[\s_-])kg$",
            r"(?:^|[\s_-])g$",
            r"(?:^|[\s_-])tonnes?$",
            r"(?:^|[\s_-])tons?$",
            r"(?:^|[\s_-])mt$",
            r"(?:^|[\s_-])kt$",
            r"(?:^|[\s_-])%$",
        ]

        normalized = (
            text.casefold()
            .replace(" ", "")
        )

        for pattern in suffix_patterns:

            if re.search(
                pattern,
                normalized,
            ):
                return normalized

        return None

    def _looks_like_unit(
        self,
        value: str,
    ) -> bool:
        text = value.casefold().strip()

        unit_tokens = {
            "usd/t",
            "usd/kg",
            "usd/lb",
            "eur/t",
            "eur/kg",
            "$/t",
            "$/kg",
            "$/lb",
            "kg",
            "g",
            "gram",
            "grams",
            "ton",
            "tons",
            "tonne",
            "tonnes",
            "mt",
            "kt",
            "%",
            "percent",
            "pct",
            "bbl",
            "barrel",
            "barrels",
        }

        return text in unit_tokens

    # ------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------

    def _inference_confidence(
        self,
        semantic_type: str,
        role: str,
        numeric_ratio: float,
        date_ratio: float,
        unique_ratio: float,
    ) -> float:
        """
        Conservative confidence score.

        This is confidence in schema inference,
        NOT confidence in the final user answer.
        """

        if semantic_type in {
            "integer",
            "numeric",
        }:
            return min(
                1.0,
                max(
                    numeric_ratio,
                    0.90,
                ),
            )

        if semantic_type in {
            "date",
            "datetime",
        }:
            return min(
                1.0,
                max(
                    date_ratio,
                    0.90,
                ),
            )

        if semantic_type == "boolean":
            return 0.95

        if semantic_type == "identifier":
            return 0.90

        if semantic_type == "categorical":
            return 0.85

        if semantic_type == "text":
            return 0.80

        return 0.50

    # ------------------------------------------------------------------
    # INDEXES
    # ------------------------------------------------------------------

    def _build_column_indexes(
        self,
        profile: TableProfile,
    ) -> None:

        for column in profile.columns:

            if column.internal:
                continue

            if column.semantic_type in {
                "integer",
                "numeric",
            }:
                profile.numeric_columns.append(
                    column.name
                )

            if column.semantic_type == "categorical":
                profile.categorical_columns.append(
                    column.name
                )

            if column.semantic_type == "date":
                profile.date_columns.append(
                    column.name
                )

            if column.semantic_type == "datetime":
                profile.datetime_columns.append(
                    column.name
                )

            if column.semantic_type == "identifier":
                profile.identifier_columns.append(
                    column.name
                )

            if column.semantic_type == "text":
                profile.text_columns.append(
                    column.name
                )

            if column.likely_measure:
                profile.measure_columns.append(
                    column.name
                )

            if column.likely_dimension:
                profile.dimension_columns.append(
                    column.name
                )

            if column.likely_time:
                profile.time_columns.append(
                    column.name
                )

    # ------------------------------------------------------------------
    # SAFE VALUE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_scalar(
        value: Any,
    ) -> Any:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if hasattr(
            value,
            "item",
        ):
            try:
                return value.item()
            except Exception:
                pass

        return value

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> Optional[float]:
        try:
            value = float(value)

            if not math.isfinite(
                value
            ):
                return None

            return value

        except (
            TypeError,
            ValueError,
        ):
            return None