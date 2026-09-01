"""
Persistent structured store for CSV / XLSX uploads.

Design goals:
- CSV = one logical table
- Excel = one logical table per worksheet
- Preserve original rows and metadata
- Build a schema for every logical table
- Infer basic semantic types for safer query planning
- Never mix unrelated Excel sheets
- Remain independent from embeddings / semantic RAG
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import DATA_DIR, TABLE_STORE_DIR

logger = logging.getLogger(__name__)


INTERNAL_COLUMNS = {
    "__source_file",
    "__document_id",
    "__sheet_name",
    "__row_number",
    "__table_id",
}


def make_document_id(filename: str) -> str:
    filename = filename.lower()
    filename = re.sub(r"[^a-z0-9.]+", "_", filename).strip("_")
    return f"uploaded_{filename.replace('.', '_')}"


def make_table_id(document_id: str, sheet_name: Optional[str] = None) -> str:
    """
    Create a stable logical table identifier.

    CSV:
        uploaded_sales_csv

    Excel:
        uploaded_sales_xlsx__sheet_sales
    """
    if not sheet_name:
        return document_id

    normalized_sheet = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(sheet_name).casefold(),
    ).strip("_")

    return f"{document_id}__sheet_{normalized_sheet}"


class TableStore:

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or TABLE_STORE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def upsert_dataframe(
        self,
        df: pd.DataFrame,
        document_id: str,
        document_name: str,
        source_type: str,
        sheet_name: Optional[str] = None,
        table_id: Optional[str] = None,
    ) -> dict:
        """
        Store one logical table.

        CSV:
            one file -> one table

        Excel:
            one worksheet -> one table
        """

        if df is None or df.empty:
            raise ValueError("Cannot store an empty table.")

        stored = df.copy()

        table_id = table_id or make_table_id(
            document_id=document_id,
            sheet_name=sheet_name,
        )

        # --------------------------------------------------------------
        # Clean column names
        # --------------------------------------------------------------
        stored.columns = self._normalize_column_names(stored.columns)

        # --------------------------------------------------------------
        # Internal metadata
        # --------------------------------------------------------------
        stored["__document_id"] = document_id
        stored["__table_id"] = table_id
        stored["__source_file"] = document_name

        if sheet_name is not None:
            stored["__sheet_name"] = str(sheet_name)

        if "__row_number" not in stored.columns:
            stored["__row_number"] = range(1, len(stored) + 1)

        # --------------------------------------------------------------
        # Storage paths
        # --------------------------------------------------------------
        table_dir = self.root / table_id
        table_dir.mkdir(parents=True, exist_ok=True)

        data_path = table_dir / "data.jsonl"
        schema_path = table_dir / "schema.json"

        # --------------------------------------------------------------
        # Store data
        # --------------------------------------------------------------
        stored.to_json(
            data_path,
            orient="records",
            lines=True,
            date_format="iso",
        )

        # --------------------------------------------------------------
        # Build schema
        # --------------------------------------------------------------
        schema = self._build_schema(
            stored,
            document_id=document_id,
            document_name=document_name,
            source_type=source_type,
            sheet_name=sheet_name,
            table_id=table_id,
        )

        schema_path.write_text(
            json.dumps(
                schema,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Stored table | table_id=%s | document_id=%s | "
            "sheet=%s | rows=%d | columns=%d | path=%s",
            table_id,
            document_id,
            sheet_name,
            len(stored),
            len(stored.columns),
            data_path,
        )

        return schema

    def upsert_from_file(
        self,
        file_path: Path,
        filename: str,
        document_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Ingest CSV/XLS/XLSX.

        CSV:
            one file -> one table

        Excel:
            one worksheet -> one table

        Returns:
            A document-level ingestion summary.
        """

        path = Path(file_path)

        suffix = (
            Path(filename).suffix.lower()
            or path.suffix.lower()
        )

        document_id = document_id or make_document_id(filename)

        if suffix == ".csv":

            df = pd.read_csv(path)

            if df.empty:
                return None

            schema = self.upsert_dataframe(
                df=df,
                document_id=document_id,
                document_name=filename,
                source_type="csv",
                sheet_name=None,
                table_id=document_id,
            )

            return {
                "document_id": document_id,
                "document_name": filename,
                "source_type": "csv",
                "tables": [schema],
            }

        if suffix in {".xlsx", ".xls"}:

            return self._upsert_excel(
                path=path,
                filename=filename,
                document_id=document_id,
            )

        logger.warning(
            "Unsupported structured file type | filename=%s",
            filename,
        )

        return None

    def delete(self, document_id: str) -> None:
        """
        Delete all logical tables belonging to a document.
        """

        matching_tables = []

        for schema_path in self.root.glob("*/schema.json"):
            try:
                schema = json.loads(
                    schema_path.read_text(
                        encoding="utf-8"
                    )
                )

                if schema.get("document_id") == document_id:
                    matching_tables.append(
                        schema_path.parent
                    )

            except Exception:
                logger.exception(
                    "Failed to inspect schema %s",
                    schema_path,
                )

        for table_dir in matching_tables:
            for child in table_dir.iterdir():
                if child.is_file():
                    child.unlink()

            table_dir.rmdir()

    def list_schemas(self) -> List[dict]:
        """
        Return all logical table schemas.
        """

        schemas = []

        for schema_path in sorted(
            self.root.glob("*/schema.json")
        ):
            try:
                schemas.append(
                    json.loads(
                        schema_path.read_text(
                            encoding="utf-8"
                        )
                    )
                )

            except Exception:
                logger.exception(
                    "Failed to read schema %s",
                    schema_path,
                )

        return self._dedupe_schemas(schemas)

    def load_dataframe(
        self,
        table_id: str,
    ) -> pd.DataFrame:
        """
        Load a logical table by table_id.
        """

        data_path = (
            self.root
            / table_id
            / "data.jsonl"
        )

        if not data_path.exists():
            return pd.DataFrame()

        try:
            return pd.read_json(
                data_path,
                lines=True,
            )
        except Exception:
            logger.exception(
                "Failed to load table %s",
                table_id,
            )
            return pd.DataFrame()

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all logical tables.
        """

        tables = {}

        for schema in self.list_schemas():

            table_id = schema.get(
                "table_id"
            )

            if not table_id:
                continue

            df = self.load_dataframe(
                table_id
            )

            if not df.empty:
                tables[table_id] = df

        return tables

    def sync_from_data_dir(self) -> None:
        """
        Synchronize supported structured files from DATA_DIR.
        """

        if not DATA_DIR.exists():
            return

        for path in sorted(DATA_DIR.iterdir()):

            if path.suffix.lower() not in {
                ".csv",
                ".xlsx",
                ".xls",
            }:
                continue

            try:

                self.upsert_from_file(
                    path,
                    path.name,
                )

            except Exception:
                logger.exception(
                    "Failed to sync table from %s",
                    path,
                )

    # ------------------------------------------------------------------
    # EXCEL
    # ------------------------------------------------------------------

    def _upsert_excel(
        self,
        path: Path,
        filename: str,
        document_id: str,
    ) -> Optional[dict]:
        """
        Store each Excel worksheet independently.
        """

        workbook = pd.ExcelFile(path)

        tables = []

        for sheet_name in workbook.sheet_names:

            try:
                sheet = pd.read_excel(
                    path,
                    sheet_name=sheet_name,
                )

            except Exception:
                logger.exception(
                    "Failed to read Excel sheet | "
                    "file=%s | sheet=%s",
                    filename,
                    sheet_name,
                )
                continue

            if sheet.empty:
                logger.info(
                    "Skipping empty Excel sheet | "
                    "file=%s | sheet=%s",
                    filename,
                    sheet_name,
                )
                continue

            table_id = make_table_id(
                document_id=document_id,
                sheet_name=sheet_name,
            )

            schema = self.upsert_dataframe(
                df=sheet,
                document_id=document_id,
                document_name=filename,
                source_type="xlsx",
                sheet_name=str(sheet_name),
                table_id=table_id,
            )

            tables.append(schema)

        if not tables:
            return None

        return {
            "document_id": document_id,
            "document_name": filename,
            "source_type": "xlsx",
            "tables": tables,
        }

    # ------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------

    def _build_schema(
        self,
        df: pd.DataFrame,
        document_id: str,
        document_name: str,
        source_type: str,
        sheet_name: Optional[str],
        table_id: str,
    ) -> dict:
        """
        Build schema metadata for one logical table.
        """

        columns = []

        for name in df.columns:

            series = df[name]

            semantic_type = self._infer_semantic_type(
                series
            )

            sample_values = self._sample_values(
                series
            )

            columns.append(
                {
                    "name": str(name),
                    "dtype": str(series.dtype),

                    "semantic_type": semantic_type,

                    "nullable": bool(
                        series.isna().any()
                    ),

                    "unique_count": int(
                        series.nunique(
                            dropna=True
                        )
                    ),

                    "sample_values": sample_values,

                    "internal": (
                        str(name)
                        in INTERNAL_COLUMNS
                    ),
                }
            )

        return {
            "document_id": document_id,
            "document_name": document_name,
            "table_id": table_id,
            "sheet_name": (
                str(sheet_name)
                if sheet_name is not None
                else None
            ),
            "source_type": source_type,
            "indexed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "row_count": int(len(df)),
            "column_count": int(len(columns)),
            "columns": columns,
        }

    # ------------------------------------------------------------------
    # SEMANTIC TYPE INFERENCE
    # ------------------------------------------------------------------

    def _infer_semantic_type(
        self,
        series: pd.Series,
    ) -> str:
        """
        Infer a conservative semantic type.

        This is metadata for planning, not truth.

        Supported values:
            boolean
            integer
            numeric
            datetime
            date
            categorical
            text
            identifier
        """

        non_null = series.dropna()

        if non_null.empty:
            return "text"

        dtype = series.dtype

        # Boolean
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"

        # Integer
        if pd.api.types.is_integer_dtype(dtype):
            if self._looks_like_identifier(
                series
            ):
                return "identifier"
            return "integer"

        # Float / decimal
        if pd.api.types.is_float_dtype(dtype):
            return "numeric"

        # Datetime
        if pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            return "datetime"

        # Object columns may contain dates/numbers.
        if pd.api.types.is_object_dtype(dtype):

            date_ratio = self._date_parse_ratio(
                non_null
            )

            if date_ratio >= 0.90:
                return "date"

            numeric_ratio = self._numeric_parse_ratio(
                non_null
            )

            if numeric_ratio >= 0.95:
                numeric_values = pd.to_numeric(
                    non_null,
                    errors="coerce",
                )

                if self._looks_like_identifier(
                    series
                ):
                    return "identifier"

                if (
                    numeric_values.notna().sum()
                    > 0
                ):
                    return "numeric"

            unique_count = non_null.nunique(
                dropna=True
            )

            row_count = len(non_null)

            # Low-cardinality strings are useful
            # grouping/filtering dimensions.
            if (
                unique_count <= 100
                or (
                    row_count > 0
                    and unique_count / row_count
                    <= 0.20
                )
            ):
                return "categorical"

            return "text"

        return "text"

    def _looks_like_identifier(
        self,
        series: pd.Series,
    ) -> bool:
        """
        Conservative identifier detection.

        Avoid treating arbitrary numeric measures as IDs.
        """

        name = str(
            series.name or ""
        ).casefold()

        identifier_tokens = {
            "id",
            "code",
            "key",
            "identifier",
            "number",
            "no",
            "sku",
            "isin",
            "cusip",
        }

        tokens = set(
            re.findall(
                r"[a-z0-9]+",
                name,
            )
        )

        if tokens & identifier_tokens:
            return True

        # Unique integer measures (Quantity, Revenue, Age) are
        # not identifiers. Name tokens win over uniqueness.
        measure_tokens = {
            "quantity",
            "qty",
            "amount",
            "revenue",
            "price",
            "sales",
            "units",
            "volume",
            "count",
            "cost",
            "profit",
            "salary",
            "age",
            "rate",
            "score",
            "weight",
            "total",
            "value",
        }

        if tokens & measure_tokens:
            return False

        non_null = series.dropna()

        if non_null.empty:
            return False

        # Unique integer-like values are sometimes IDs.
        # Only use this as a weak signal.
        unique_ratio = (
            non_null.nunique(dropna=True)
            / len(non_null)
        )

        return bool(
            unique_ratio >= 0.98
            and pd.api.types.is_integer_dtype(
                series
            )
        )

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
    # SAMPLE VALUES
    # ------------------------------------------------------------------

    def _sample_values(
        self,
        series: pd.Series,
    ) -> List[str]:
        """
        Store representative values for schema matching.

        Do not store every distinct value because a production
        dataset may contain millions of values.
        """

        values = []

        try:
            unique_values = (
                series
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )

            for value in unique_values:

                if not value:
                    continue

                if value.casefold() == "nan":
                    continue

                values.append(value)

                if len(values) >= 50:
                    break

        except Exception:
            logger.exception(
                "Failed to build sample values for %s",
                series.name,
            )

        return values

    # ------------------------------------------------------------------
    # COLUMN NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_column_names(
        self,
        columns,
    ) -> List[str]:
        """
        Normalize column names while preserving their meaning.

        Examples:

            ' Revenue ' -> 'Revenue'
            'Order ID'  -> 'Order ID'

        We intentionally do NOT convert everything to snake_case
        because original column names are useful to users and
        query planners.
        """

        normalized = []

        seen = {}

        for original in columns:

            name = str(original).strip()

            if not name:
                name = "Unnamed"

            # Avoid collisions caused by duplicate
            # Excel headers.
            base = name

            count = seen.get(
                base.casefold(),
                0,
            )

            if count:
                name = f"{base}_{count + 1}"

            seen[
                base.casefold()
            ] = count + 1

            normalized.append(name)

        return normalized

    # ------------------------------------------------------------------
    # DEDUPLICATION
    # ------------------------------------------------------------------

    def _dedupe_schemas(
        self,
        schemas: List[dict],
    ) -> List[dict]:
        """
        Deduplicate only true duplicate logical tables.

        IMPORTANT:
        Different Excel sheets must NOT be deduplicated merely
        because they happen to have the same columns.
        """

        grouped: Dict[str, dict] = {}

        for schema in schemas:

            table_id = schema.get(
                "table_id"
            )

            if not table_id:
                # Backward compatibility with old schemas.
                table_id = schema.get(
                    "document_id"
                )

            current = grouped.get(
                table_id
            )

            if current is None:
                grouped[
                    table_id
                ] = schema
                continue

            if (
                schema.get(
                    "indexed_at",
                    "",
                )
                >= current.get(
                    "indexed_at",
                    "",
                )
            ):
                grouped[
                    table_id
                ] = schema

        return list(
            grouped.values()
        )