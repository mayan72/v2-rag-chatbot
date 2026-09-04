"""
Hybrid question answering.

1. If uploaded tables can answer an aggregation exactly, do that.
2. Otherwise fall back to semantic RAG.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag.query_planner import QueryPlanner
from rag.structured_executor import StructuredExecutor, StructuredResult
from rag.table_store import TableStore
from debug_trace import dbg
from logger.console import format_filters, qlog

logger = logging.getLogger(__name__)


class HybridQAEngine:

    def __init__(
        self,
        table_store: Optional[TableStore] = None,
        planner: Optional[QueryPlanner] = None,
        executor: Optional[StructuredExecutor] = None,
        llm: Any = None,
    ):
        self.table_store = table_store or TableStore()
        self.planner = planner or QueryPlanner(table_store=self.table_store)
        self.executor = executor or StructuredExecutor(self.table_store)
        self.llm = llm
        self.table_store.sync_from_data_dir()

    def answer(self, question: str) -> Optional[StructuredResult]:
        dbg("HYBRID_START", question=question)

        schemas = self.table_store.list_schemas()
        if not schemas:
            self.table_store.sync_from_data_dir()
            schemas = self.table_store.list_schemas()

        dbg(
            "HYBRID_SCHEMAS",
            schema_count=len(schemas),
            tables=[
                {
                    "document_id": schema.get("document_id"),
                    "document_name": schema.get("document_name"),
                    "row_count": schema.get("row_count"),
                    "columns": [
                        column.get("name")
                        for column in schema.get("columns", [])
                    ][:40],
                }
                for schema in schemas
            ],
        )

        if not schemas:
            qlog("DOCUMENTS", available="-")
            dbg("HYBRID_SKIP", reason="no tables in table_store")
            return None

        plan = self.planner.plan(
            question=question,
            schemas=schemas,
            llm=self.llm,
        )

        dbg(
            "HYBRID_PLAN",
            mode=plan.mode,
            operation=plan.operation,
            target_column=plan.target_column,
            table_id=plan.table_id,
            confidence=plan.confidence,
            reason=plan.reason,
            filters=[
                {
                    "column": item.column,
                    "op": item.op,
                    "value": item.value,
                    "score": item.score,
                }
                for item in plan.filters
            ],
        )

        selected_document = ""
        if plan.table_id:
            for schema in schemas:
                ids = {
                    schema.get("document_id"),
                    schema.get("table_id"),
                }
                if plan.table_id in ids:
                    selected_document = (
                        schema.get("document_name")
                        or plan.table_id
                    )
                    break

        qlog(
            "QUERY TYPE",
            type=plan.mode,
            operation=plan.operation,
            document=selected_document,
            table=plan.table_id,
            column=plan.target_column,
            columns=plan.target_columns,
            group_by=plan.group_by,
            filters=format_filters(plan.filters),
            confidence=plan.confidence,
            reason=plan.reason,
        )

        if plan.refuse_semantic_fallback:
            errors = (
                "; ".join(plan.validation_errors)
                or plan.reason
                or "Could not apply the requested subset filter."
            )
            dbg(
                "HYBRID_BLOCKED",
                reason=errors,
                leftover_filters=[
                    {
                        "column": item.column,
                        "value": item.value,
                    }
                    for item in plan.filters
                ],
            )
            return StructuredResult(
                matched=False,
                blocked=True,
                answer=errors,
                error=errors,
                operation=plan.operation or "",
                table_id=plan.table_id or "",
                document_name=selected_document,
                filters=[
                    {
                        "column": item.column,
                        "op": item.op,
                        "value": item.value,
                    }
                    for item in plan.filters
                ],
                confidence=0.0,
            )

        # QueryPlanner emits mode="structured" for table calculations.
        # The executor only runs those plans. "aggregate" is accepted
        # only as a backward-compatible alias.
        if plan.mode not in {"structured", "aggregate"}:
            dbg("HYBRID_SKIP", reason="plan is not structured", mode=plan.mode)
            return None

        return self.executor.execute(plan, schemas)
