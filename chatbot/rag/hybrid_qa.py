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
        self.planner = planner or QueryPlanner()
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
            dbg("HYBRID_SKIP", reason="no tables in table_store")
        if not schemas:
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

        logger.info(
            "Query plan | mode=%s | op=%s | filters=%s | confidence=%.2f | reason=%s",
            plan.mode,
            plan.operation,
            [(item.column, item.op, item.value) for item in plan.filters],
            plan.confidence,
            plan.reason,
        )

        # QueryPlanner emits mode="structured" for table calculations.
        # The executor only runs those plans. "aggregate" is accepted
        # only as a backward-compatible alias.
        if plan.mode not in {"structured", "aggregate"}:
            dbg("HYBRID_SKIP", reason="plan is not structured", mode=plan.mode)
            return None

        return self.executor.execute(plan, schemas)
