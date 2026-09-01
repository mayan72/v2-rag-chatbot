from pathlib import Path

import pandas as pd

from rag.query_planner import QueryPlanner
from rag.structured_executor import StructuredExecutor
from rag.table_store import TableStore


def _engine(tmp_path: Path, frames: dict):
    store = TableStore(root=tmp_path / "tables")
    for name, frame in frames.items():
        store.upsert_dataframe(
            df=frame,
            document_id=f"uploaded_{name}",
            document_name=f"{name}.xlsx",
            source_type="xlsx",
        )
    planner = QueryPlanner()
    executor = StructuredExecutor(store)
    return store, planner, executor


def test_count_filters_any_schema(tmp_path):
    sales = pd.DataFrame(
        {
            "region": ["EMEA", "APAC", "EMEA", "AMER"],
            "status": ["Open", "Closed", "Open", "Open"],
            "amount": [10, 20, 30, 40],
        }
    )
    store, planner, executor = _engine(tmp_path, {"sales": sales})
    schemas = store.list_schemas()

    plan = planner.plan(
        "Give me the count of rows where status is Open for EMEA",
        schemas,
        llm=None,
    )

    assert plan.mode == "structured"
    result = executor.execute(plan, schemas)
    assert result.answer == "2"


def test_sum_on_different_file(tmp_path):
    hr = pd.DataFrame(
        {
            "department": ["Finance", "Finance", "Engineering"],
            "level": ["Senior", "Junior", "Senior"],
            "salary": [120000, 80000, 150000],
        }
    )
    store, planner, executor = _engine(tmp_path, {"hr": hr})
    schemas = store.list_schemas()

    plan = planner.plan(
        "What is the sum of salary where department is Finance",
        schemas,
        llm=None,
    )

    assert plan.operation == "sum"
    result = executor.execute(plan, schemas)
    assert result.answer == "200000.0"


def test_typo_value_still_matches(tmp_path):
    metals = pd.DataFrame(
        {
            "commodity_name": [
                "Aluminium cash-settlement (LME)",
                "Copper cash-settlement (LME)",
                "Aluminium cash-settlement (LME)",
            ],
            "risk_rating": ["High", "Medium", "Medium"],
        }
    )
    store, planner, executor = _engine(tmp_path, {"metals": metals})
    schemas = store.list_schemas()

    plan = planner.plan(
        "Give me the count of rows where risk rating is medium for Almunium case settlement",
        schemas,
        llm=None,
    )

    result = executor.execute(plan, schemas)
    assert result.answer == "1"

    plan_high = planner.plan(
        "Give me the count of rows where risk rating is High for Almunium case settlement",
        schemas,
        llm=None,
    )
    result_high = executor.execute(plan_high, schemas)
    assert result_high.answer == "1"


def test_total_revenue_for_region(tmp_path):
    sales = pd.DataFrame(
        {
            "Region": ["North", "South", "North", "West"],
            "Revenue": [275000, 50000, 330000, 10000],
        }
    )
    store, planner, executor = _engine(tmp_path, {"sales": sales})
    schemas = store.list_schemas()

    plan = planner.plan(
        "What is the total revenue for the North region?",
        schemas,
        llm=None,
    )

    assert plan.mode == "structured"
    assert plan.operation == "sum"
    assert plan.target_column == "Revenue"
    assert any(
        item.column == "Region" and "north" in item.value.casefold()
        for item in plan.filters
    )
    result = executor.execute(plan, schemas)
    assert float(result.answer) == 605000.0


def test_correlation_quantity_sold_and_revenue(tmp_path):
    sales = pd.DataFrame(
        {
            "Order_ID": list(range(101, 119)),
            "Quantity": list(range(1, 19)),
            "Unit_Price": [1000] * 18,
            "Revenue": [1000 * qty for qty in range(1, 19)],
        }
    )
    store, planner, executor = _engine(tmp_path, {"sales": sales})
    schemas = store.list_schemas()

    quantity_schema = next(
        column
        for column in next(schema["columns"] for schema in schemas)
        if column["name"] == "Quantity"
    )
    assert quantity_schema["semantic_type"] == "integer"

    plan = planner.plan(
        "What is the correlation between Quantity sold and Revenue, "
        "and is the relationship positive or negative?",
        schemas,
        llm=None,
    )

    assert plan.valid
    assert plan.operation == "correlation"
    assert set(plan.target_columns) == {"Quantity", "Revenue"}

    result = executor.execute(plan, schemas)
    assert result.matched
    value = float(result.answer)
    assert value > 0.99


def test_correlation_resolves_quantity_typed_as_identifier():
    planner = QueryPlanner()
    schemas = [
        {
            "table_id": "uploaded_xlsx__sheet_sales_data",
            "document_id": "uploaded_xlsx",
            "document_name": "rag_structured_numeric_test.xlsx",
            "sheet_name": "Sales_Data",
            "row_count": 18,
            "columns": [
                {
                    "name": "Order_ID",
                    "semantic_type": "identifier",
                    "dtype": "int64",
                },
                {
                    "name": "Quantity",
                    "semantic_type": "identifier",
                    "dtype": "int64",
                },
                {
                    "name": "Unit_Price",
                    "semantic_type": "numeric",
                    "dtype": "int64",
                },
                {
                    "name": "Revenue",
                    "semantic_type": "numeric",
                    "dtype": "int64",
                },
            ],
        }
    ]

    plan = planner.plan(
        "What is the correlation between Quantity sold and Revenue, "
        "and is the relationship positive or negative?",
        schemas,
        llm=None,
    )

    assert plan.valid
    assert plan.operation == "correlation"
    assert set(plan.target_columns) == {"Quantity", "Revenue"}


def test_non_aggregate_stays_semantic(tmp_path):
    sales = pd.DataFrame({"region": ["EMEA"], "note": ["Demand improved"]})
    store, planner, _ = _engine(tmp_path, {"sales": sales})
    plan = planner.plan(
        "Summarize the latest demand commentary",
        store.list_schemas(),
        llm=None,
    )
    assert plan.mode == "semantic"
