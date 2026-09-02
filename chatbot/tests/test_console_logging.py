from logger.console import (
    configure_logging,
    document_names,
    format_filters,
    qlog,
)


class _Filter:
    def __init__(self, column, op, value):
        self.column = column
        self.op = op
        self.value = value


def test_format_filters_from_query_filter():
    filters = [
        _Filter("region", "eq", "EMEA"),
        _Filter("status", "eq", "Open"),
    ]
    assert format_filters(filters) == "region eq EMEA; status eq Open"


def test_document_names_dedupes_and_prefers_document_name():
    chunks = [
        type("Chunk", (), {"metadata": {"document_name": "sales.xlsx"}})(),
        type("Chunk", (), {"metadata": {"document_name": "sales.xlsx"}})(),
        {"document_name": "hr.csv"},
    ]
    assert document_names(chunks) == ["sales.xlsx", "hr.csv"]


def test_qlog_writes_compact_terminal_line(capsys):
    configure_logging()
    qlog(
        "QUERY TYPE",
        type="structured",
        operation="sum",
        document="sales.xlsx",
        filters="region eq EMEA",
    )
    captured = capsys.readouterr().out
    assert "QUERY TYPE" in captured
    assert "type=structured" in captured
    assert "document=sales.xlsx" in captured
    assert "operation=sum" in captured
