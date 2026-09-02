"""
Compact terminal logging for RAG debugging.

Full step traces still go to chatbot/logs/rag_debug.log via debug_trace.
The terminal only shows high-signal lines: question, query type,
documents, plan, and result.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Iterable, Mapping

QUERY_LOGGER_NAME = "query"

logger = logging.getLogger(QUERY_LOGGER_NAME)

_CONFIGURED = False


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "-"
        return ", ".join(_fmt(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "-"
        return ", ".join(
            f"{key}={_fmt(item)}"
            for key, item in value.items()
        )
    return str(value)


def format_filters(filters: Any) -> str:
    if not filters:
        return ""

    parts = []
    for item in filters:
        if hasattr(item, "column"):
            parts.append(
                f"{item.column} {item.op} {item.value}"
            )
            continue
        if isinstance(item, Mapping):
            column = (
                item.get("column")
                or item.get("field")
                or ""
            )
            op = item.get("op") or item.get("operator") or "="
            value = item.get("value")
            if column:
                parts.append(f"{column} {op} {value}")
            else:
                parts.append(_fmt(item))
            continue
        parts.append(_fmt(item))
    return "; ".join(parts)


def document_names(
    items: Iterable[Any],
) -> list[str]:
    names: list[str] = []
    seen = set()

    for item in items:
        name = ""
        if isinstance(item, Mapping):
            name = (
                item.get("document_name")
                or item.get("source")
                or item.get("filename")
                or item.get("document_id")
                or ""
            )
        elif hasattr(item, "metadata"):
            meta = item.metadata or {}
            name = (
                meta.get("document_name")
                or meta.get("source")
                or meta.get("filename")
                or ""
            )
        elif isinstance(item, str):
            name = item

        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    return names


def qlog(event: str, **fields: Any) -> None:
    configure_logging()
    parts = []
    for key, value in fields.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        parts.append(f"{key}={_fmt(value)}")
    if parts:
        logger.info("%s | %s", event, " | ".join(parts))
    else:
        logger.info("%s", event)


def configure_logging() -> None:
    """Quiet routine INFO logs; keep warnings and query traces."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    if not root.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(message)s",
            stream=sys.stdout,
        )
    else:
        root.setLevel(logging.WARNING)
        for handler in root.handlers:
            handler.setLevel(logging.WARNING)
            if handler.formatter is None:
                handler.setFormatter(formatter)

    for noisy in (
        "httpx",
        "httpcore",
        "openai",
        "google",
        "google_genai",
        "chromadb",
        "chromadb.telemetry",
        "sentence_transformers",
        "uvicorn.access",
        "watchfiles",
        "urllib3",
        "langchain",
        "langchain_core",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    query = logging.getLogger(QUERY_LOGGER_NAME)
    query.setLevel(logging.INFO)
    query.propagate = False
    if not any(
        isinstance(handler, logging.StreamHandler)
        for handler in query.handlers
    ):
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(message)s")
        )
        query.addHandler(handler)

    _CONFIGURED = True
