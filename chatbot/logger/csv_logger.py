"""
CSV Logger

Responsibilities
----------------
1. Create CSV log file if it does not exist.
2. Append one row per request.
3. Never overwrite existing logs.
4. Store execution metrics consistently.
5. Production-safe logging.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Any

from config import LOG_DIR

logger = logging.getLogger(__name__)


CSV_LOG_FILE = LOG_DIR / "rag_logs.csv"


CSV_COLUMNS = [

    "timestamp",

    "status",

    "request_id",

    # ---------------------------------------------------------
    # LLM
    # ---------------------------------------------------------

    "provider",

    "model",

    # ---------------------------------------------------------
    # Request / Response
    # ---------------------------------------------------------

    "question",

    "answer",

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    "confidence",

    "max_similarity",

    "should_answer",

    "chunks_retrieved",

    "context_length",

    "retrieval_threshold",

    "top_k",

    # ---------------------------------------------------------
    # Performance
    #
    # All times are stored in milliseconds.
    # ---------------------------------------------------------

    "retrieval_time_ms",

    "llm_time_ms",

    "llm_provider_latency_ms",

    "total_time_ms",

    # ---------------------------------------------------------
    # LLM configuration
    # ---------------------------------------------------------

    "temperature",

    # ---------------------------------------------------------
    # Tokens
    # ---------------------------------------------------------

    "input_tokens",

    "output_tokens",

    "total_tokens",

    # ---------------------------------------------------------
    # Cost
    # ---------------------------------------------------------

    "input_cost",

    "output_cost",

    "embedding_cost",

    "total_cost",

    # ---------------------------------------------------------
    # Sources / Errors
    # ---------------------------------------------------------

    "sources",

    "error",
]


class CSVLogger:

    def __init__(self):

        self.file_path = Path(
            CSV_LOG_FILE
        )

        self._create_file()

        logger.info(
            "CSV Logger initialized: %s",
            self.file_path,
        )

    # ============================================================
    # Create CSV
    # ============================================================

    def _create_file(self):

        if self.file_path.exists():
            return

        with open(
            self.file_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS,
            )

            writer.writeheader()

    # ============================================================
    # Append
    # ============================================================

    def log(
        self,
        payload: Dict[str, Any],
    ) -> None:

        row = {}

        for column in CSV_COLUMNS:

            value = payload.get(
                column,
                "",
            )

            # --------------------------------------------------
            # Keep numerical values readable
            # --------------------------------------------------

            if isinstance(value, float):

                value = round(
                    value,
                    3,
                )

            # --------------------------------------------------
            # Convert lists/dictionaries to strings
            # --------------------------------------------------

            if isinstance(
                value,
                (list, dict),
            ):

                value = str(value)

            row[column] = value

        # ------------------------------------------------------
        # Append row
        # ------------------------------------------------------

        with open(
            self.file_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_COLUMNS,
            )

            writer.writerow(row)

        logger.debug(
            "CSV log written | "
            "Provider=%s | "
            "Model=%s | "
            "Question=%s",
            row.get("provider"),
            row.get("model"),
            row.get("question"),
        )

    # ============================================================
    # Error Logging
    # ============================================================

    def log_error(
        self,
        payload: Dict[str, Any],
    ) -> None:

        payload.setdefault(
            "status",
            "FAILED",
        )

        self.log(payload)