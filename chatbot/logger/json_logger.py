"""
JSON Logger

Creates:
    logs/rag_runs.jsonl

Each request is stored as one JSON object per line.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

from config import JSON_LOG_FILE

logger = logging.getLogger(__name__)


class JsonLogger:
    """
    Append-only JSONL logger.
    """

    def __init__(self):

        self.file_path = Path(JSON_LOG_FILE)

        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Success
    # ==========================================================

    def log(
        self,
        payload: Dict[str, Any],
    ) -> None:

        with open(
            self.file_path,
            "a",
            encoding="utf-8",
        ) as fp:

            json.dump(
                payload,
                fp,
                ensure_ascii=False,
                default=str,
            )

            fp.write("\n")

        logger.debug("JSON log written.")

    # ==========================================================
    # Failure
    # ==========================================================

    def log_error(
        self,
        payload: Dict[str, Any],
    ) -> None:

        payload = dict(payload)

        payload["status"] = "FAILED"

        with open(
            self.file_path,
            "a",
            encoding="utf-8",
        ) as fp:

            json.dump(
                payload,
                fp,
                ensure_ascii=False,
                default=str,
            )

            fp.write("\n")

        logger.error("Failure JSON log written.")