"""
Central Run Logger

Responsibilities
----------------
1. Log successful requests.
2. Log failed requests.
3. Write JSON logs.
4. Write CSV logs.
5. Produce structured application logs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any

from logger.csv_logger import CSVLogger
from logger.json_logger import JsonLogger

logger = logging.getLogger(__name__)


class RunLogger:
    """
    Central logging class.

    Chatbot sends one payload.
    RunLogger writes it everywhere.
    """

    def __init__(self):

        self.csv_logger = CSVLogger()

        self.json_logger = JsonLogger()

        logger.info("RunLogger initialized.")

    # ============================================================
    # Success Logging
    # ============================================================

    def log_success(
        self,
        payload: Dict[str, Any],
    ) -> None:
        """
        Log successful request.
        """

        payload = dict(payload)

        payload["status"] = "SUCCESS"

        payload["timestamp"] = datetime.utcnow().isoformat()

        self.json_logger.log(payload)

        self.csv_logger.log(payload)

        logger.debug("Request logged successfully.")

    # ============================================================
    # Failure Logging
    # ============================================================

    def log_failure(
        self,
        question: str,
        error: Exception,
        stage: str,
    ) -> None:
        """
        Log failed request.
        """

        payload = {

            "status": "FAILED",

            "question": question,

            "error": str(error),

            "stage": stage,

            "timestamp": datetime.utcnow().isoformat()

        }

        self.json_logger.log_error(payload)
        self.csv_logger.log_error(payload)

        logger.exception(
            "Pipeline failed at stage=%s",
            stage,
        )