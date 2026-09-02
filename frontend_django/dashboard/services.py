"""
Services responsible for communicating
with the FastAPI backend.
"""

from __future__ import annotations

import logging

import requests

from .constants import CHAT_ENDPOINT


logger = logging.getLogger(__name__)


# ============================================================
# FastAPI Configuration
# ============================================================

FASTAPI_BASE_URL = "http://127.0.0.1:8000"


class FastAPIService:
    """
    Client used by Django to communicate with FastAPI.

    Django does not directly access:
        - ChromaDB
        - RAGChatbot
        - CSV logs
        - HistoryService
        - MetricsService

    All backend communication goes through FastAPI.
    """

    TIMEOUT_CHAT = 120
    TIMEOUT_READ = 30

    def __init__(
        self,
        base_url=FASTAPI_BASE_URL,
    ):

        self.base_url = base_url.rstrip("/")

    # ==========================================================
    # Chat
    # ==========================================================

    def chat(
        self,
        question: str,
    ):

        logger.info(
            "Calling FastAPI chat endpoint."
        )

        response = requests.post(

            f"{self.base_url}/chat",

            json={
                "question": question,
            },

            timeout=self.TIMEOUT_CHAT,
        )

        response.raise_for_status()

        return response.json()

    # ==========================================================
    # History
    # ==========================================================

    def history(
        self,
        limit: int = 100,
    ):

        logger.info(
            "Calling FastAPI history endpoint. limit=%s",
            limit,
        )

        response = requests.get(

            f"{self.base_url}/history",

            params={
                "limit": limit,
            },

            timeout=self.TIMEOUT_READ,
        )

        response.raise_for_status()

        return response.json()

    # ==========================================================
    # Analytics
    # ==========================================================

    def analytics(self):

        logger.info(
            "Calling FastAPI analytics endpoint."
        )

        response = requests.get(

            f"{self.base_url}/analytics",

            timeout=self.TIMEOUT_READ,
        )

        response.raise_for_status()

        return response.json()

    # ==========================================================
    # Knowledge Upload
    # ==========================================================

    def upload_knowledge(
        self,
        file,
    ):

        response = requests.post(

            f"{self.base_url}/knowledge/upload",

            files={
                "file": (
                    file.name,
                    file,
                    getattr(
                        file,
                        "content_type",
                        "application/octet-stream",
                    ),
                )
            },

            timeout=600,
        )

        response.raise_for_status()

        return response.json()

    def clear_knowledge(self):

        response = requests.post(
            f"{self.base_url}/knowledge/clear",
            timeout=self.TIMEOUT_CHAT,
        )

        response.raise_for_status()

        return response.json()

    # ==========================================================
    # Metrics
    # ==========================================================

    def metrics(self):

        logger.info(
            "Calling FastAPI metrics endpoint."
        )

        response = requests.get(

            f"{self.base_url}/metrics",

            timeout=self.TIMEOUT_READ,
        )

        response.raise_for_status()

        return response.json()

    # ==========================================================
    # Download Q&A Excel
    # ==========================================================

    def download_question_answers(self):

        response = requests.get(

            f"{self.base_url}/history/download",

            timeout=60,

        )

        response.raise_for_status()

        return response

    # ==========================================================
    # Health
    # ==========================================================

    def health(self):

        logger.info(
            "Calling FastAPI health endpoint."
        )

        response = requests.get(

            f"{self.base_url}/health",

            timeout=5,

        )

        response.raise_for_status()

        return response.json()


# ============================================================
# Singleton
# ============================================================

fastapi_service = FastAPIService()


# ============================================================
# Backward Compatibility
# ============================================================

class RAGService:
    """
    Backward-compatible wrapper.

    Existing code using:

        RAGService.ask(question)

    will continue to work.

    Internally it uses FastAPIService.
    """

    TIMEOUT = 120

    @staticmethod
    def ask(
        question: str,
    ):

        return fastapi_service.chat(
            question
        )

