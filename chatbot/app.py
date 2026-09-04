"""
FastAPI entry point for the RAG Chatbot.

Responsibilities:
- Start FastAPI
- Load chatbot once during startup
- Expose REST APIs
- Handle exceptions
- Return JSON responses

Business logic should NOT be written here.
"""
from fastapi.responses import StreamingResponse
from services.qa_export_service import QAExportService
from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path
import tempfile
# from fastapi import FastAPI, HTTPException
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
)
import pandas as pd
from pydantic import BaseModel
from typing import Optional
from services.metrics_service import MetricsService
from rag.chatbot import RAGChatbot
from services.history_service import HistoryService
from services.metrics_service import MetricsService
from services.knowledge_service import KnowledgeService
from logger.console import configure_logging, qlog
# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

configure_logging()

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Global chatbot instance
# -------------------------------------------------------------------------

chatbot = None

history_service = HistoryService()

metrics_service = MetricsService()
qa_export_service = QAExportService()
knowledge_service = KnowledgeService()

# -------------------------------------------------------------------------
# FastAPI lifespan
# -------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the chatbot only once.
    """

    global chatbot

    qlog("STARTUP", status="loading chatbot")

    chatbot = RAGChatbot()

    qlog("STARTUP", status="chatbot ready")

    yield

    logger.info("Application shutdown complete.")


# -------------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------------

app = FastAPI(
    title="CSV RAG Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------------------------------------------------------
# Request Model
# -------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    conversation_id: str = ""


# -------------------------------------------------------------------------
# Response Model
# -------------------------------------------------------------------------

class ChatResponse(BaseModel):

    answer: str

    confidence: float

    provider: str

    model: str

    response_time_ms: float

    retrieval_time_ms: float

    llm_time_ms: float

    llm_provider_latency_ms: float

    total_time_ms: float

    input_tokens: int

    output_tokens: int

    total_tokens: int

    cost: float

    sources: list = []

    conversation_id: str = ""

    original_question: str = ""

    resolved_question: str = ""

    status: str = "SUCCESS"

    intent: str = ""

    clarification_required: bool = False

    clarification_question: Optional[str] = None

    clarification_options: list = []

    formula: Optional[str] = None

    calculation_result: Optional[float] = None


# -------------------------------------------------------------------------
# Response Model
# -------------------------------------------------------------------------

class ChatResponse(BaseModel):

    answer: str

    confidence: float

    provider: str

    model: str

    response_time_ms: float

    retrieval_time_ms: float

    llm_time_ms: float

    llm_provider_latency_ms: float

    total_time_ms: float

    input_tokens: int

    output_tokens: int

    total_tokens: int

    cost: float

    sources: list

# -------------------------------------------------------------------------
# Health Check
# -------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# -------------------------------------------------------------------------
# Chat Endpoint
# -------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    start = time.perf_counter()

    try:

        result = chatbot.ask(
            request.question,
            conversation_id=request.conversation_id,
        )

        total_time = (time.perf_counter() - start) * 1000

        return ChatResponse(

    answer=result["answer"],

    confidence=result["confidence"],

    provider=result["provider"],

    model=result["model"],

    response_time_ms=round(
        total_time,
        2,
    ),

    retrieval_time_ms=result[
        "retrieval_time_ms"
    ],

    llm_time_ms=result[
        "llm_time_ms"
    ],

    llm_provider_latency_ms=result[
        "llm_provider_latency_ms"
    ],

    total_time_ms=result[
        "total_time_ms"
    ],

    input_tokens=result[
        "input_tokens"
    ],

    output_tokens=result[
        "output_tokens"
    ],

    total_tokens=result[
        "total_tokens"
    ],

    cost=result[
        "cost"
    ],

    sources=result.get(
        "sources",
        [],
    ),
    conversation_id=result.get("conversation_id", ""),
    original_question=result.get("original_question", request.question),
    resolved_question=result.get("resolved_question", request.question),
    status=result.get("status", "SUCCESS"),
    intent=result.get("intent", ""),
    clarification_required=bool(result.get("clarification_required")),
    clarification_question=result.get("clarification_question"),
    clarification_options=result.get("clarification_options") or [],
    formula=result.get("formula"),
    calculation_result=result.get("calculation_result"),
)

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------------------------------------------------------------------------
# Root
# -------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Production RAG Chatbot API is running."
    }


metrics_service = MetricsService()

@app.get("/dashboard")
def dashboard():

    return metrics_service.get_dashboard_summary()


# -------------------------------------------------------------------------
# Knowledge Upload
# -------------------------------------------------------------------------

@app.post("/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
):
    """
    Upload and index a knowledge document.

    Supported:
    - PDF
    - JPG
    - CSV
    - XLSX

    Maximum size:
    - 10 MB
    """

    temp_path = None

    try:

        # ---------------------------------------------------------
        # Validate filename
        # ---------------------------------------------------------

        if not file.filename:

            raise HTTPException(
                status_code=400,
                detail="Filename is required.",
            )

        filename = Path(
            file.filename
        ).name

        # ---------------------------------------------------------
        # Read file
        # ---------------------------------------------------------

        content = await file.read()

        file_size = len(content)

        qlog(
            "UPLOAD",
            file=filename,
            size_bytes=file_size,
        )

        # ---------------------------------------------------------
        # Validate
        # ---------------------------------------------------------

        knowledge_service.validate_file(
            filename=filename,
            file_size=file_size,
        )

        # ---------------------------------------------------------
        # Temporary file
        # ---------------------------------------------------------

        suffix = Path(filename).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_file.write(content)

            temp_path = Path(
                temp_file.name
            )

        # ---------------------------------------------------------
        # Index document
        # ---------------------------------------------------------

        result = knowledge_service.index_document(
            file_path=temp_path,
            filename=filename,
        )

        return {
            "success": True,
            "message": (
                "Document indexed successfully."
            ),
            "data": result,
        }

    except ValueError as exc:

        logger.warning(
            "Knowledge upload validation failed | %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "Knowledge upload failed.",
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    finally:

        # ---------------------------------------------------------
        # Remove temporary file
        # ---------------------------------------------------------

        if temp_path and temp_path.exists():

            try:

                temp_path.unlink()

            except Exception:

                logger.warning(
                    "Unable to remove temporary file: %s",
                    temp_path,
                )


@app.post("/knowledge/clear")
def clear_knowledge():
    """
    Delete stored embeddings and uploaded table data.
    Upload/index behavior is unchanged.
    """

    try:
        result = knowledge_service.clear_all()
        return {
            "success": True,
            "message": "Knowledge data cleared.",
            "data": result,
        }

    except Exception as exc:
        logger.exception("Knowledge clear failed.")
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/settings")
def settings():

    return metrics_service.get_settings()

@app.get("/history")
def get_history(
    limit: int = 100,
):
    try:

        records = history_service.get_history(
            limit=limit
        )

        return {
            "success": True,
            "data": records,
        }

    except Exception as exc:

        logger.exception(
            "Failed to retrieve history."
        )

        return {
            "success": False,
            "message": str(exc),
            "data": [],
        }

@app.get("/analytics")
def get_analytics():

    try:

        analytics = (
            history_service.get_analytics()
        )

        summary = (
            history_service.get_summary()
        )

        return {
            "success": True,
            "data": {
                "analytics": analytics,
                "summary": summary,
            },
        }

    except Exception as exc:

        logger.exception(
            "Failed to retrieve analytics."
        )

        return {
            "success": False,
            "message": str(exc),
        }


# -------------------------------------------------------------------------
# Download Question / Answer Excel
# -------------------------------------------------------------------------

@app.get("/history/download")
def download_question_answers():

    try:

        df = qa_export_service.create_excel()

        if df.empty:

            raise HTTPException(
                status_code=404,
                detail="No question and answer history available.",
            )

        # ------------------------------------------------------
        # Create Excel in memory
        # ------------------------------------------------------

        from io import BytesIO

        output = BytesIO()

        with pd.ExcelWriter(
            output,
            engine="openpyxl",
        ) as writer:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Q&A History",
            )

            worksheet = writer.sheets[
                "Q&A History"
            ]

            # --------------------------------------------------
            # Basic column widths
            # --------------------------------------------------

            worksheet.column_dimensions[
                "A"
            ].width = 60

            worksheet.column_dimensions[
                "B"
            ].width = 100

        output.seek(0)

        return StreamingResponse(

            output,

            media_type=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),

            headers={
                "Content-Disposition":
                    'attachment; '
                    'filename="chatbot_question_answers.xlsx"'
            },
        )

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "Failed to create Q&A Excel."
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )