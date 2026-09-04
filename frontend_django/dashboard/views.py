import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST

from .services import fastapi_service
from django.http import (
    JsonResponse,
    HttpResponse,
)
import json
import time
logger = logging.getLogger(__name__)

# ============================================================
# Dashboard
# ============================================================

from .services import fastapi_service

def dashboard(request):

    try:

        result = fastapi_service.analytics()

        data = result.get(
            "data",
            {},
        )

        summary = data.get(
            "summary",
            {},
        )

    except Exception:

        logger.exception(
            "Unable to load dashboard summary."
        )

        summary = {}

    return render(
        request,
        "dashboard/chat.html",
        {
            "summary": summary,
        },
    )


# ============================================================
# Chat
# ============================================================

@ensure_csrf_cookie
def chat(request):

    return render(
        request,
        "dashboard/chat.html",
    )


# ============================================================
# Chat API
# ============================================================
@require_POST
def ask_ai(request):

    try:

        data = json.loads(
            request.body
        )

        question = data.get(
            "question",
            "",
        ).strip()

        if not question:

            return JsonResponse(
                {
                    "success": False,
                    "message": "Question is required.",
                },
                status=400,
            )

        conversation_id = data.get(
            "conversation_id",
            "",
        )

        result = fastapi_service.chat(
            question,
            conversation_id=conversation_id,
        )

        return JsonResponse(
            {
                "success": True,
                "data": result,
            }
        )

    except Exception as exc:

        logger.exception(
            "AI request failed."
        )

        return JsonResponse(
            {
                "success": False,
                "message": str(exc),
            },
            status=500,
        )

# ============================================================
# API Health
# ============================================================

def api_health(request):

    try:

        start_time = time.perf_counter()

        result = fastapi_service.health()

        response_time = (
            time.perf_counter() - start_time
        ) * 1000

        return JsonResponse(
            {
                "success": True,
                "status": result.get(
                    "status",
                    "unknown",
                ),
                "response_time_ms": round(
                    response_time,
                    2,
                ),
            }
        )

    except Exception as exc:

        logger.exception(
            "FastAPI health check failed."
        )

        return JsonResponse(
            {
                "success": False,
                "status": "offline",
                "message": str(exc),
            },
            status=503,
        )
# ============================================================
# History
# ============================================================
def history(request):

    try:

        result = fastapi_service.history(
            limit=100
        )

        records = result.get(
            "data",
            [],
        )

        logger.info(
            "History records received: %d",
            len(records),
        )

    except Exception:

        logger.exception(
            "Unable to load history."
        )

        records = []

    return render(
        request,
        "dashboard/history.html",
        {
            "records": records,
        },
    )


# ============================================================
# Analytics
# ============================================================
def analytics(request):

    try:

        result = fastapi_service.analytics()

        data = result.get(
            "data",
            {},
        )

        analytics_data = data.get(
            "analytics",
            {},
        )

        summary = data.get(
            "summary",
            {},
        )

    except Exception:

        logger.exception(
            "Unable to load analytics."
        )

        analytics_data = {}

        summary = {}

    return render(
        request,
        "dashboard/analytics.html",
        {
            "analytics": analytics_data,
            "summary": summary,
        },
    )
# ============================================================
# Knowledge
# ============================================================

def knowledge(request):

    return render(
        request,
        "dashboard/knowledge.html",
    )

# ============================================================
# Knowledge Upload
# ============================================================

@require_POST
def upload_knowledge(request):

    try:

        uploaded_file = request.FILES.get(
            "file"
        )

        if not uploaded_file:

            return JsonResponse(
                {
                    "success": False,
                    "message": "No file uploaded.",
                },
                status=400,
            )

        logger.info(
            "Knowledge upload requested | file=%s",
            uploaded_file.name,
        )

        result = fastapi_service.upload_knowledge(
            uploaded_file
        )

        return JsonResponse(
            result
        )

    except Exception as exc:

        logger.exception(
            "Knowledge upload failed."
        )

        return JsonResponse(
            {
                "success": False,
                "message": str(exc),
            },
            status=500,
        )


@require_POST
def clear_knowledge(request):

    try:

        result = fastapi_service.clear_knowledge()

        return JsonResponse(
            result
        )

    except Exception as exc:

        logger.exception(
            "Knowledge clear failed."
        )

        return JsonResponse(
            {
                "success": False,
                "message": str(exc),
            },
            status=500,
        )


# ============================================================
# Settings
# ============================================================

def settings_page(request):

    return render(
        request,
        "dashboard/settings.html",
    )

# ============================================================
# Download Question / Answer Excel
# ============================================================

def download_question_answers(request):

    try:

        response = (
            fastapi_service.download_question_answers()
        )

        django_response = HttpResponse(

            response.content,

            content_type=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),

        )

        django_response[
            "Content-Disposition"
        ] = (
            'attachment; '
            'filename="chatbot_question_answers.xlsx"'
        )

        return django_response

    except requests.HTTPError as exc:

        logger.exception(
            "Unable to download Q&A Excel."
        )

        return JsonResponse(
            {
                "success": False,
                "message": str(exc),
            },
            status=404,
        )

    except Exception as exc:

        logger.exception(
            "Unable to download Q&A Excel."
        )

        return JsonResponse(
            {
                "success": False,
                "message": str(exc),
            },
            status=500,
        )