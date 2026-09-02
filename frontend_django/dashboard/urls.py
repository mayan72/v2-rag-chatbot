from django.urls import path
from . import views


urlpatterns = [

    path(
        "",
        views.dashboard,
        name="dashboard",
    ),

    path(
        "chat/",
        views.chat,
        name="chat",
    ),

    path(
        "history/",
        views.history,
        name="history",
    ),

    path(
        "analytics/",
        views.analytics,
        name="analytics",
    ),

    path(
        "knowledge/",
        views.knowledge,
        name="knowledge",
    ),

    path(
        "settings/",
        views.settings_page,
        name="settings",
    ),

    # Browser -> Django -> FastAPI
    path(
        "api/chat/",
        views.ask_ai,
        name="ask_ai",
    ),
    path(
        "api/health/",
        views.api_health,
        name="api_health",
    ),
    path(
        "knowledge/upload/",
        views.upload_knowledge,
        name="upload_knowledge",
    ),
    path(
        "knowledge/clear/",
        views.clear_knowledge,
        name="clear_knowledge",
    ),
    path(
        "history/download/",
        views.download_question_answers,
        name="download_question_answers",
    ),
    
        
]