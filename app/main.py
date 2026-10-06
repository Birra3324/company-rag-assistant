"""Application factory. Keep this module importable as `app.main:app`."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import ask, health, ingest, ui
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.limits import BodySizeLimitMiddleware, RateLimitMiddleware
from app.core.logging import RequestIdMiddleware, setup_logging
from app.models.schemas import RootOut
from app.rag.ingest import maybe_auto_ingest


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not get_settings().api_key.strip():
        raise RuntimeError("API_KEY must be configured before starting the service")
    maybe_auto_ingest()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Local RAG knowledge assistant for fictional Vision AI Ops docs. "
            "Token-aware chunking, hybrid BM25 + dense retrieval, extractive answers."
        ),
        lifespan=lifespan,
    )
    # Added last runs first: request id wraps the limit checks.
    application.add_middleware(RateLimitMiddleware)
    application.add_middleware(BodySizeLimitMiddleware)
    application.add_middleware(RequestIdMiddleware)
    register_exception_handlers(application)
    application.include_router(health.router)
    application.include_router(ui.router)
    application.include_router(ask.router)
    application.include_router(ingest.router)

    @application.get("/", response_model=RootOut, tags=["health"])
    def root() -> RootOut:
        return RootOut(service=settings.app_name, version=__version__)

    return application


app = create_app()
