"""Public health check."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthOut
from app.rag.store import get_store

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_settings()
    store = get_store(settings)
    chunks = store.count()
    ok = True
    status = "ok" if chunks else "empty"
    return HealthOut(
        ok=ok,
        status=status,
        chunks=chunks,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        vector_backend=store.backend,
        auto_ingest=settings.auto_ingest_on_startup,
    )
