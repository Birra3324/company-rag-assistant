"""Ingest local markdown/text into the vector store."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import IngestOut
from app.rag.ingest import ingest_directory

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestOut)
def ingest_endpoint() -> IngestOut:
    settings = get_settings()
    result = ingest_directory(settings=settings)
    return IngestOut(
        documents=result.documents,
        chunks=result.chunks,
        embedding_provider=settings.embedding_provider,
        docs_path=result.docs_path,
    )
