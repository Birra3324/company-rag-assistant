"""Ask / query the ingested company knowledge base."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_request_id
from app.models.schemas import AskIn, AskOut, SourceOut
from app.rag.chunking import stable_title
from app.rag.pipeline import ask

router = APIRouter(tags=["ask"])


def _run(payload: AskIn) -> AskOut:
    result = ask(payload.question.strip(), top_k=payload.top_k)
    sources = [
        SourceOut(
            source=hit.source,
            title=stable_title(hit.title, hit.source),
            score=round(hit.score, 4),
            excerpt=hit.text[:280],
            chunk_id=hit.chunk_id or None,
            heading=(hit.heading or None),
        )
        for hit in result.hits
    ]
    _ = get_request_id()
    return AskOut(
        answer=result.answer,
        sources=sources,
        retrieved=len(result.hits),
        embedding_provider=result.embedding_provider,
        llm_provider=result.llm_provider,
    )


@router.post("/ask", response_model=AskOut)
def ask_endpoint(payload: AskIn) -> AskOut:
    return _run(payload)


@router.post("/query", response_model=AskOut)
def query_endpoint(payload: AskIn) -> AskOut:
    """Alias of POST /ask."""
    return _run(payload)
