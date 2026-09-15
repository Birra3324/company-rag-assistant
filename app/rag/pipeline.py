"""Ask a question: embed → retrieve (hybrid BM25 + dense) → generate."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import HTTPException

from app.core.config import Settings, get_settings
from app.rag.embeddings import Embedder, get_embedder
from app.rag.generate import Generator, get_generator
from app.rag.retrieve import hybrid_blend
from app.rag.store import Hit, VectorStore, get_store

log = logging.getLogger("rag.pipeline")


@dataclass(frozen=True)
class AskResult:
    answer: str
    hits: list[Hit]
    embedding_provider: str
    llm_provider: str


def hybrid_rerank(question: str, hits: list[Hit], k: int) -> list[Hit]:
    """Blend BM25 + dense over a candidate list (used by tests and fallback)."""
    return hybrid_blend(question, hits, hits, k)


def ask(
    question: str,
    *,
    top_k: int | None = None,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    generator: Generator | None = None,
) -> AskResult:
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    store = store or get_store(settings)
    generator = generator or get_generator(settings)
    k = top_k or settings.retrieve_k

    if store.count() == 0:
        raise HTTPException(
            status_code=409,
            detail="No documents in the vector store. POST /ingest first.",
        )

    query_vec = embedder.embed([question])[0]
    pool = max(k * 8, settings.retrieve_pool)
    dense_hits = store.query(query_vec, k=pool)
    corpus = store.list_chunks() or dense_hits
    hits = hybrid_blend(question, dense_hits, corpus, k)
    hits = [h for h in hits if h.score >= settings.min_retrieve_score] or hits[:1]
    answer = generator.generate(question, hits)
    log.info(
        "ask retrieved=%s embedder=%s llm=%s",
        len(hits),
        embedder.name,
        generator.name,
    )
    return AskResult(
        answer=answer,
        hits=hits,
        embedding_provider=embedder.name,
        llm_provider=generator.name,
    )
