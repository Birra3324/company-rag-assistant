"""Ask a question: embed → retrieve (hybrid) → generate."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from fastapi import HTTPException

from app.core.config import Settings, get_settings
from app.rag.embeddings import QUERY_STOP, Embedder, get_embedder, tokenize
from app.rag.generate import Generator, get_generator
from app.rag.store import Hit, VectorStore, get_store

log = logging.getLogger("rag.pipeline")


@dataclass(frozen=True)
class AskResult:
    answer: str
    hits: list[Hit]
    embedding_provider: str
    llm_provider: str


def _query_terms(question: str) -> set[str]:
    return {t for t in tokenize(question) if t not in QUERY_STOP and len(t) > 1}


def _keyword_overlap(question: str, text: str) -> float:
    q = _query_terms(question)
    if not q:
        return 0.0
    d = set(tokenize(text))
    return len(q & d) / len(q)


def hybrid_rerank(question: str, hits: list[Hit], k: int) -> list[Hit]:
    q = _query_terms(question)
    df: Counter[str] = Counter()
    for hit in hits:
        df.update(set(tokenize(f"{hit.title} {hit.text}")))
    ranked: list[Hit] = []
    for hit in hits:
        blob = f"{hit.title} {hit.text}"
        terms = set(tokenize(blob))
        heading = hit.text.split("\n", 1)[0]
        heading_cov = (len(q & set(tokenize(heading))) / len(q)) if q else 0.0
        overlap = (len(q & terms) / len(q)) if q else 0.0
        idf = 0.0
        if q:
            idf = sum(1.0 / (1.0 + df[t]) for t in q if t in terms) / len(q)
        score = 0.40 * hit.score + 0.25 * overlap + 0.20 * heading_cov + 0.15 * idf
        ranked.append(
            Hit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                source=hit.source,
                title=hit.title,
                score=score,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:k]


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
    raw_hits = store.query(query_vec, k=max(k * 3, k))
    hits = hybrid_rerank(question, raw_hits, k)
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
