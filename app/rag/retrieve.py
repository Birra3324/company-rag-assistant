"""Hybrid retrieval: dense cosine + BM25 fused without GPU or API keys.

The Day 11 pipeline only reranked the dense shortlist with keyword overlap.
Hashing embeddings are weak on paraphrases, so Days 12–14 score the **full**
corpus with BM25 and fuse ranks (weighted reciprocal rank fusion). The
default path stays local: numpy + stdlib, no ranker model.
"""

from __future__ import annotations

import math
from collections import Counter

from app.rag.embeddings import QUERY_STOP, tokenize
from app.rag.store import Hit

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60.0

# BM25 is weighted slightly above dense because the default embedder is hashing.
DENSE_RRF_WEIGHT = 1.0
BM25_RRF_WEIGHT = 1.3
HEADING_RRF_WEIGHT = 0.75

DISPLAY_DENSE = 0.35
DISPLAY_BM25 = 0.45
DISPLAY_OVERLAP = 0.20


def query_terms(question: str) -> list[str]:
    return [t for t in tokenize(question) if t not in QUERY_STOP and len(t) > 1]


def bm25_scores(query: list[str], docs_tokens: list[list[str]]) -> list[float]:
    """Okapi BM25 over an in-memory corpus (small FAQ collections)."""
    n_docs = len(docs_tokens)
    if n_docs == 0:
        return []
    if not query:
        return [0.0] * n_docs
    avgdl = sum(len(doc) for doc in docs_tokens) / n_docs
    df: Counter[str] = Counter()
    for doc in docs_tokens:
        df.update(set(doc))
    scores: list[float] = []
    for doc in docs_tokens:
        tf = Counter(doc)
        dl = len(doc) or 1
        score = 0.0
        for term in query:
            freq = tf[term]
            if freq == 0:
                continue
            n_q = df[term]
            idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
            denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / (avgdl or 1.0))
            score += idf * (freq * (BM25_K1 + 1.0)) / denom
        scores.append(score)
    return scores


def _ranks(scores: list[float]) -> list[int]:
    """1-based ranks; higher score is better. Stable for ties."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _overlap(question_terms: set[str], hit: Hit) -> float:
    if not question_terms:
        return 0.0
    doc_terms = set(tokenize(f"{hit.title} {hit.heading} {hit.text}"))
    return len(question_terms & doc_terms) / len(question_terms)


def _heading_coverage(question_terms: set[str], hit: Hit) -> float:
    if not question_terms:
        return 0.0
    heading = hit.heading or hit.text.split("\n", 1)[0]
    blob = set(tokenize(f"{hit.title} {heading}"))
    return len(question_terms & blob) / len(question_terms)


def hybrid_blend(
    question: str,
    dense_hits: list[Hit],
    corpus: list[Hit],
    k: int,
) -> list[Hit]:
    """Fuse dense cosine ranks with BM25 and heading-keyword ranks.

    ``dense_hits`` carry cosine scores. ``corpus`` is every indexed chunk
    (typically ``store.list_chunks()``). Ranking uses weighted RRF; the
    returned ``Hit.score`` is a 0–1 blend so ``min_retrieve_score`` still
    makes sense.
    """
    if not corpus:
        corpus = list(dense_hits)
    if not corpus:
        return []

    dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
    terms = query_terms(question)
    term_set = set(terms)
    docs_tokens = [tokenize(f"{hit.title} {hit.heading} {hit.text}") for hit in corpus]
    bm25 = bm25_scores(terms, docs_tokens)
    max_bm25 = max(bm25) if bm25 else 0.0

    dense_scores = [
        dense_by_id[hit.chunk_id].score if hit.chunk_id in dense_by_id else -1.0
        for hit in corpus
    ]
    heading_scores = [_heading_coverage(term_set, hit) for hit in corpus]
    overlaps = [_overlap(term_set, hit) for hit in corpus]

    dense_ranks = _ranks(dense_scores)
    bm25_ranks = _ranks(bm25)
    heading_ranks = _ranks(heading_scores)

    ranked: list[tuple[float, float, Hit]] = []
    for i, hit in enumerate(corpus):
        rrf = (
            DENSE_RRF_WEIGHT / (RRF_K + dense_ranks[i])
            + BM25_RRF_WEIGHT / (RRF_K + bm25_ranks[i])
            + HEADING_RRF_WEIGHT / (RRF_K + heading_ranks[i])
        )
        cosine = max(dense_by_id[hit.chunk_id].score, 0.0) if hit.chunk_id in dense_by_id else 0.0
        bm25_norm = (bm25[i] / max_bm25) if max_bm25 > 0 else 0.0
        display = (
            DISPLAY_DENSE * cosine
            + DISPLAY_BM25 * bm25_norm
            + DISPLAY_OVERLAP * overlaps[i]
        )
        ranked.append((rrf, min(display, 1.0), hit))

    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[Hit] = []
    for _, display, hit in ranked[: max(k, 1)]:
        out.append(
            Hit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                source=hit.source,
                title=hit.title,
                score=float(display),
                heading=hit.heading,
            )
        )
    return out
