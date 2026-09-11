"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    ok: bool
    status: str
    chunks: int
    embedding_provider: str
    llm_provider: str
    vector_backend: str
    auto_ingest: bool


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=12)


class SourceOut(BaseModel):
    source: str
    title: str
    score: float
    excerpt: str


class AskOut(BaseModel):
    answer: str
    sources: list[SourceOut]
    retrieved: int
    embedding_provider: str
    llm_provider: str


class IngestOut(BaseModel):
    documents: int
    chunks: int
    embedding_provider: str
    docs_path: str


class RootOut(BaseModel):
    service: str
    version: str
    health: str = "/health"
    docs: str = "/docs"
    ask: str = "POST /ask"
    query: str = "POST /query"
    ingest: str = "POST /ingest"
