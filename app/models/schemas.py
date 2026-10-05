"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError


class HealthOut(BaseModel):
    ok: bool
    status: str
    chunks: int
    embedding_provider: str
    llm_provider: str
    vector_backend: str
    auto_ingest: bool


class AskIn(BaseModel):
    """Question payload. Env caps (MAX_TOP_K, MAX_QUESTION_CHARS) can only tighten the absolute ceilings."""

    question: str = Field(min_length=3, max_length=8000)
    top_k: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def apply_demo_caps(self) -> "AskIn":
        from app.core.config import get_settings

        settings = get_settings()
        if len(self.question) > settings.max_question_chars:
            raise PydanticCustomError(
                "question_too_long",
                "question exceeds {limit} characters",
                {"limit": settings.max_question_chars},
            )
        if self.top_k is not None and self.top_k > settings.max_top_k:
            raise PydanticCustomError(
                "top_k_too_large",
                "top_k exceeds max of {limit}",
                {"limit": settings.max_top_k},
            )
        return self


class SourceOut(BaseModel):
    source: str
    title: str
    score: float
    excerpt: str
    chunk_id: str | None = None
    heading: str | None = None


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
    ui: str = "/ui"
    docs: str = "/docs"
    ask: str = "POST /ask"
    query: str = "POST /query"
    ingest: str = "POST /ingest"
