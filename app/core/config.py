"""Environment-driven settings. Keys never live in source."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Company RAG Knowledge Assistant"
    app_env: str = "development"
    log_level: str = "INFO"

    # local | sentence-transformers | openai
    embedding_provider: Literal["local", "sentence-transformers", "openai"] = "local"
    embedding_dim: int = 384
    sentence_transformers_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # extractive | openai | ollama
    llm_provider: Literal["extractive", "openai", "ollama"] = "extractive"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_base_url: str = "https://api.openai.com/v1"
    ai_timeout_seconds: float = 45.0

    docs_path: str = "data/sample_docs"
    vector_store_path: str = "data/vectorstore/rag.sqlite3"
    chroma_path: str = "data/vectorstore/chroma"
    # sqlite | chroma — sqlite is the default offline-friendly path
    vector_backend: Literal["sqlite", "chroma"] = "sqlite"

    chunk_size: int = 700
    chunk_overlap: int = 120
    retrieve_k: int = 4
    min_retrieve_score: float = 0.08
    auto_ingest_on_startup: bool = True

    @property
    def is_test(self) -> bool:
        return self.app_env.lower() in {"test", "testing"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> Settings:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()
    return get_settings()
