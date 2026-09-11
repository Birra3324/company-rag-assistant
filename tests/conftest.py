"""Test env must be set before app imports so settings pick up temp store + no auto-ingest."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rag-test-"))
_STORE = _TMP / "rag.sqlite3"
_CHROMA = _TMP / "chroma"

os.environ["APP_ENV"] = "test"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LLM_PROVIDER"] = "extractive"
os.environ["OPENAI_API_KEY"] = ""
os.environ["VECTOR_BACKEND"] = "sqlite"
os.environ["VECTOR_STORE_PATH"] = str(_STORE)
os.environ["CHROMA_PATH"] = str(_CHROMA)
os.environ["DOCS_PATH"] = str(Path(__file__).resolve().parents[1] / "data" / "sample_docs")
os.environ["AUTO_INGEST_ON_STARTUP"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import reset_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.rag.embeddings import HashingEmbedder  # noqa: E402
from app.rag.store import SqliteVectorStore, reset_store  # noqa: E402

reset_settings()
reset_store()


class MockEmbedder:
    """Deterministic stand-in so CI never loads GPU models or calls OpenAI."""

    name = "mock"
    dim = 32

    def embed(self, texts: list[str]) -> np.ndarray:
        inner = HashingEmbedder(dim=self.dim)
        return inner.embed(texts)


class MockGenerator:
    name = "mock"

    def generate(self, question: str, hits) -> str:
        if not hits:
            return "NO_HITS"
        return f"MOCK:{question}:{hits[0].source}"


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def mock_generator() -> MockGenerator:
    return MockGenerator()


@pytest.fixture
def store(tmp_path: Path) -> SqliteVectorStore:
    reset_store()
    path = tmp_path / "test.sqlite3"
    return SqliteVectorStore(str(path))


def _wipe_default_store() -> None:
    reset_store()
    if _STORE.exists():
        _STORE.unlink()
    _STORE.parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def client():
    reset_settings()
    _wipe_default_store()
    with TestClient(app) as c:
        yield c
    _wipe_default_store()


@pytest.fixture
def openai_http_blocker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fail the test if RAG code opens an HTTP client (OpenAI/Ollama)."""

    blocked = MagicMock(side_effect=AssertionError("OpenAI HTTP was invoked in tests"))

    class BlockedClient:
        def __init__(self, *args, **kwargs) -> None:
            blocked(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, *args, **kwargs):
            return blocked(*args, **kwargs)

        def get(self, *args, **kwargs):
            return blocked(*args, **kwargs)

    monkeypatch.setattr("app.rag.generate.httpx.Client", BlockedClient)
    monkeypatch.setattr("app.rag.embeddings.httpx.Client", BlockedClient)
    return blocked
