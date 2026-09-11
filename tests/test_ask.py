from unittest.mock import MagicMock

import pytest

from app.core.config import get_settings
from app.rag.generate import ExtractiveGenerator, OpenAIGenerator
from app.rag.ingest import ingest_directory
from app.rag.pipeline import ask
from app.rag.store import Hit, reset_store


def _ingest(client):
    reset_store()
    r = client.post("/ingest")
    assert r.status_code == 200
    return r.json()


def test_ask_without_docs_is_409(client):
    reset_store()
    r = client.post("/ask", json={"question": "How many PTO days?"})
    assert r.status_code == 409
    assert "ingest" in r.json()["error"].lower()


def test_ask_pto_from_sample_docs(client):
    _ingest(client)
    r = client.post("/ask", json={"question": "How many PTO days do employees receive?"})
    assert r.status_code == 200
    body = r.json()
    assert body["retrieved"] >= 1
    assert body["embedding_provider"] == "local"
    assert body["llm_provider"] == "extractive"
    blob = (body["answer"] + " " + " ".join(s["excerpt"] for s in body["sources"])).lower()
    assert "20" in blob
    assert any("pto" in s["source"] or "pto" in s["title"].lower() for s in body["sources"])


def test_query_alias(client):
    _ingest(client)
    r = client.post("/query", json={"question": "What is the home office stipend?"})
    assert r.status_code == 200
    blob = r.json()["answer"].lower()
    assert "150" in blob


def test_ask_p1_sla(client):
    _ingest(client)
    r = client.post("/ask", json={"question": "What is the P1 support response time?"})
    assert r.status_code == 200
    blob = r.json()["answer"].lower()
    assert "30" in blob
    assert any("support" in s["source"] or "faq" in s["source"] for s in r.json()["sources"])


def test_ask_eu_residency(client):
    _ingest(client)
    r = client.post("/ask", json={"question": "Can we keep TraceLight data in the EU?"})
    assert r.status_code == 200
    blob = r.json()["answer"].lower()
    assert "eu" in blob


def test_ask_validation(client):
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 422


def test_pipeline_uses_mocked_embeddings_and_llm(store, mock_embedder, mock_generator):
    settings = get_settings()
    ingest_directory(settings.docs_path, settings=settings, embedder=mock_embedder, store=store)
    result = ask(
        "How many sick days?",
        settings=settings,
        embedder=mock_embedder,
        store=store,
        generator=mock_generator,
    )
    assert result.embedding_provider == "mock"
    assert result.llm_provider == "mock"
    assert result.answer.startswith("MOCK:")


def test_extractive_handles_empty_hits():
    text = ExtractiveGenerator().generate("anything?", [])
    assert "ingest" in text.lower()


def test_openai_generator_requires_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIGenerator(api_key="", model="gpt-4o-mini", base_url="https://api.openai.com/v1", timeout=5)


def test_default_ask_does_not_call_openai(client, openai_http_blocker: MagicMock):
    _ingest(client)
    r = client.post("/ask", json={"question": "What is TraceLight?"})
    assert r.status_code == 200
    assert "tracelight" in r.json()["answer"].lower()
    openai_http_blocker.assert_not_called()


def test_openai_generator_posts_chat_completions(monkeypatch):
    hits = [
        Hit(
            chunk_id="1",
            text="TraceLight is the fictional tracing product.",
            source="01.md",
            title="Overview",
            score=0.9,
        )
    ]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"choices": [{"message": {"content": "From context: TraceLight."}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.posted = []

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, headers=None, json=None):
            self.posted.append((url, headers, json))
            return FakeResponse()

    fake = FakeClient()
    monkeypatch.setattr("app.rag.generate.httpx.Client", lambda *a, **k: fake)
    gen = OpenAIGenerator(
        api_key="sk-test-not-real",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        timeout=5,
    )
    out = gen.generate("What is TraceLight?", hits)
    assert out == "From context: TraceLight."
    assert fake.posted[0][0].endswith("/chat/completions")
