from app.core.config import get_settings
from app.rag.ingest import ingest_directory
from app.rag.store import reset_store


def test_ingest_sample_docs(client):
    reset_store()
    r = client.post("/ingest")
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] >= 5
    assert body["chunks"] > 0
    assert body["embedding_provider"] == "local"

    health = client.get("/health").json()
    assert health["chunks"] == body["chunks"]


def test_ingest_directory_helper(store, mock_embedder):
    reset_store()
    settings = get_settings()
    result = ingest_directory(
        settings.docs_path,
        settings=settings,
        embedder=mock_embedder,
        store=store,
    )
    assert result.documents >= 5
    assert result.chunks == store.count()
    assert store.count() > 0
