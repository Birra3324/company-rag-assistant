def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["embedding_provider"] == "local"
    assert body["llm_provider"] == "extractive"
    assert body["vector_backend"] == "sqlite"
    assert "x-request-id" in r.headers


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "RAG" in body["service"]
    assert body["ask"] == "POST /ask"


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
