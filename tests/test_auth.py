from fastapi.testclient import TestClient
from app.main import app
from app.core.config import get_settings
import pytest


def test_data_routes_require_key(client):
    client.headers.pop("X-API-Key", None)
    for path, body in [("/ingest",{}),("/ask",{"question":"What is PTO?"}),("/query",{"question":"What is PTO?"})]:
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"X-API-Key":"wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_missing_config_fails_closed(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "")
    assert client.post("/ingest").status_code == 503
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(app):
            pass
