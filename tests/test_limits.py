from app.core.config import get_settings
from app.core.limits import reset_rate_limits


def test_question_char_cap(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_question_chars", 10)
    response = client.post("/ask", json={"question": "hello world"})
    assert response.status_code == 422
    assert "10" in str(response.json())


def test_top_k_cap(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_top_k", 2)
    response = client.post("/ask", json={"question": "How many PTO days?", "top_k": 3})
    assert response.status_code == 422
    assert "top_k" in str(response.json())


def test_default_top_k_still_rejects_above_ceiling(client):
    response = client.post("/ask", json={"question": "How many PTO days?", "top_k": 13})
    assert response.status_code == 422


def test_body_size_limit(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_body_bytes", 40)
    response = client.post(
        "/ask",
        json={"question": "How many PTO days do employees receive this year?"},
    )
    assert response.status_code == 413
    assert "bytes" in response.json()["error"]


def test_rate_limit_on_ask(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    reset_rate_limits()
    try:
        first = client.post("/ask", json={"question": "How many PTO days?"})
        second = client.post("/ask", json={"question": "How many PTO days?"})
        third = client.post("/ask", json={"question": "How many PTO days?"})
        assert first.status_code != 429
        assert second.status_code != 429
        assert third.status_code == 429
        assert third.headers["retry-after"]
        assert "rate" in third.json()["error"].lower()
    finally:
        reset_rate_limits()


def test_health_is_not_rate_limited(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 1)
    reset_rate_limits()
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/ui").status_code == 200
    reset_rate_limits()


def test_rate_limit_can_be_disabled(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 0)
    reset_rate_limits()
    for _ in range(3):
        response = client.post("/ask", json={"question": "How many PTO days?"})
        assert response.status_code != 429
    reset_rate_limits()
