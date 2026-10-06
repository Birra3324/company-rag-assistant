def test_ui_page_is_public_html(client):
    client.headers.pop("X-API-Key", None)
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert 'id="question"' in html
    assert 'id="api-key"' in html
    assert "Cited sources" in html
    assert "/ask" in html
    assert "test-rag-key" not in html


def test_root_points_at_ui(client):
    body = client.get("/").json()
    assert body["ui"] == "/ui"
    assert body["ask"] == "POST /ask"
