import numpy as np

from app.rag.chunking import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.store import SqliteVectorStore


def test_sqlite_roundtrip(tmp_path):
    store = SqliteVectorStore(str(tmp_path / "rag.sqlite3"))
    chunks = [
        Chunk("a::0", "Employees receive 20 days of paid time off.", "pto.md", "PTO", 0),
        Chunk("b::0", "AlertMesh posts JSON to a webhook.", "faq.md", "FAQ", 0),
    ]
    emb = HashingEmbedder(dim=64)
    store.replace_all(chunks, emb.embed([c.text for c in chunks]))
    assert store.count() == 2
    query = emb.embed(["How many PTO days?"])[0]
    hits = store.query(query, k=1)
    assert hits
    assert hits[0].source == "pto.md"
    store.close()


def test_replace_all_clears_previous(tmp_path):
    store = SqliteVectorStore(str(tmp_path / "rag.sqlite3"))
    emb = HashingEmbedder(dim=32)
    first = [Chunk("a::0", "alpha", "a.md", "A", 0)]
    store.replace_all(first, emb.embed(["alpha"]))
    second = [Chunk("b::0", "beta", "b.md", "B", 0)]
    store.replace_all(second, emb.embed(["beta"]))
    assert store.count() == 1
    hits = store.query(emb.embed(["beta"])[0], k=2)
    assert hits[0].source == "b.md"


def test_empty_query(tmp_path):
    store = SqliteVectorStore(str(tmp_path / "rag.sqlite3"))
    assert store.query(np.zeros(8, dtype=np.float32), k=3) == []
    store.close()
