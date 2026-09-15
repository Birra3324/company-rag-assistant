import os

import pytest

chromadb = pytest.importorskip("chromadb")

from app.rag.chunking import Chunk
from app.rag.embeddings import HashingEmbedder
from app.rag.store import ChromaVectorStore


def test_chroma_roundtrip(tmp_path):
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    store = ChromaVectorStore(str(tmp_path / "chroma"))
    chunks = [
        Chunk("a::0", "EU-West residency is available for TraceLight Cloud.", "faq.md", "FAQ", 0),
        Chunk("b::0", "The home office stipend is 150 dollars.", "remote.md", "Remote", 0),
    ]
    emb = HashingEmbedder(dim=64)
    store.replace_all(chunks, emb.embed([c.text for c in chunks]))
    assert store.count() == 2
    hits = store.query(emb.embed(["Where can we keep data in Europe?"])[0], k=1)
    assert hits
    assert hits[0].source == "faq.md"
    store.close()
