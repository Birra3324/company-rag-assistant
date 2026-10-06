import sys
import types

import numpy as np
import pytest

from app.core.config import Settings
from app.rag.embeddings import HashingEmbedder, SentenceTransformerEmbedder, get_embedder


def test_sentence_transformers_missing_extra(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(RuntimeError, match="requirements-ml"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")


def test_minilm_embed_normalizes_without_downloading(monkeypatch):
    """Optional MiniLM path: fake the library so CI never fetches weights."""

    class FakeModel:
        def get_sentence_embedding_dimension(self):
            return 8

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
            assert convert_to_numpy is True
            assert normalize_embeddings is True
            assert show_progress_bar is False
            matrix = np.arange(len(texts) * 8, dtype=np.float32).reshape(len(texts), 8)
            return matrix

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda name: FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    vectors = embedder.embed(["pto policy", "remote stipend"])
    assert embedder.name == "sentence-transformers"
    assert embedder.dim == 8
    assert vectors.shape == (2, 8)
    assert vectors.dtype == np.float32


def test_get_embedder_selects_minilm_when_configured(monkeypatch):
    sentinel = object()

    def factory(model_name, dim=384):
        assert model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert dim == 384
        return sentinel

    monkeypatch.setattr("app.rag.embeddings.SentenceTransformerEmbedder", factory)
    settings = Settings(
        embedding_provider="sentence-transformers",
        sentence_transformers_model="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert get_embedder(settings) is sentinel


def test_default_provider_stays_hashing_without_minilm():
    embedder = get_embedder(Settings(embedding_provider="local"))
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.name == "local"
