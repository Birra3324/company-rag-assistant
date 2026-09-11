import sys
import types

import pytest

from app.rag.embeddings import SentenceTransformerEmbedder


def test_sentence_transformers_missing_extra(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(RuntimeError, match="requirements-ml"):
        SentenceTransformerEmbedder("all-MiniLM-L6-v2")
