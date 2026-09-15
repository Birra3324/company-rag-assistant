import numpy as np
import pytest

from app.rag.embeddings import HashingEmbedder, OpenAIEmbedder, get_embedder


def test_hashing_embedder_is_normalized():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["Vision AI Ops PTO policy", "unrelated zucchini recipes"])
    assert vecs.shape == (2, 64)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_text_has_higher_cosine():
    emb = HashingEmbedder(dim=256)
    pto = "Employees receive 20 days of paid time off each calendar year."
    pto_q = "How many PTO days do employees get?"
    other = "AlertMesh posts evaluator JSON to a customer webhook URL."
    a, b, c = emb.embed([pto, pto_q, other])
    assert float(np.dot(a, b)) > float(np.dot(a, c))


def test_default_embedder_is_local():
    embedder = get_embedder()
    assert embedder.name == "local"


def test_openai_embedder_requires_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(
            api_key="",
            model="text-embedding-3-small",
            base_url="https://api.openai.com/v1",
            timeout=5.0,
        )
