"""Embedding providers: local hashing (default), sentence-transformers, OpenAI."""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import httpx
import numpy as np

from app.core.config import Settings, get_settings

log = logging.getLogger("rag.embeddings")

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
QUERY_STOP = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "many",
    "of",
    "or",
    "the",
    "to",
    "we",
    "what",
    "where",
    "who",
}


class Embedder(ABC):
    """Turn text into L2-normalized float32 vectors."""

    name: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder(Embedder):
    """Signed hashing-trick embeddings. Offline, no model download, no API key.

    Quality is lower than MiniLM / OpenAI — enough for a small FAQ corpus
    when combined with keyword overlap. Swap via EMBEDDING_PROVIDER.
    """

    name = "local"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            grams: list[str] = []
            for i, tok in enumerate(tokens):
                grams.append(tok)
                if i + 1 < len(tokens):
                    grams.append(f"{tok}_{tokens[i + 1]}")
            if not grams:
                continue
            for gram in grams:
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, idx] += sign
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEmbedder(Embedder):
    """Optional local transformer embeddings. Requires sentence-transformers extra."""

    name = "sentence-transformers"

    def __init__(self, model_name: str, dim: int = 384) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised via unit test mock
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "pip install -r requirements-ml.txt"
            ) from exc
        log.info("loading sentence-transformers model %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension() or dim)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder(Embedder):
    """Optional OpenAI embeddings. Requires OPENAI_API_KEY."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float,
        dim: int = 1536,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "input": texts}
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()["data"]
        ordered = sorted(data, key=lambda item: item["index"])
        matrix = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    provider = settings.embedding_provider
    if provider == "local":
        return HashingEmbedder(dim=settings.embedding_dim)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(
            settings.sentence_transformers_model,
            dim=settings.embedding_dim,
        )
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
            timeout=settings.ai_timeout_seconds,
        )
    raise ValueError(f"unknown embedding provider: {provider}")
