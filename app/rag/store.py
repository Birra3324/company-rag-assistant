"""Vector stores: SQLite+numpy (default) or embedded Chroma."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings, get_settings
from app.rag.chunking import Chunk

log = logging.getLogger("rag.store")

_store: "VectorStore | None" = None


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    source: str
    title: str
    score: float
    heading: str = ""


class VectorStore(Protocol):
    backend: str

    def replace_all(self, chunks: list[Chunk], embeddings: np.ndarray) -> int: ...

    def query(self, vector: np.ndarray, k: int) -> list[Hit]: ...

    def list_chunks(self) -> list[Hit]: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class SqliteVectorStore:
    """Embedded cosine index: one SQLite file, numpy blobs, no extra process.

    Day 11 default so clone → ingest → ask works offline. The add/query
    surface is small enough to swap for Chroma or FAISS later.
    """

    backend = "sqlite"

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                meta TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.commit()

    def replace_all(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._conn.execute("DELETE FROM chunks")
        rows = []
        for chunk, vector in zip(chunks, embeddings, strict=True):
            vec = np.asarray(vector, dtype=np.float32)
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.source,
                    chunk.title,
                    vec.tobytes(),
                    int(vec.shape[0]),
                    json.dumps({"index": chunk.index, "heading": chunk.heading}),
                )
            )
        self._conn.executemany(
            "INSERT INTO chunks (chunk_id, text, source, title, embedding, dim, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def query(self, vector: np.ndarray, k: int) -> list[Hit]:
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        qn = np.linalg.norm(query)
        if qn == 0:
            return []
        query = query / qn
        cur = self._conn.execute(
            "SELECT chunk_id, text, source, title, embedding, dim, meta FROM chunks"
        )
        scored: list[Hit] = []
        for chunk_id, text, source, title, blob, dim, meta in cur.fetchall():
            stored = np.frombuffer(blob, dtype=np.float32)
            if stored.size != dim or stored.size != query.size:
                continue
            denom = np.linalg.norm(stored)
            if denom == 0:
                continue
            score = float(np.dot(query, stored / denom))
            heading = ""
            try:
                heading = str(json.loads(meta or "{}").get("heading") or "")
            except json.JSONDecodeError:
                heading = ""
            scored.append(
                Hit(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    title=title,
                    score=score,
                    heading=heading,
                )
            )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:k]

    def list_chunks(self) -> list[Hit]:
        """All chunks without vectors — used for BM25 over the full corpus."""
        cur = self._conn.execute(
            "SELECT chunk_id, text, source, title, meta FROM chunks"
        )
        hits: list[Hit] = []
        for chunk_id, text, source, title, meta in cur.fetchall():
            heading = ""
            try:
                heading = str(json.loads(meta or "{}").get("heading") or "")
            except json.JSONDecodeError:
                heading = ""
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    title=title,
                    score=0.0,
                    heading=heading,
                )
            )
        return hits

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()


class ChromaVectorStore:
    """Embedded Chroma PersistentClient. No separate server.

    Used when VECTOR_BACKEND=chroma. Embeddings are passed in so Chroma
    never downloads its default ONNX model.
    """

    backend = "chroma"
    collection_name = "company_docs"

    def __init__(self, path: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_all(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if not chunks:
            return 0
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=np.asarray(embeddings, dtype=np.float32).tolist(),
            metadatas=[
                {
                    "source": c.source,
                    "title": c.title,
                    "index": c.index,
                    "heading": c.heading,
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def query(self, vector: np.ndarray, k: int) -> list[Hit]:
        n = self.count()
        if n == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=min(k, n),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            meta = meta or {}
            # Chroma cosine distance ≈ 1 - cosine similarity
            score = 1.0 - float(dist)
            hits.append(
                Hit(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    score=score,
                    heading=str(meta.get("heading", "")),
                )
            )
        return hits

    def list_chunks(self) -> list[Hit]:
        n = self.count()
        if n == 0:
            return []
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        hits: list[Hit] = []
        for chunk_id, text, meta in zip(ids, docs, metas, strict=False):
            meta = meta or {}
            hits.append(
                Hit(
                    chunk_id=str(chunk_id),
                    text=str(text),
                    source=str(meta.get("source", "")),
                    title=str(meta.get("title", "")),
                    score=0.0,
                    heading=str(meta.get("heading", "")),
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def close(self) -> None:
        return None


def get_store(settings: Settings | None = None) -> VectorStore:
    global _store
    if _store is None:
        settings = settings or get_settings()
        if settings.vector_backend == "chroma":
            _store = ChromaVectorStore(settings.chroma_path)
        else:
            _store = SqliteVectorStore(settings.vector_store_path)
        log.info("vector store backend=%s", _store.backend)
    return _store


def reset_store() -> None:
    global _store
    if _store is not None:
        _store.close()
    _store = None
