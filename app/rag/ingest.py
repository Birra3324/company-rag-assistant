"""Load local markdown/text, chunk, embed, and write the vector store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from app.rag.chunking import chunk_text, load_documents
from app.rag.embeddings import Embedder, get_embedder
from app.rag.store import VectorStore, get_store

log = logging.getLogger("rag.ingest")


@dataclass(frozen=True)
class IngestResult:
    documents: int
    chunks: int
    docs_path: str


def ingest_directory(
    docs_path: str | Path | None = None,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> IngestResult:
    settings = settings or get_settings()
    root = Path(docs_path or settings.docs_path)
    embedder = embedder or get_embedder(settings)
    store = store or get_store(settings)

    documents = load_documents(root)
    chunks = []
    for path, title, text in documents:
        source = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        chunks.extend(
            chunk_text(
                text,
                source=source,
                title=title,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        )
    if not chunks:
        store.replace_all([], embedder.embed([]).reshape(0, embedder.dim))
        log.warning("ingest found no chunks under %s", root)
        return IngestResult(documents=len(documents), chunks=0, docs_path=str(root))

    embeddings = embedder.embed([c.text for c in chunks])
    stored = store.replace_all(chunks, embeddings)
    log.info(
        "ingested documents=%s chunks=%s embedder=%s path=%s",
        len(documents),
        stored,
        embedder.name,
        root,
    )
    return IngestResult(documents=len(documents), chunks=stored, docs_path=str(root))


def maybe_auto_ingest(
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> IngestResult | None:
    settings = settings or get_settings()
    if not settings.auto_ingest_on_startup or settings.is_test:
        return None
    store = store or get_store(settings)
    if store.count() > 0:
        return None
    return ingest_directory(settings=settings, embedder=embedder, store=store)


def main() -> None:
    result = ingest_directory()
    print(
        f"Ingested {result.documents} documents → {result.chunks} chunks "
        f"from {result.docs_path}"
    )


if __name__ == "__main__":
    main()
