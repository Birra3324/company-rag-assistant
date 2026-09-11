"""Split markdown/text files into overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DOC_SUFFIXES = {".md", ".markdown", ".txt"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    index: int


_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_WHITESPACE = re.compile(r"[ \t]+")


def title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
        if stripped:
            return stripped[:120]
    return path.stem.replace("-", " ").title()


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    source: str,
    title: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Chunk]:
    """Header-aware splitter with character overlap.

    Day 11 scaffold: split on markdown headings first so FAQ sections
    stay separable, then pack paragraphs into ~chunk_size windows.
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    sections = re.split(r"(?m)(?=^#{1,6}\s+)", cleaned)
    pieces: list[str] = []
    for section in sections:
        section = _normalize(section)
        if not section:
            continue
        if len(section) <= chunk_size:
            pieces.append(section)
            continue
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", section) if p.strip()]
        buf = ""
        for para in paragraphs:
            if len(para) <= chunk_size:
                candidate = f"{buf}\n\n{para}".strip() if buf else para
                if len(candidate) <= chunk_size:
                    buf = candidate
                else:
                    if buf:
                        pieces.append(buf)
                    buf = para
            else:
                if buf:
                    pieces.append(buf)
                    buf = ""
                start = 0
                while start < len(para):
                    end = min(start + chunk_size, len(para))
                    pieces.append(para[start:end].strip())
                    if end >= len(para):
                        break
                    start = max(end - overlap, start + 1)
        if buf:
            pieces.append(buf)

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        if not piece:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{source}::{i}",
                text=piece,
                source=source,
                title=title,
                index=i,
            )
        )
    return chunks


def load_documents(docs_path: str | Path) -> list[tuple[Path, str, str]]:
    root = Path(docs_path)
    if not root.exists():
        raise FileNotFoundError(f"docs path not found: {root}")
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DOC_SUFFIXES
    )
    loaded: list[tuple[Path, str, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        title = title_from_text(path, text)
        loaded.append((path, title, text))
    return loaded
