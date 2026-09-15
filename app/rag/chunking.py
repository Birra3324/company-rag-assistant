"""Split markdown/text files into overlapping, header-aware chunks.

CHUNK_SIZE / CHUNK_OVERLAP semantics
------------------------------------
Both settings are measured in **approximate word tokens**, not characters.

A token is an alphanumeric sequence ``[a-z0-9]+`` (the same tokenizer the
local hashing embedder uses). English prose is roughly 4 characters per
token, so Day 11's old 700-character window is about **180 tokens**.

- ``CHUNK_SIZE`` (default 180): target tokens **per chunk**, including the
  section heading that is prepended so a split FAQ answer still retrieves.
- ``CHUNK_OVERLAP`` (default 40): tokens copied from the end of one chunk
  onto the start of the next so a sentence that straddles a boundary is
  still findable.

Chunks never start mid-heading. Long sections split on paragraphs, then
sentences, then token windows — never a naive character slice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DOC_SUFFIXES = {".md", ".markdown", ".txt"}

# Same definition the hashing embedder uses so chunk budgets match retrieval.
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    index: int
    heading: str = ""


def token_count(text: str) -> int:
    """Count alphanumeric tokens in ``text`` (CHUNK_SIZE units)."""
    return len(_TOKEN.findall(text))


def stable_title(title: str, source: str) -> str:
    """Document title for citations: H1 when present, else a clean stem."""
    cleaned = (title or "").strip()
    if cleaned:
        return cleaned
    stem = Path(source).stem
    return stem.replace("-", " ").replace("_", " ").title() or "Untitled"


def title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
        if stripped and not stripped.startswith("_"):
            return stripped[:120]
    return path.stem.replace("-", " ").title()


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def _attach_heading(heading_line: str, body: str) -> str:
    body = body.strip()
    if not heading_line:
        return body
    if body.lstrip().startswith("#"):
        return body
    return f"{heading_line}\n\n{body}".strip()


def _split_sentences(text: str) -> list[str]:
    """Keep markdown structure; split prose on sentence boundaries."""
    out: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("#", "|", "-", "*")):
            out.append(line)
            continue
        parts = [p.strip() for p in _SENTENCE.split(line) if p.strip()]
        out.extend(parts or [line])
    return out


def _units(text: str, chunk_size: int) -> list[str]:
    """Paragraphs as pack units; long prose is further split into sentences."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    units: list[str] = []
    soft_limit = max(chunk_size // 2, 24)
    for para in paragraphs:
        if token_count(para) <= soft_limit or para.lstrip().startswith(("#", "|")):
            units.append(para)
            continue
        units.extend(_split_sentences(para) or [para])
    return units


def _token_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    matches = list(_TOKEN.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []
    windows: list[str] = []
    start_i = 0
    n = len(matches)
    size = max(chunk_size, 1)
    ov = min(max(overlap, 0), size - 1) if size > 1 else 0
    while start_i < n:
        end_i = min(start_i + size, n)
        piece = text[matches[start_i].start() : matches[end_i - 1].end()].strip()
        if piece:
            windows.append(piece)
        if end_i >= n:
            break
        start_i = max(end_i - ov, start_i + 1)
    return windows


def _pack_units(
    units: list[str],
    *,
    heading_line: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    if not units:
        return [heading_line] if heading_line else []

    heading_tokens = token_count(heading_line)
    budget = max(chunk_size - heading_tokens, 8)
    pieces: list[str] = []
    start = 0
    n = len(units)

    while start < n:
        token_sum = 0
        end = start
        while end < n:
            unit_tokens = token_count(units[end])
            if unit_tokens > budget and end == start:
                for window in _token_windows(units[end], budget, overlap):
                    packed = _attach_heading(heading_line, window)
                    if packed:
                        pieces.append(packed)
                start = end + 1
                end = start
                token_sum = 0
                break
            if end > start and token_sum + unit_tokens > budget:
                break
            token_sum += unit_tokens
            end += 1
            if token_sum >= budget:
                break
        if end == start:
            continue
        body = "\n\n".join(units[start:end]).strip()
        packed = _attach_heading(heading_line, body)
        if packed:
            pieces.append(packed)
        if end >= n:
            break
        next_start = end
        acc = 0
        for i in range(end - 1, start, -1):
            acc += token_count(units[i])
            next_start = i
            if acc >= overlap:
                break
        start = max(next_start, start + 1)

    return pieces


def _sections(cleaned: str) -> list[tuple[str, str, str]]:
    """Return (heading_text, heading_line, section_body) tuples."""
    parts = re.split(r"(?m)(?=^#{1,6}\s+)", cleaned)
    sections: list[tuple[str, str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first, _, rest = part.partition("\n")
        match = _HEADING_LINE.match(first.strip())
        if match:
            heading = match.group(2).strip()
            heading_line = first.strip()
            body = rest.strip()
            sections.append((heading, heading_line, body))
        else:
            sections.append(("", "", _normalize(part)))
    return sections


def chunk_text(
    text: str,
    *,
    source: str,
    title: str,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[Chunk]:
    """Header-aware splitter with token windows and overlap.

    ``chunk_size`` / ``overlap`` are token counts (see module docstring).
    """
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return []

    chunk_size = max(int(chunk_size), 8)
    overlap = max(int(overlap), 0)
    if overlap >= chunk_size:
        overlap = max(chunk_size // 4, 1)

    doc_title = stable_title(title, source)
    pieces: list[tuple[str, str]] = []  # (heading, text)
    for heading, heading_line, body in _sections(cleaned):
        section_heading = heading or doc_title
        units = _units(body, chunk_size) if body else []
        packed = _pack_units(
            units,
            heading_line=heading_line,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not packed and heading_line:
            packed = [heading_line]
        for piece in packed:
            if piece:
                pieces.append((section_heading, piece))

    chunks: list[Chunk] = []
    for i, (heading, piece) in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=f"{source}::{i:04d}",
                text=piece,
                source=source,
                title=doc_title,
                index=i,
                heading=heading,
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
