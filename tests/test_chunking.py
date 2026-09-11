from pathlib import Path

from app.rag.chunking import chunk_text, load_documents, title_from_text


def test_title_from_heading():
    title = title_from_text(Path("x.md"), "# Hello World\n\nBody")
    assert title == "Hello World"


def test_chunk_splits_markdown_headings():
    text = "# Title\n\nIntro paragraph.\n\n## PTO days\n\n20 PTO days.\n\n## Sick leave\n\n8 sick days."
    chunks = chunk_text(text, source="p.md", title="Policy", chunk_size=700, overlap=20)
    bodies = " ".join(c.text for c in chunks)
    assert any("PTO days" in c.text and "Sick leave" not in c.text for c in chunks)
    assert "20 PTO days" in bodies


def test_chunk_respects_size():
    text = "Paragraph one.\n\n" + ("word " * 80) + "\n\nParagraph three."
    chunks = chunk_text(text, source="s.md", title="T", chunk_size=120, overlap=20)
    assert chunks
    assert all(c.source == "s.md" for c in chunks)
    assert all(c.title == "T" for c in chunks)


def test_load_sample_docs():
    docs = load_documents("data/sample_docs")
    names = {p.name for p, _, _ in docs}
    assert len(docs) >= 5
    assert "02-pto-policy.md" in names
    assert "04-support-faq.md" in names
