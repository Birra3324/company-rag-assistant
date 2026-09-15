from app.rag.chunking import chunk_text, load_documents, stable_title, title_from_text, token_count


def test_title_from_heading():
    from pathlib import Path

    title = title_from_text(Path("x.md"), "# Hello World\n\nBody")
    assert title == "Hello World"


def test_stable_title_falls_back_to_stem():
    assert stable_title("", "02-pto-policy.md") == "02 Pto Policy"
    assert stable_title("Time Off and Leave Policy", "02-pto-policy.md") == (
        "Time Off and Leave Policy"
    )


def test_chunk_splits_markdown_headings():
    text = "# Title\n\nIntro paragraph.\n\n## PTO days\n\n20 PTO days.\n\n## Sick leave\n\n8 sick days."
    chunks = chunk_text(text, source="p.md", title="Policy", chunk_size=180, overlap=20)
    bodies = " ".join(c.text for c in chunks)
    assert any("PTO days" in c.text and "Sick leave" not in c.text for c in chunks)
    assert "20 PTO days" in bodies
    assert any(c.heading == "PTO days" for c in chunks)
    assert all("::" in c.chunk_id for c in chunks)


def test_chunk_size_is_token_budget():
    words = " ".join(f"token{i:03d}" for i in range(90))
    text = f"# Heading\n\n{words}"
    chunks = chunk_text(text, source="s.md", title="T", chunk_size=40, overlap=8)
    assert len(chunks) >= 2
    # Heading tokens are included; body of each piece stays near the budget.
    for chunk in chunks:
        assert token_count(chunk.text) <= 55
        assert chunk.text.lstrip().startswith("#") or "Heading" in chunk.text
    # Overlap: some tokens from the end of chunk 0 appear in chunk 1.
    first_tokens = set(__import__("re").findall(r"token\d+", chunks[0].text))
    second_tokens = set(__import__("re").findall(r"token\d+", chunks[1].text))
    assert first_tokens & second_tokens


def test_chunk_respects_source_and_title():
    text = "Paragraph one.\n\n" + ("word " * 80) + "\n\nParagraph three."
    chunks = chunk_text(text, source="s.md", title="T", chunk_size=40, overlap=8)
    assert chunks
    assert all(c.source == "s.md" for c in chunks)
    assert all(c.title == "T" for c in chunks)


def test_load_sample_docs():
    docs = load_documents("data/sample_docs")
    names = {p.name for p, _, _ in docs}
    assert len(docs) >= 5
    assert "02-pto-policy.md" in names
    assert "04-support-faq.md" in names
