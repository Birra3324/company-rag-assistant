from app.rag.retrieve import bm25_scores, hybrid_blend, query_terms
from app.rag.store import Hit


def _hit(chunk_id: str, text: str, source: str, title: str, score: float, heading: str = "") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=text,
        source=source,
        title=title,
        score=score,
        heading=heading,
    )


def test_query_terms_drop_stopwords():
    terms = query_terms("How many PTO days do employees receive?")
    assert "how" not in terms
    assert "pto" in terms
    assert "days" in terms


def test_bm25_ranks_distinctive_term_first():
    docs = [
        ["employees", "receive", "stipend", "monthly"],
        ["employees", "receive", "20", "pto", "days", "accrued"],
        ["webhook", "alertmesh", "json"],
    ]
    scores = bm25_scores(["pto", "days"], docs)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_hybrid_blend_prefers_bm25_when_dense_is_wrong():
    pto = _hit(
        "pto::0",
        "## PTO days\n\nFull-time employees receive 20 PTO days of paid time off.",
        "02-pto-policy.md",
        "Time Off and Leave Policy",
        score=0.05,
        heading="PTO days",
    )
    other = _hit(
        "sec::0",
        "## Vendors\n\nThird-party tools need a short security review.",
        "03-it-security.md",
        "IT Security and Acceptable Use",
        score=0.9,
        heading="Vendors",
    )
    ranked = hybrid_blend("How many PTO days do employees receive?", [other, pto], [pto, other], k=2)
    assert ranked[0].source == "02-pto-policy.md"
    assert ranked[0].chunk_id == "pto::0"
