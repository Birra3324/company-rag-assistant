from app.rag.eval_harness import run_eval


def test_eval_harness_meets_threshold(store):
    report = run_eval(store=store, threshold=0.75)
    assert report.total >= 10
    assert report.embedding_provider == "local"
    assert report.llm_provider == "extractive"
    assert report.ok, (
        f"pass_rate={report.pass_rate:.2f} "
        + ", ".join(c.id for c in report.cases if not c.passed)
    )
    assert report.pass_rate >= 0.75


def test_eval_harness_scores_expected_source():
    from app.rag.eval_harness import score_case

    scored = score_case(
        {
            "id": "pto-days",
            "question": "How many PTO days?",
            "expected_sources": ["02-pto-policy.md"],
            "expected_keywords": ["20"],
        },
        answer="Employees receive 20 PTO days.\n\nSources: [1] Time Off and Leave Policy.",
        sources=["02-pto-policy.md", "01-company-overview.md"],
        excerpts=["## PTO days"],
    )
    assert scored.passed
    assert scored.retrieval_ok
    assert scored.keyword_ok


def test_eval_source_list_is_or():
    from app.rag.eval_harness import score_case

    scored = score_case(
        {
            "id": "p2-sla",
            "question": "P2 time?",
            "expected_sources": ["04-support-faq.md", "01-company-overview.md"],
            "expected_keywords": ["4"],
        },
        answer="P2 is 4 business hours.",
        sources=["01-company-overview.md"],
        excerpts=["P2 (degraded evaluator runs): first response in 4 business hours"],
    )
    assert scored.retrieval_ok
    assert scored.passed
