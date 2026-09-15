"""Offline eval harness for the sample Vision AI Ops corpus.

Scores retrieval (at least one expected source in top-k) and answer quality
(expected keywords in the answer or retrieved excerpts). No GPU, no API keys.

CLI::

    python -m app.rag.eval_harness
    python -m app.rag.eval_harness --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.config import Settings, get_settings
from app.rag.embeddings import Embedder, get_embedder
from app.rag.generate import Generator, get_generator
from app.rag.ingest import ingest_directory
from app.rag.pipeline import ask
from app.rag.store import VectorStore, get_store

DEFAULT_GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden.json"
DEFAULT_THRESHOLD = 0.75


@dataclass
class CaseScore:
    id: str
    question: str
    retrieval_ok: bool
    keyword_ok: bool
    passed: bool
    retrieval_recall: float
    keyword_hit_rate: float
    retrieved_sources: list[str]
    expected_sources: list[str]
    missing_keywords: list[str]


@dataclass
class EvalReport:
    total: int
    passed: int
    pass_rate: float
    threshold: float
    embedding_provider: str
    llm_provider: str
    cases: list[CaseScore] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.pass_rate + 1e-9 >= self.threshold


def load_golden(path: Path | str | None = None) -> dict:
    golden_path = Path(path) if path else DEFAULT_GOLDEN
    return json.loads(golden_path.read_text(encoding="utf-8"))


def _blob(answer: str, excerpts: list[str]) -> str:
    return (answer + " " + " ".join(excerpts)).lower()


def score_case(case: dict, *, answer: str, sources: list[str], excerpts: list[str]) -> CaseScore:
    expected_sources = list(case.get("expected_sources") or [])
    expected_keywords = list(case.get("expected_keywords") or [])
    retrieved = list(sources)
    expected_set = {item.lower() for item in expected_sources}
    retrieved_set = {item.lower() for item in retrieved}
    matched = expected_set & retrieved_set
    recall = (len(matched) / len(expected_set)) if expected_set else 1.0
    # expected_sources is an OR list: at least one listed file must appear in top-k.
    retrieval_ok = bool(matched) if expected_set else True

    haystack = _blob(answer, excerpts)
    missing = [kw for kw in expected_keywords if kw.lower() not in haystack]
    hit_rate = (
        (len(expected_keywords) - len(missing)) / len(expected_keywords)
        if expected_keywords
        else 1.0
    )
    keyword_ok = not missing
    return CaseScore(
        id=str(case.get("id") or case["question"][:40]),
        question=case["question"],
        retrieval_ok=retrieval_ok,
        keyword_ok=keyword_ok,
        passed=retrieval_ok and keyword_ok,
        retrieval_recall=recall,
        keyword_hit_rate=hit_rate,
        retrieved_sources=retrieved,
        expected_sources=expected_sources,
        missing_keywords=missing,
    )


def run_eval(
    *,
    golden_path: Path | str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    generator: Generator | None = None,
    ingest: bool = True,
) -> EvalReport:
    """Ingest the sample corpus (unless skipped) and score golden questions."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    store = store or get_store(settings)
    generator = generator or get_generator(settings)
    golden = load_golden(golden_path)

    if ingest:
        ingest_directory(
            golden.get("docs_path") or settings.docs_path,
            settings=settings,
            embedder=embedder,
            store=store,
        )

    cases: list[CaseScore] = []
    for case in golden["cases"]:
        result = ask(
            case["question"],
            settings=settings,
            embedder=embedder,
            store=store,
            generator=generator,
        )
        scored = score_case(
            case,
            answer=result.answer,
            sources=[hit.source for hit in result.hits],
            excerpts=[hit.text for hit in result.hits],
        )
        cases.append(scored)

    passed = sum(1 for item in cases if item.passed)
    total = len(cases)
    return EvalReport(
        total=total,
        passed=passed,
        pass_rate=(passed / total) if total else 0.0,
        threshold=threshold,
        embedding_provider=embedder.name,
        llm_provider=generator.name,
        cases=cases,
    )


def format_table(report: EvalReport) -> str:
    lines = [
        f"eval  pass={report.passed}/{report.total}  "
        f"rate={report.pass_rate:.2f}  threshold={report.threshold:.2f}  "
        f"embedder={report.embedding_provider}  llm={report.llm_provider}",
        f"{'id':<22} {'retr':<6} {'kw':<6} result",
        "-" * 48,
    ]
    for case in report.cases:
        lines.append(
            f"{case.id:<22} "
            f"{'ok' if case.retrieval_ok else 'MISS':<6} "
            f"{'ok' if case.keyword_ok else 'MISS':<6} "
            f"{'PASS' if case.passed else 'FAIL'}"
        )
        if not case.passed:
            if not case.retrieval_ok:
                lines.append(
                    f"  expected sources {case.expected_sources} "
                    f"got {case.retrieved_sources}"
                )
            if case.missing_keywords:
                lines.append(f"  missing keywords {case.missing_keywords}")
    lines.append("-" * 48)
    lines.append("PASS" if report.ok else "BELOW THRESHOLD")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score RAG quality on the sample corpus")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip ingest (use whatever is already in the vector store)",
    )
    args = parser.parse_args(argv)

    report = run_eval(
        golden_path=args.golden,
        threshold=args.threshold,
        ingest=not args.no_ingest,
    )
    if args.json:
        payload = asdict(report)
        print(json.dumps(payload, indent=2))
    else:
        print(format_table(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
