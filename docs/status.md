# Status — Days 11–18

Portfolio track for [company-rag-assistant](https://github.com/Birra3324/company-rag-assistant). Local demo only; no hosted URL. Default path stays offline (hashing embeddings + SQLite + extractive answers).

## Days 11–14 checklist

| Day | Intent | Status |
| --- | --- | --- |
| 11 | Scaffold FastAPI + ingest + ask + sample Vision AI Ops docs + pytest + Docker + CI | **Done** (this PR includes the scaffold; originally PR #1) |
| 12 | Token-aware / header-aware chunking; hybrid BM25 + dense retrieval (offline default) | **Done** |
| 13 | Eval harness: golden questions + CLI/pytest scorer, no GPU or API keys | **Done** |
| 14 | Citation polish (`title`, optional `chunk_id` / `heading`); README + this checklist | **Done** |

### Day 12 — chunking and hybrid retrieval

- [x] `CHUNK_SIZE` / `CHUNK_OVERLAP` are **token** budgets (alphanumeric word tokens), documented in `app/rag/chunking.py` and `.env.example`
- [x] Markdown headings still bound sections; long sections split on paragraphs → sentences → token windows
- [x] Section heading is prepended onto split pieces so FAQ answers stay retrievable
- [x] BM25 over the full corpus fused with dense cosine via weighted reciprocal rank fusion (`app/rag/retrieve.py`)
- [x] Default path needs no GPU, no OpenAI key, no extra containers

### Day 13 — eval harness

- [x] Golden set at `evals/golden.json` (expected sources + keywords)
- [x] CLI: `python -m app.rag.eval_harness` (also `scripts/eval.sh`)
- [x] pytest: `tests/test_eval_harness.py` (pass rate ≥ 0.75 on the sample corpus)
- [x] CI runs pytest (includes the harness) without keys or GPU

### Day 14 — citations and docs

- [x] `POST /ask` sources include stable document titles, `chunk_id`, and section `heading`
- [x] Extractive answers cite `Sources: [1] Title`
- [x] README status updated for Days 12–14 done / Days 15–18 next

## Days 15–18 (next)

| Day | Intent | This repo |
| --- | --- | --- |
| 15 | MiniLM default-path polish (`sentence-transformers` quality without making it required) | later |
| 16 | Citation UI / demo screenshots (recruiter walkthrough of `/docs` + example answers) | later |
| 17 | Optional auth (demo-grade, not a real SSO) + Docker/demo hardening | later |
| 18 | Recruiter walkthrough notes, residual bugs, freeze the demo | later |

Constraints that stay true for the whole track:

- Fictional Vision AI Ops docs only
- No real API keys, Slack tokens, or customer traces in git
- No UiPath / Workato / MuleSoft / ServiceNow claims
