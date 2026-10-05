# Status — Days 11–18

Portfolio track for [company-rag-assistant](https://github.com/Birra3324/company-rag-assistant). Local demo only; no hosted URL. Default path stays offline (hashing embeddings + SQLite + extractive answers).

Hiring-manager walkthrough: [docs/walkthrough.md](walkthrough.md).

## Days 11–14 checklist

| Day | Intent | Status |
| --- | --- | --- |
| 11 | Scaffold FastAPI + ingest + ask + sample Vision AI Ops docs + pytest + Docker + CI | **Done** |
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
- [x] README status updated through Day 14

## Days 15–18 checklist

| Day | Intent | Status |
| --- | --- | --- |
| 15 | MiniLM optional path, documented and tested, default stays hashing | **Done** |
| 16 | Citation UI on `GET /ui` plus demo screenshots under `docs/` | **Done** |
| 17 | Demo API key plus request size, `top_k`, and rate limits | **Done** |
| 18 | Recruiter walkthrough, known eval miss recorded, demo frozen | **Done** |

### Day 15 — MiniLM optional path

- [x] `EMBEDDING_PROVIDER=sentence-transformers` loads `sentence-transformers/all-MiniLM-L6-v2` via `requirements-ml.txt`
- [x] Default `EMBEDDING_PROVIDER=local` still uses hashing embeddings: no GPU, no download, no API key
- [x] Tests stub `SentenceTransformer` (`tests/test_sentence_transformers_optional.py`) so CI never fetches weights
- [x] README tells you to re-ingest after switching providers (vector spaces differ)

### Day 16 — citation UI and screenshots

- [x] Single static page at `GET /ui` (`app/static/index.html`), no extra frontend build
- [x] Ask a question, read the extractive answer, expand citations (title, heading, chunk id, snippet)
- [x] API key stays in the browser tab (`sessionStorage`) and is sent only as `X-API-Key` to this origin
- [x] Screenshots generated from a headless run of the app: [ui-ask.png](ui-ask.png), [ui-answer.png](ui-answer.png)

### Day 17 — demo protection

- [x] `API_KEY` required to boot; `/ask`, `/query`, and `/ingest` return 401 without `X-API-Key`; `/health` and `/ui` stay public
- [x] `MAX_TOP_K` (default 12) and `MAX_QUESTION_CHARS` (default 2000) tighten `/ask`
- [x] `MAX_BODY_BYTES` (default 65536) returns 413
- [x] `RATE_LIMIT_PER_MINUTE` (default 120, set `0` to disable) returns 429 with `Retry-After` on the data routes only
- [x] Limits covered by `tests/test_limits.py` and `tests/test_auth.py`
- [x] Docker Compose passes the same caps through and still binds to loopback

### Day 18 — walkthrough

- [x] [docs/walkthrough.md](walkthrough.md): problem, architecture, 3-command run, sample questions, eval numbers, trade-offs, production follow-ups
- [x] README status no longer lists Days 15–18 as next
- [x] Known limit recorded: golden case `hybrid-days` retrieves the remote-work doc but the extractive picker can quote the Cambridge office chunk instead (11/12, rate 0.92, floor 0.75)

Constraints that stay true for the whole track:

- Fictional Vision AI Ops docs only
- No real API keys, Slack tokens, or customer traces in git
- No UiPath / Workato / MuleSoft / ServiceNow claims
