# Company RAG Knowledge Assistant

FastAPI retrieval-augmented generation (RAG) service that answers questions from local markdown. Sample corpus is a **fictional** company, Vision AI Ops (TraceLight / AlertMesh). Portfolio demo only — no hosted URL, no real customer data, no production secrets.

Built for Days 11–18 of a 30-day AI automation plan. **Days 11–14 are in this repo:** scaffold, token-aware chunking, hybrid BM25 + dense retrieval, an offline eval harness, and citation polish.

Source: [github.com/Birra3324/company-rag-assistant](https://github.com/Birra3324/company-rag-assistant)

## How to demo (under 10 minutes)

```bash
git clone https://github.com/Birra3324/company-rag-assistant.git
cd company-rag-assistant
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 1) Ingest the five sample FAQ/policy docs
python -m app.rag.ingest

# 2) Run the API
uvicorn app.main:app --reload --port 8788
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8788/health

curl -sS http://127.0.0.1:8788/ask \
  -H "content-type: application/json" \
  -d '{"question":"How many PTO days do employees receive?"}'

curl -sS http://127.0.0.1:8788/query \
  -H "content-type: application/json" \
  -d '{"question":"Can we keep TraceLight data in the EU?"}'
```

`POST /query` is an alias of `POST /ask`. Interactive docs: [http://127.0.0.1:8788/docs](http://127.0.0.1:8788/docs).

If `AUTO_INGEST_ON_STARTUP=true` (default outside tests), the first API boot ingests `data/sample_docs/` when the vector store is empty — so you can skip the ingest command after a fresh clone.

## Architecture

```mermaid
flowchart LR
  Docs["data/sample_docs/*.md"] --> Chunk["Token-aware chunks"]
  Chunk --> Embed["Embed"]
  Embed --> Store[("SQLite cosine index\noptional Chroma")]
  User["curl /ask"] --> API[FastAPI]
  API --> EmbedQ["Embed question"]
  EmbedQ --> Dense["Dense top-n"]
  Store --> Dense
  Store --> BM25["BM25 over corpus"]
  Dense --> Fuse["Weighted RRF blend"]
  BM25 --> Fuse
  Fuse --> Gen["Extractive answer\n+ numbered citations"]
  Gen --> User
```

Default path is fully local: **hashing-trick embeddings + SQLite cosine store + BM25 hybrid retrieval + extractive answers**. No GPU, no API key, no extra containers.

## Stack

| Piece | Default (offline) | Optional via env |
| --- | --- | --- |
| API | FastAPI 3.12, port **8788** | — |
| Chunking | Header-aware, **token** windows (`CHUNK_SIZE=180`) | — |
| Embeddings | Local signed hashing (384-d, numpy) | `sentence-transformers` MiniLM, or OpenAI `text-embedding-3-small` |
| Retrieval | Dense cosine + BM25 fused (RRF) | `VECTOR_BACKEND=chroma` after `pip install -r requirements-ml.txt` |
| Generator | Extractive sentences + `[n]` citations | OpenAI chat, or Ollama (`llama3.2`) |
| Eval | `evals/golden.json` + `python -m app.rag.eval_harness` | — |
| Tests | pytest + TestClient; embeddings/LLM mocked or local hashing | CI never calls OpenAI or loads GPU models |

This is not an iPaaS/RPA demo and does not claim UiPath, Workato, MuleSoft, or ServiceNow.

## API

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/` | Service pointers |
| GET | `/health` | Chunk count, providers, backend |
| POST | `/ingest` | Re-read `DOCS_PATH` and rebuild the index |
| POST | `/ask` | `{ "question": "...", "top_k": 4 }` |
| POST | `/query` | Same body as `/ask` |

Example `200` from `/ask` (shape, not a live capture):

```json
{
  "answer": "Full-time employees receive **20 PTO days** of paid time off each calendar year, accrued monthly...\n\nSources: [1] Time Off and Leave Policy.",
  "sources": [
    {
      "source": "02-pto-policy.md",
      "title": "Time Off and Leave Policy",
      "score": 0.41,
      "excerpt": "## PTO days\n\nFull-time employees receive **20 PTO days** of paid time off...",
      "chunk_id": "02-pto-policy.md::0001",
      "heading": "PTO days"
    }
  ],
  "retrieved": 4,
  "embedding_provider": "local",
  "llm_provider": "extractive"
}
```

`POST /ask` returns **409** if the store is empty (ingest first). Source titles come from the document H1; `chunk_id` and section `heading` are optional extras for citations.

## Sample documents

Fictional Vision AI Ops content under `data/sample_docs/`:

1. Company overview (TraceLight, AlertMesh, Cambridge listing)
2. PTO and leave (20 PTO days, 8 sick days, parental leave)
3. IT security and acceptable use (MFA, no real secrets in chat)
4. Customer support FAQ (P1 = 30 minutes, EU-West residency)
5. Remote work and expenses ($150 stipend, hybrid Tues/Thu)

Drop additional `.md` / `.txt` files in that folder and `POST /ingest` again.

## Chunking (`CHUNK_SIZE`)

`CHUNK_SIZE` and `CHUNK_OVERLAP` are **approximate word tokens**, not characters. A token is an alphanumeric sequence (the same tokenizer the local hashing embedder uses). English prose is ~4 characters per token, so the Day 11 700-character window is about 180 tokens.

- Default `CHUNK_SIZE=180`, `CHUNK_OVERLAP=40`
- Markdown headings bound sections first
- Long sections pack paragraphs, then sentences, then token windows
- The section heading is prepended onto each piece so a split FAQ still retrieves

## Environment variables

See `.env.example`. Important ones:

| Variable | Purpose |
| --- | --- |
| `EMBEDDING_PROVIDER` | `local` (default), `sentence-transformers`, or `openai` |
| `LLM_PROVIDER` | `extractive` (default), `openai`, or `ollama` |
| `OPENAI_API_KEY` | Only when a provider is `openai`. Never hard-coded. |
| `VECTOR_BACKEND` | `sqlite` (default) or `chroma` |
| `DOCS_PATH` | Defaults to `data/sample_docs` |
| `CHUNK_SIZE` | Target tokens per chunk (default 180) |
| `CHUNK_OVERLAP` | Overlapping tokens (default 40) |
| `RETRIEVE_K` | Chunks returned to the generator (default 4) |
| `RETRIEVE_POOL` | Dense shortlist size before BM25 fusion (default 24) |
| `AUTO_INGEST_ON_STARTUP` | Ingest when the store is empty (`false` in pytest) |

OpenAI and sentence-transformers are **opt-in**. Leaving the key blank keeps the local path.

Better local embeddings (downloads MiniLM on first use):

```bash
pip install -r requirements-ml.txt
# in .env
EMBEDDING_PROVIDER=sentence-transformers
VECTOR_BACKEND=chroma   # optional
```

## Testing

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Tests use a temp SQLite file, local hashing (or a mock embedder/generator), and block OpenAI HTTP from the RAG layer. They do not need a GPU, Chroma, or API keys. The same command runs on GitHub Actions (push/PR to `main`, Python 3.12).

## Eval harness

Golden questions live in `evals/golden.json` (expected source files + keywords). The scorer ingests the sample corpus with the default offline stack and checks retrieval + answer/excerpt overlap:

```bash
python -m app.rag.eval_harness
python -m app.rag.eval_harness --json
# or
scripts/eval.sh
```

Pass rate must be at least **0.75** (9/12 on the current golden set is the CI floor; the default path usually clears more). No GPU and no API keys. pytest covers the same path in `tests/test_eval_harness.py`.

## Docker

```bash
docker compose up --build
curl -sS http://127.0.0.1:8788/health
```

The image is `python:3.12-slim`. Compose runs only the API; the index lives in a volume. OpenAI remains optional via `OPENAI_API_KEY`.

## Project layout

```
app/                FastAPI app, settings, RAG pipeline
app/rag/chunking.py Token-aware header-aware splitter
app/rag/retrieve.py BM25 + dense hybrid fusion
app/rag/eval_harness.py  Offline eval CLI
data/sample_docs/   Fictional FAQ + policy markdown
evals/golden.json   Golden questions for the sample corpus
scripts/            ingest.sh, run_dev.sh, eval.sh
tests/              pytest (mocked / local, no keys)
docs/status.md      Days 11–18 checklist
.github/workflows/ci.yml
```

## Status and later days

**Days 11–14 (this PR):** working scaffold, token-aware chunking, hybrid BM25 + dense retrieval, eval harness, citation fields (`title`, `chunk_id`, `heading`).

**Days 15–18 (next):** MiniLM default-path polish, citation UI / demo screenshots, optional demo auth, recruiter walkthrough notes.

See [docs/status.md](docs/status.md) for the checklist.

## License

MIT © 2026 Birra Gemedi
