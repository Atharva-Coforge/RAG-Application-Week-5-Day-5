# Expense-policy RAG

A grounded CLI assistant for the Week 5 Mini RAG lab. It splits the employee
expense policy into six section chunks, retrieves the top three by cosine
distance, and answers only from that evidence.

## Selected stack

| Layer | Choice | Decision |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | [ADR-001](docs/architecture-decisions.md) |
| Vector store | PostgreSQL + pgvector | [ADR-002](docs/architecture-decisions.md) |
| Generation | local Ollama `qwen3:8b` | [ADR-003](docs/architecture-decisions.md) |

Chroma, FAISS, and `mistral:7b` remain available for comparison.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the `vector` extension (or `docker compose up -d`)
- [Ollama](https://ollama.com) with `qwen3:8b` pulled
- The MiniLM weights on disk after the first download

## Setup

```bash
cp .env.example .env
```

Edit `.env`. Do not commit it.

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE=pgvector
GENERATION_MODEL=qwen3:8b
OLLAMA_HOST=http://host.docker.internal:11434
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/DBNAME
```

`DATABASE_URL` is required only for pgvector. From this project's container,
Postgres and Ollama on the Mac host are reached through `host.docker.internal`.

Install the selected extras and lockfile:

```bash
uv sync --extra embeddings --extra pgvector --extra generation --group dev
```

Keep those extras on later `uv run` commands. A bare `uv run expense-rag`
can drop optional packages from the environment.

If MiniLM is already cached and you want to stay offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Docker Postgres

`docker-compose.yml` starts `pgvector/pgvector:pg17` as
`postgres` / `postgres` / `expense_rag` on port 5432:

```bash
docker compose up -d
```

Point `DATABASE_URL` at that instance, or at any other reachable Postgres
that can create the `vector` extension. The application also applies
[`migrations/001_create_policy_chunks.sql`](migrations/001_create_policy_chunks.sql)
on first connect.

## CLI

```bash
uv run --extra embeddings --extra pgvector --extra generation expense-rag ingest
uv run --extra embeddings --extra pgvector --extra generation \
  expense-rag ask "How much can I spend on food each day?"
uv run --extra embeddings --extra pgvector --extra generation expense-rag evaluate
```

- `ingest` stores exactly six policy chunks.
- `ask` prints one assignment-shaped JSON response on stdout.
- `evaluate` asks the six gold questions, prints an acceptance report on
  stdout, writes `data/artifacts/evaluation/acceptance-<UTC>.json`, and
  exits 0 only when the assignment checks pass.

Optional overrides: `--policy`, `--gold`, `--backend chroma|faiss|pgvector`,
and `--model qwen3:8b|mistral:7b` on `evaluate`.

Diagnostics go to stderr. Missing credentials, a missing index, or an
empty store exit with code 1.

## Acceptance checks

`evaluate` records:

- six stored chunks with complete text, vectors, and metadata
- at most three retrieved chunks per question, distances numeric and ascending
- expected section in the top three for at least five supported questions
- a citation on every supported answer
- the canonical gym-membership refusal with `citation: null`

## Tests and lint

The non-secret suite does not call Ollama or Hugging Face:

```bash
uv run pytest tests/unit tests/contract tests/integration
uv run ruff check .
uv run mypy src
```

pgvector integration tests skip unless extras are installed and Postgres
is reachable.

## Comparison experiments

Repeat the three decision-gate measurements after `uv sync` with the extras
each runner needs:

```bash
uv run --extra embeddings python -m expense_rag.evaluation.embedding_runner
uv run --extra embeddings --extra chroma --extra faiss --extra pgvector \
  python -m expense_rag.evaluation.vector_store_runner
uv run --extra embeddings --extra generation \
  python -m expense_rag.evaluation.generation_runner
```

Results:

- [Embedding comparison](docs/embedding-model-comparison.md)
- [Vector-store comparison](docs/vector-store-comparison.md)
- [LLM comparison](docs/llm-model-comparison.md)
