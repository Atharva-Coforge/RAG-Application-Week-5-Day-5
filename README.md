# Expense-Policy RAG

A grounded Retrieval-Augmented Generation (RAG) CLI for the employee expense
policy. Ask a question; the app retrieves the three closest policy sections and
answers only from that evidence. If the policy does not contain the answer, it
refuses.

```bash
uv run --extra embeddings --extra pgvector --extra generation \
  expense-rag ask "Can I book first-class airfare?"
```

## Table of contents

- [What this application does](#what-this-application-does)
- [Selected stack](#selected-stack)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to run](#how-to-run)
- [Response shape](#response-shape)
- [Tests](#tests)
- [Comparison experiments](#comparison-experiments)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## What this application does

The source document is [`data/policy.md`](data/policy.md): six numbered
sections (Meals, Hotels, Airfare, Ground Transportation, Receipts, Submission
Deadline).

On each question the pipeline:

1. Embeds the question with MiniLM.
2. Retrieves the top three chunks by cosine distance (smaller is closer).
3. Sends only those excerpts to a local Ollama model.
4. Returns JSON with an answer, a citation, and the retrieved chunks.

Unsupported questions, such as gym memberships, return the exact refusal:

`The provided policy does not answer this question.`

with `citation: null`.

This is a UV-managed CLI. There is no web UI or HTTP API.

## Selected stack

Choices were measured, then locked. Details are in
[`docs/architecture-decisions.md`](docs/architecture-decisions.md).

| Layer | Selected value | Why |
| --- | --- | --- |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | Perfect gold retrieval, largest margin, smallest model |
| Vector store | PostgreSQL + pgvector | Running database; owns the SQL migration |
| Generation | local Ollama `qwen3:8b` | Complete citations, 100% valid JSON, lower latency |

Chroma, FAISS, and `mistral:7b` stay available for comparison.

Retrieval `top_k` is fixed at **3**.

## Repository layout

```text
.
├── src/expense_rag/              Application package
│   ├── cli.py                    ingest / ask / evaluate entry point
│   ├── config.py                 Validated Settings and VectorBackend
│   ├── env.py                    Gitignored .env loader
│   ├── models.py                 Shared Pydantic contracts
│   ├── embeddings/               Embedding protocol and MiniLM adapter
│   ├── ingestion/                Policy parser and ingest service
│   ├── retrieval/                Exact cosine scorer and retrieve service
│   ├── vector_stores/            Chroma, FAISS, and pgvector adapters
│   ├── generation/               Grounded prompt, Ollama adapter, service
│   └── evaluation/               Gold loader, acceptance runner, comparisons
├── tests/                        Unit, contract, and integration tests
├── data/
│   ├── policy.md                 Source expense policy
│   ├── gold/gold-data.md         Six evaluation questions
│   └── artifacts/                Local reports (gitignored)
├── migrations/                   pgvector schema
├── docs/                         ADRs and comparison write-ups
├── .env.example                  Non-secret environment template
├── docker-compose.yml            Optional local Postgres + pgvector
└── pyproject.toml                UV project, extras, and CLI script
```

| Path | Role |
| --- | --- |
| [`src/expense_rag/cli.py`](src/expense_rag/cli.py) | Commands and JSON output |
| [`src/expense_rag/config.py`](src/expense_rag/config.py) | Runtime settings |
| [`data/policy.md`](data/policy.md) | Document that is chunked and stored |
| [`data/gold/gold-data.md`](data/gold/gold-data.md) | Six gold cases |
| [`.env.example`](.env.example) | Names of every runtime knob |
| [`migrations/001_create_policy_chunks.sql`](migrations/001_create_policy_chunks.sql) | pgvector table |
| [`docs/architecture-decisions.md`](docs/architecture-decisions.md) | Why MiniLM, pgvector, and Qwen won |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the `vector` extension, or Docker
- [Ollama](https://ollama.com) with `qwen3:8b` pulled

```bash
ollama pull qwen3:8b
```

MiniLM downloads on first ingest unless the weights are already cached.

## Installation

```bash
git clone https://github.com/Atharva-Coforge/RAG-Application-Week-5-Day-5
cd RAG-Application-Week-5-Day-5
cp .env.example .env
```

Edit `.env` with local values. Do not commit it.

Install the selected extras from the lockfile:

```bash
uv sync --extra embeddings --extra pgvector --extra generation --group dev
```

Keep those extras on later `uv run` commands. A bare `uv run expense-rag` can
uninstall optional packages.

If MiniLM is already on disk and you want to stay offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Optional local Postgres

```bash
docker compose up -d
```

That starts `pgvector/pgvector:pg17` as `postgres` / `postgres` / `expense_rag`
on port 5432. Point `DATABASE_URL` at that instance, or at any other reachable
Postgres. The app creates the `vector` extension and `policy_chunks` table on
first connect.

From this project's container, Postgres and Ollama on the host are typically
`host.docker.internal`.

## Configuration

All runtime knobs come from the process environment or the gitignored `.env`
file.

| Variable | Required | Purpose |
| --- | --- | --- |
| `EMBEDDING_MODEL` | yes | Embedding model name (selected: MiniLM) |
| `VECTOR_STORE` | yes | `chroma`, `faiss`, or `pgvector` |
| `GENERATION_MODEL` | yes | `qwen3:8b` or `mistral:7b` |
| `OLLAMA_HOST` | yes | Ollama base URL |
| `DATABASE_URL` | only for pgvector | PostgreSQL connection string |

Example `.env`:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE=pgvector
GENERATION_MODEL=qwen3:8b
OLLAMA_HOST=http://host.docker.internal:11434
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/DBNAME
```

Secrets must stay out of git. Invalid or placeholder values fail at startup.

## How to run

Install extras, ingest the policy once, then ask.

```bash
uv sync --extra embeddings --extra pgvector --extra generation --group dev

uv run --extra embeddings --extra pgvector --extra generation expense-rag ingest

uv run --extra embeddings --extra pgvector --extra generation \
  expense-rag ask "Can I book first-class airfare?"
```

Skip `ingest` only when the six chunks are already in the selected store.

### Commands

| Command | What it does |
| --- | --- |
| `expense-rag ingest` | Parse, embed, and replace the six policy chunks |
| `expense-rag ask "…"` | Retrieve top 3 and print one grounded JSON answer |
| `expense-rag evaluate` | Run the six gold questions and write an acceptance report |

Useful flags:

```bash
expense-rag ingest --policy data/policy.md --backend pgvector
expense-rag ask "Can I book first-class airfare?" --backend pgvector
expense-rag evaluate --gold data/gold/gold-data.md --backend pgvector --model qwen3:8b
```

`--backend` may be `chroma`, `faiss`, or `pgvector`. `--model` on `evaluate`
may be `qwen3:8b` or `mistral:7b`.

JSON goes to stdout. Diagnostics go to stderr. Missing credentials, a missing
policy file, or an empty store exit with code 1.

`evaluate` writes
`data/artifacts/evaluation/acceptance-<UTC>.json` (gitignored) and exits 0
only when the assignment checks pass:

- exactly six stored chunks with text, vectors, and metadata
- at most three retrieved chunks; distances numeric and ascending
- expected section in the top three for at least five supported questions
- a citation on every supported answer
- gym-membership refusal with `citation: null`

## Response shape

`ask` prints the assignment contract:

```json
{
  "answer": "Employees must purchase economy airfare. Business-class airfare requires written approval from a vice president.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "3. Airfare"
  },
  "retrieved_chunks": [
    {
      "section": "3. Airfare",
      "distance": 0.53
    }
  ]
}
```

Distances are numbers. At most three chunks are returned, nearest first.

## Tests

The default suite does not call Ollama or Hugging Face:

```bash
uv run pytest tests/unit tests/contract tests/integration
uv run ruff check .
uv run mypy src
```

pgvector integration tests skip unless extras are installed and Postgres is
reachable. CI runs the same non-secret commands.

## Comparison experiments

Optional reruns of the three decision gates:

```bash
uv run --extra embeddings python -m expense_rag.evaluation.embedding_runner

uv run --extra embeddings --extra chroma --extra faiss --extra pgvector \
  python -m expense_rag.evaluation.vector_store_runner

uv run --extra embeddings --extra generation \
  python -m expense_rag.evaluation.generation_runner
```

Recorded results:

- [Embedding comparison](docs/embedding-model-comparison.md)
- [Vector-store comparison](docs/vector-store-comparison.md)
- [LLM comparison](docs/llm-model-comparison.md)

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `pgvector dependencies are not installed` | Re-run with `--extra embeddings --extra pgvector --extra generation` |
| `policy has not been ingested` | Run `ingest` before `ask` or `evaluate` |
| `DATABASE_URL is required` | Set it in `.env` when `VECTOR_STORE=pgvector` |
| Cannot reach Postgres or Ollama from Docker | Use `host.docker.internal` in `DATABASE_URL` and `OLLAMA_HOST` |
| Hugging Face network calls on a cached model | Export `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` |

## License

MIT. See [LICENSE](LICENSE).
