# Assignment Submission: Mini RAG Lab

This file is the index for the zip. Full setup, configuration, and
troubleshooting live in [`README.md`](README.md).

Selected stack: MiniLM embeddings, PostgreSQL + pgvector, local Ollama
`qwen3:8b`. Retrieval `top_k` is 3.

## Submission checklist

| Requirement | Location in this zip |
| --- | --- |
| Application source code | [`src/expense_rag/`](src/expense_rag/) |
| `policy.md` | [`data/policy.md`](data/policy.md) |
| Database migration / schema | [`migrations/001_create_policy_chunks.sql`](migrations/001_create_policy_chunks.sql) |
| Ingestion command | `expense-rag ingest` (see below) |
| Instructions for running | [`README.md`](README.md) |
| Automated tests | [`tests/`](tests/) |
| Saved output for all six questions | This file, section [Saved output](#saved-output-for-the-six-required-questions) |

Gold questions used by `evaluate`: [`data/gold/gold-data.md`](data/gold/gold-data.md).

## Ingestion command

Parse `data/policy.md`, embed the six sections, and replace the store:

```bash
uv run --extra embeddings --extra pgvector --extra generation expense-rag ingest
```

Optional flags:

```bash
uv run --extra embeddings --extra pgvector --extra generation \
  expense-rag ingest --policy data/policy.md --backend pgvector
```

Implementation: [`src/expense_rag/cli.py`](src/expense_rag/cli.py) (`ingest`
subcommand) and [`src/expense_rag/ingestion/`](src/expense_rag/ingestion/).

## How to run

Copy [`.env.example`](.env.example) to `.env` and set local values. Do not
commit `.env`. Then:

```bash
uv sync --extra embeddings --extra pgvector --extra generation --group dev

# Optional local Postgres
docker compose up -d

uv run --extra embeddings --extra pgvector --extra generation expense-rag ingest

uv run --extra embeddings --extra pgvector --extra generation \
  expense-rag ask "Can I book first-class airfare?"
```

Ask all six gold questions and write an acceptance report:

```bash
uv run --extra embeddings --extra pgvector --extra generation expense-rag evaluate
```

`evaluate` exits 0 only when acceptance checks pass. Prerequisites, extras, and
environment variables are in [`README.md`](README.md).

## Automated tests

Default tests do not call Ollama or Hugging Face:

```bash
uv run pytest tests/unit tests/contract tests/integration
uv run ruff check .
uv run mypy src
```

pgvector integration tests skip unless extras are installed and Postgres is
reachable.

## Saved output for the six required questions

Live run on 2026-09-21 with MiniLM, pgvector, and `qwen3:8b`.

Acceptance: **passed**. Six chunks stored; distances numeric and ascending;
expected section in the top three for all five supported questions; gym
refusal with `citation: null`.

### 1. How much can I spend on food each day?

Expected: `$65` per day; citation section 1, Meals.

```json
{
  "answer": "You can spend up to $65 per day on food while traveling overnight.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "1. Meals"
  },
  "retrieved_chunks": [
    { "section": "1. Meals", "distance": 0.5518577992916107 },
    { "section": "2. Hotels", "distance": 0.7672109007835388 },
    { "section": "5. Receipts", "distance": 0.8392695880371575 }
  ]
}
```

### 2. Can I book first-class airfare?

Expected: economy required; business class needs approval; citation section 3.

```json
{
  "answer": "No, you cannot book first-class airfare without written approval from a vice president.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "3. Airfare"
  },
  "retrieved_chunks": [
    { "section": "3. Airfare", "distance": 0.529352275767786 },
    { "section": "4. Ground Transportation", "distance": 0.7939039468765259 },
    { "section": "1. Meals", "distance": 0.7958625704050064 }
  ]
}
```

### 3. My hotel costs $250. What do I need?

Expected: manager approval before booking; citation section 2.

```json
{
  "answer": "You need a manager's approval for the hotel costs since the policy states that higher rates require manager approval. The hotel costs exceed the $225 per night limit.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "2. Hotels"
  },
  "retrieved_chunks": [
    { "section": "2. Hotels", "distance": 0.4352981448173523 },
    { "section": "1. Meals", "distance": 0.7433698773384094 },
    { "section": "5. Receipts", "distance": 0.754422947500768 }
  ]
}
```

### 4. Do I need a receipt for a $20 taxi?

Expected: no receipt required under this policy; citation section 5.

```json
{
  "answer": "No, you do not need a receipt for a $20 taxi.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "5. Receipts"
  },
  "retrieved_chunks": [
    { "section": "5. Receipts", "distance": 0.424387370086468 },
    { "section": "4. Ground Transportation", "distance": 0.6511859893798828 },
    { "section": "1. Meals", "distance": 0.7215414047241211 }
  ]
}
```

### 5. Can I claim a limousine upgrade?

Expected: luxury vehicle upgrades are not reimbursable; citation section 4.

```json
{
  "answer": "No, you cannot claim a limousine upgrade.",
  "citation": {
    "document": "Employee Expense Policy",
    "version": "2.0",
    "section": "4. Ground Transportation"
  },
  "retrieved_chunks": [
    { "section": "4. Ground Transportation", "distance": 0.49227705643909747 },
    { "section": "3. Airfare", "distance": 0.748369544938857 },
    { "section": "5. Receipts", "distance": 0.7563586525489506 }
  ]
}
```

### 6. Does the company reimburse gym memberships?

Expected: the policy does not answer the question; no citation.

```json
{
  "answer": "The provided policy does not answer this question.",
  "citation": null,
  "retrieved_chunks": [
    { "section": "4. Ground Transportation", "distance": 0.7273002727448773 },
    { "section": "2. Hotels", "distance": 0.7295602721237969 },
    { "section": "5. Receipts", "distance": 0.7875970855239528 }
  ]
}
```
