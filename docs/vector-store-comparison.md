# Vector Store Comparison

## Decision status

Selected: PostgreSQL with pgvector.

pgvector is the default application store because it is a running database and
the submitted migration belongs to it. Chroma and FAISS remain implemented
comparison adapters. The accepted decision is recorded in
[`architecture-decisions.md`](architecture-decisions.md).

## Backends

1. Chroma with persistent local storage.
2. FAISS `IndexFlatIP` with a `.faiss` index and JSON metadata sidecar.
3. PostgreSQL with pgvector and exact `<=>` cosine ordering.

## Fair-comparison controls

- Frozen embedding model:
  `sentence-transformers/all-MiniLM-L6-v2`.
- The same six unit-normalized, 384-dimensional policy embeddings.
- The same six gold questions.
- A maximum of three results ordered by ascending cosine distance.
- Twenty complete replacement measurements.
- Two hundred query rounds across all six questions.
- Persistence verified by closing and reopening each store.
- Every result compared against the exact in-memory cosine reference.

Chroma does not expose brute-force exact vector search. To preserve the chosen
exact-search requirement, the Chroma adapter retrieves its six stored vectors
and performs exact application-side cosine ranking. These timings therefore do
not measure Chroma's native HNSW query path.

## Replacement guarantee

Every adapter validates the complete replacement before mutation. pgvector
then performs replacement in a database transaction. Chroma and the FAISS
index/metadata sidecar do not promise crash-safe atomic replacement; this
limitation will be considered in the final operational comparison.

## Current measured results

Environment: Python 3.12.14, Linux aarch64, CPU.

| Backend | Hit@1 | Hit@3 | Exact parity | Reopened count | Replace p50 | Replace p95 | Query p50 | Query p95 | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chroma | 100% | 100% | Yes | 6 | 19.743 ms | 22.228 ms | 0.834 ms | 0.888 ms | 1.215 MB |
| FAISS | 100% | 100% | Yes | 6 | 0.499 ms | 0.538 ms | 0.028 ms | 0.035 ms | 0.072 MB |
| pgvector | 100% | 100% | Yes | 6 | 1.400 ms | 1.636 ms | 0.435 ms | 0.639 ms | Database-managed |

## Running the shared contract suite

The normal CI suite runs the fast test-only memory implementation:

```bash
uv run pytest tests/contract/test_vector_stores.py
```

Run persistent Chroma and FAISS against the same contract:

```bash
uv run --all-extras pytest tests/contract/test_vector_stores.py \
  --vector-backends=chroma,faiss
```

For pgvector, PostgreSQL must already be running. Put the real connection
string in the gitignored `.env` file (copy `.env.example`). Do not commit
or export the password.

```bash
uv run --all-extras pytest tests/contract/test_vector_stores.py \
  --vector-backends=pgvector

uv run --all-extras python -m expense_rag.evaluation.vector_store_runner
```

Raw results are written to
`data/artifacts/evaluation/vector-store-comparison.json`.
