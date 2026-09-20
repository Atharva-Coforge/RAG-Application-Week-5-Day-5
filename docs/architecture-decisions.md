# Architecture Decisions

## ADR-001: Use all-MiniLM-L6-v2 for embeddings

- **Status:** Accepted
- **Date:** 2026-09-19

### Context

The application needs one embedding model for policy chunks and user questions.
The model must retrieve the expected policy section in the top three for at
least five supported gold questions. It should also remain practical for a
small local CLI application.

We compared these models under the same conditions:

1. `sentence-transformers/all-MiniLM-L6-v2`
2. `BAAI/bge-small-en-v1.5`
3. `intfloat/e5-small-v2`

Every candidate used the same six policy chunks, six gold questions,
unit-normalized 384-dimensional vectors, and exact cosine distance. Each model
ran on CPU in an isolated process with two warm-up rounds and 20 measured
rounds. Model-specific documented query and passage formatting was applied.

### Decision

Use `sentence-transformers/all-MiniLM-L6-v2` as the embedding model and as the
frozen embedding baseline for subsequent vector-store comparisons.

### Evidence

MiniLM produced:

- 100% Hit@1 across all five supported questions.
- 100% Hit@3 across all five supported questions.
- Mean reciprocal rank of 1.00.
- The largest mean distance margin: 0.2542.
- The lowest measured query p95 latency: 23.82 ms.
- The smallest downloaded model size: 87.34 MB.
- The lowest measured peak process RAM: 903.70 MB.

BGE also achieved 100% Hit@1 and Hit@3, but its mean distance margin was
smaller at 0.1212, query p95 was slower at 44.25 ms, and its downloaded size
was larger at 128.27 MB.

E5 achieved 100% Hit@3 but only 80% Hit@1 and an MRR of 0.90. It ranked the
expected Receipts section second for the `$20 taxi` question, had the smallest
mean distance margin at 0.0399, and had a query p95 of 39.70 ms.

### Consequences

- Stored embeddings will have 384 dimensions.
- Embedding adapters must return unit-normalized vectors.
- MiniLM receives the canonical text without model-specific query or passage
  prefixes.
- Chroma, FAISS, and pgvector comparisons must reuse MiniLM embeddings so the
  database is the only changing variable.
- The application requires the optional local embedding dependencies and must
  download the MiniLM model before its first use.

### Limitations

The comparison uses a deliberately small assignment dataset. The single
unsupported gym-membership question is not sufficient to tune or validate a
reliable distance threshold, so unsupported-question distance remains an
observation rather than a selection metric.

Detailed measurements are recorded in
[`embedding-model-comparison.md`](embedding-model-comparison.md).

## ADR-002: Use PostgreSQL with pgvector as the default store

- **Status:** Accepted
- **Date:** 2026-09-19

### Context

The application needs one default vector store for policy chunks. We compared
Chroma, FAISS, and PostgreSQL with pgvector using frozen MiniLM embeddings,
exact cosine ranking, the same six gold questions, persistence reopen checks,
and the shared VectorStore contract.

The assignment also asks for a database migration or schema definition. FAISS
and Chroma persist files, not SQL tables. pgvector is the backend that owns
[`../migrations/001_create_policy_chunks.sql`](../migrations/001_create_policy_chunks.sql).

### Decision

Use PostgreSQL with pgvector as the default application vector store. Keep
Chroma and FAISS as implemented comparison adapters.

### Evidence

All three backends retrieved the expected section first for every supported
question and matched the exact cosine reference:

- FAISS query p95: 0.035 ms; replace p95: 0.538 ms.
- pgvector query p95: 0.639 ms; replace p95: 1.636 ms.
- Chroma query p95: 0.888 ms; replace p95: 22.228 ms.

pgvector was not the fastest store. It was selected because it is a running
database the application can query, it provides transactional replacement,
and the submitted schema belongs to that backend.

### Consequences

- Default `VECTOR_STORE` is `pgvector`.
- Runtime requires a reachable PostgreSQL instance and `DATABASE_URL` from
  the process environment or the gitignored `.env` file.
- Stored embeddings remain 384-dimensional unit-normalized vectors.
- Search uses `ORDER BY embedding <=> query_vector ASC LIMIT 3`.
- Chroma and FAISS stay available for repeatable comparison runs.

### Limitations

This lab dataset is six chunks. Approximate indexes and large-corpus behavior
were not part of the selection. Connection details are environment-specific
and must not be committed.

Detailed measurements are recorded in
[`vector-store-comparison.md`](vector-store-comparison.md).
