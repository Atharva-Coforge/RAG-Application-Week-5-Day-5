# Embedding Model Comparison

## Decision status

Selected: `sentence-transformers/all-MiniLM-L6-v2`.

MiniLM will be the frozen embedding baseline for all vector-store comparisons.
The accepted decision and its consequences are recorded in
[`architecture-decisions.md`](architecture-decisions.md).

## Candidates

1. `sentence-transformers/all-MiniLM-L6-v2`
2. `BAAI/bge-small-en-v1.5`
3. `intfloat/e5-small-v2`

## Fair-comparison controls

- Canonical passage text: section number, newline, section title, newline,
  original section body.
- Unit-normalized vectors and exact cosine distance for every model.
- Documented model formatting:
  - MiniLM: plain query and passage text.
  - BGE: documented retrieval instruction on queries.
  - E5: `query:` and `passage:` prefixes.
- CPU execution in a clean subprocess for each model.
- Two warm-up rounds followed by 20 measured rounds.
- The same six policy sections and six gold questions for every model.

Environment: Python 3.12.14, Linux aarch64, CPU.

## Measured results

| Model | Hit@1 | Hit@3 | MRR | Mean margin | Query p50 | Query p95 | Peak RAM | Model size | Dimensions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sentence-transformers/all-MiniLM-L6-v2` | 100% | 100% | 1.00 | 0.2542 | 3.80 ms | 23.82 ms | 903.70 MB | 87.34 MB | 384 |
| `BAAI/bge-small-en-v1.5` | 100% | 100% | 1.00 | 0.1212 | 7.84 ms | 44.25 ms | 911.47 MB | 128.27 MB | 384 |
| `intfloat/e5-small-v2` | 80% | 100% | 0.90 | 0.0399 | 7.48 ms | 39.70 ms | 927.98 MB | 128.25 MB | 384 |

All three models satisfy the assignment requirement that the expected section
appear in the top three for the five supported questions.

## Retrieval observations

- MiniLM ranked the expected section first for all five supported questions.
- BGE ranked the expected section first for all five supported questions.
- E5 ranked the expected receipt section second for the `$20 taxi` question;
  it ranked Ground Transportation first.
- Mean distance margin was largest for MiniLM. A larger positive margin means
  the expected section was more clearly separated from the nearest incorrect
  section.
- MiniLM had the lowest query latency and smallest downloaded model size in
  this run.

For the unsupported gym-membership question, the nearest distances were:

- MiniLM: 0.7273, nearest section 4.
- BGE: 0.3948, nearest section 4.
- E5: 0.2001, nearest section 4.

These unsupported distances are observational only. One unsupported question
is not enough to tune and evaluate a reliable refusal threshold or F1 score.

## Accuracy-first ordering

Applying the agreed ordering—Hit@3 eligibility, then Hit@1, MRR, distance
margin, query p95, and memory—produced:

1. `sentence-transformers/all-MiniLM-L6-v2`
2. `BAAI/bge-small-en-v1.5`
3. `intfloat/e5-small-v2`

Based on this evidence, the user selected
`sentence-transformers/all-MiniLM-L6-v2`.

Raw results are generated locally at
`data/artifacts/evaluation/embedding-comparison.json`.
