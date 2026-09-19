# Embedding Model Comparison

## Models

1. `sentence-transformers/all-MiniLM-L6-v2`
2. `BAAI/bge-small-en-v1.5`
3. `intfloat/e5-small-v2`

## Comparison Metrics

### Retrieval quality

- **Hit@1:** Percentage of supported questions whose expected policy section is
  ranked first.
- **Hit@3:** Percentage of supported questions whose expected policy section is
  present in the top three results.
- **Mean Reciprocal Rank (MRR):** Rewards models that rank the expected section
  closer to first place.
- **Unsupported-question precision, recall, and F1:** Measures how reliably a
  calibrated distance threshold separates answerable questions from questions
  not covered by the policy.
- **Distance margin:** Difference between the expected section's distance and
  the nearest incorrect section's distance. A larger positive separation makes
  retrieval decisions more reliable.

### Performance and resource usage

- **Document embedding latency:** Time required to embed and index all policy
  chunks.
- **Query embedding latency:** Median and p95 time required to embed one
  question.
- **Retrieval latency:** Median and p95 time required to search the index after
  the query embedding has been created.
- **Throughput:** Questions embedded per second.
- **Peak memory usage:** Maximum RAM used while loading and running the model.
- **Model size:** Downloaded model size on disk.
- **Embedding dimensions:** Number of values in each stored vector, which
  affects storage and search cost.

## Fair-comparison controls

- Use the same six policy chunks and the same evaluation questions for every
  model.
- Embed the same text representation for every model: section number, newline,
  section title, newline, then the original section body text.
- Require every provider adapter to return unit-normalized vectors.
- Normalize vectors and use cosine distance for every comparison.
- Use each model's documented input format. In particular,
  `intfloat/e5-small-v2` uses `query:` for questions and `passage:` for policy
  chunks.
- Warm up each model before measuring latency and repeat timing runs.
- Tune any unsupported-question threshold on a validation set, then report
  results on a separate test set.

## Results

| Model | Hit@1 | Hit@3 | MRR | Unsupported F1 | Query p50 | Query p95 | Peak RAM | Model size | Dimensions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sentence-transformers/all-MiniLM-L6-v2` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 384 |
| `BAAI/bge-small-en-v1.5` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 384 |
| `intfloat/e5-small-v2` | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 384 |
