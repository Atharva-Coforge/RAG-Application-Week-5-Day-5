# LLM Model Comparison

## Decision status

Selected: `qwen3:8b` via local Ollama.

The accepted decision and its consequences are recorded in
[`architecture-decisions.md`](architecture-decisions.md). Runtime code reads
that name from `GENERATION_MODEL`.

`qwen3:8b` was chosen because the latest comparison produced correct
citations for every supported question, 100% valid structured output, slightly
lower latency than `mistral:7b`, and a correct gym-membership refusal. Gym
refusal was also correct for `mistral:7b`; that metric was tied. `mistral:7b`
missed the expected Receipts citation on the $20 taxi question.

## Candidates

1. `mistral:7b` via local Ollama
2. `qwen3:8b` via local Ollama

## Fair-comparison controls

- Frozen retrieval: selected MiniLM embeddings and exact in-memory cosine
  top-three results, computed once and reused for both models.
- The same six gold questions and required answers.
- The same grounded prompt and `GenerationDecision` JSON schema.
- Official Ollama chat API, `format=<JSON schema>`, `temperature=0`,
  `think=False`.
- Prompt rule: refuse only when no excerpt has a relevant rule; allow
  synonyms, premium-airfare mapping, and dollar comparisons; use excerpt
  facts only.
- Ollama host from gitignored `.env`: `OLLAMA_HOST`.
- One quality pass over all six questions.
- Three measured latency rounds of those six questions.
- Citation correctness is automatic (supported answers must cite the expected
  section). Gym refusal is automatic (canonical refusal text, `citation=null`).
- Answer meaning is not auto-scored; required_answer and model_answer are
  written side by side for your judgment.

Environment: Python 3.12.14, Linux aarch64, local Ollama.

## Measured results

Latest run: synonym-aware prompt and `think=False`.

| Model | Valid JSON | Citation correct | Gym refusal | Quality pass | Latency p50 | Latency p95 |
|---|---:|---:|---|---:|---:|---:|
| `mistral:7b` | 100% | 80% (4 / 5) | Correct | 18.145 s | 2.399 s | 3.046 s |
| `qwen3:8b` | 100% | 100% (5 / 5) | Correct | 16.699 s | 2.311 s | 3.041 s |

Both models returned valid structured output for every question and both
refused the gym-membership question with the canonical refusal and no
citation.

`qwen3:8b` cited the expected section for all five supported questions.

`mistral:7b` cited the expected section for meals, airfare, hotel, and
limousine. The $20 taxi answer said no receipt is required, but it cited
Ground Transportation instead of Receipts.

## Side-by-side answers

Meaning is for you to judge against `required_answer`. Citation correctness
above is separate.

| Question | Required answer | `mistral:7b` | `qwen3:8b` |
|---|---|---|---|
| How much can I spend on food each day? | Employees may claim up to $65 per day for meals while traveling overnight. | $65 | You can spend up to $65 per day on food while traveling overnight. |
| Can I book first-class airfare? | Employees must purchase economy airfare. Business-class airfare requires written approval from a vice president. | Booking first-class airfare is not allowed without written approval from a vice president. | No, you cannot book first-class airfare without written approval from a vice president. |
| My hotel costs $250. What do I need? | Hotels are reimbursable up to $225 per night. A manager must approve higher rates before booking. | You need a manager approval for your hotel cost of $250 as it exceeds the reimbursable limit of $225 per night. | You need a manager's approval for the hotel costs since the policy states that higher rates require manager approval. The hotel costs exceed the $225 per night limit. |
| Do I need a receipt for a $20 taxi? | Receipts are required only for individual expenses of $25 or more, so a receipt is not required for a $20 taxi expense. | No, you do not need a receipt for a $20 taxi. (cited 4. Ground Transportation) | No, you do not need a receipt for a $20 taxi. |
| Can I claim a limousine upgrade? | Luxury vehicle upgrades are not reimbursable. | Luxury vehicle upgrades, including limousine upgrades, are not reimbursable according to section 4 of the expense policy. | No, you cannot claim a limousine upgrade. |
| Does the company reimburse gym memberships? | The provided policy does not answer this question. | The provided policy does not answer this question. | The provided policy does not answer this question. |

## Earlier runs

These used the older refuse-if-wording-differs prompt.

| Run | `mistral:7b` citation | `qwen3:8b` citation | `qwen3:8b` p50 |
|---|---:|---:|---:|
| Thinking on (Ollama default) | 40% (2 / 5) | 80% (4 / 5) | 9.393 s |
| `think=False`, old prompt | 40% (2 / 5) | 80% (4 / 5) | 1.960 s |

## Running the comparison

PostgreSQL is not required. MiniLM and Ollama are required. To avoid Hub
metadata checks when MiniLM is already cached:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
uv sync --extra embeddings --extra generation --group dev
uv run python -m expense_rag.evaluation.generation_runner
```

Raw results are written to
`data/artifacts/evaluation/generation-comparison.json`.
