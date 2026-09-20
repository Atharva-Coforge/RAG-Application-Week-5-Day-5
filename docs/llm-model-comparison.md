# LLM Model Comparison

## Decision status

Pending user selection.

Candidates are local Ollama models. No winner is recorded until you choose
after reviewing citation correctness, gym-membership refusal, structured-output
reliability, latency, and the side-by-side answers.

## Candidates

1. `mistral:7b` via local Ollama
2. `qwen3:8b` via local Ollama

## Fair-comparison controls

- Frozen retrieval: selected MiniLM embeddings and exact in-memory cosine
  top-three results, computed once and reused for both models.
- The same six gold questions and required answers.
- The same grounded prompt and `GenerationDecision` JSON schema.
- Official Ollama chat API, `format=<JSON schema>`, `temperature=0`.
- Qwen3 thinking is enabled by default in Ollama. The measured run below used
  that default. A follow-up run will set `think=False` for both models.
- Ollama host from gitignored `.env`: `OLLAMA_HOST`.
- One quality pass over all six questions.
- Three measured latency rounds of those six questions.
- Citation correctness is automatic (supported answers must cite the expected
  section). Gym refusal is automatic (canonical refusal text, `citation=null`).
- Answer meaning is not auto-scored; required_answer and model_answer are
  written side by side for your judgment.

Environment: Python 3.12.14, Linux aarch64, local Ollama.

## Measured results

| Model | Valid JSON | Citation correct | Gym refusal | Quality pass | Latency p50 | Latency p95 |
|---|---:|---:|---|---:|---:|---:|
| `mistral:7b` | 100% | 40% (2 / 5) | Correct | 14.403 s | 1.793 s | 2.547 s |
| `qwen3:8b` | 100% | 80% (4 / 5) | Correct | 67.561 s | 9.393 s | 16.078 s |

Both models returned valid structured output for every question and both
refused the gym-membership question with the canonical refusal and no
citation.

`mistral:7b` cited the expected section for airfare and limousine. It refused
meals, the $250 hotel, and the $20 taxi even though those questions are
supported.

`qwen3:8b` cited the expected section for meals, hotel, receipts, and
limousine. It refused first-class airfare. The hotel answer cited Hotels and
also mentioned receipts.

## Side-by-side answers

Meaning is for you to judge against `required_answer`. Citation correctness
above is separate.

| Question | Required answer | `mistral:7b` | `qwen3:8b` |
|---|---|---|---|
| How much can I spend on food each day? | Employees may claim up to $65 per day for meals while traveling overnight. | The provided policy does not answer this question. | Employees may claim up to $65 per day for meals while traveling overnight. |
| Can I book first-class airfare? | Employees must purchase economy airfare. Business-class airfare requires written approval from a vice president. | Employees must purchase economy airfare. Business-class airfare requires written approval from a vice president. | The provided policy does not answer this question. |
| My hotel costs $250. What do I need? | Hotels are reimbursable up to $225 per night. A manager must approve higher rates before booking. | The provided policy does not answer this question. | Your hotel cost of $250 exceeds the $225 per night reimbursement limit. You need manager approval for higher rates (per section 2. Hotels) and must provide a receipt (per section 5. Receipts). |
| Do I need a receipt for a $20 taxi? | Receipts are required only for individual expenses of $25 or more, so a receipt is not required for a $20 taxi expense. | The provided policy does not answer this question. | No, a receipt is not required for a $20 taxi expense. Receipts are only required for individual expenses of $25 or more, as stated in section 5. |
| Can I claim a limousine upgrade? | Luxury vehicle upgrades are not reimbursable. | No, you cannot claim a limousine upgrade as it is not reimbursable according to section 4 of the expense policy. | No, luxury vehicle upgrades are not reimbursable. |
| Does the company reimburse gym memberships? | The provided policy does not answer this question. | The provided policy does not answer this question. | The provided policy does not answer this question. |

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
