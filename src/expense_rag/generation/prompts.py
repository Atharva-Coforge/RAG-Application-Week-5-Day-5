"""Single grounded prompt with evidence labels and refusal instruction."""

from __future__ import annotations

from expense_rag.models import SearchResult

SYSTEM_INSTRUCTION = """Answer the question using only the policy excerpts below.
Include the section that supports your answer.
If the excerpts do not contain the answer, respond:
"The provided policy does not answer this question."

Return structured output with these fields only:
- supported: true if the excerpts answer the question, otherwise false
- answer: the grounded answer, or the refusal sentence when unsupported
- cited_chunk_id: the chunk_id of the supporting excerpt when supported, otherwise null
"""


def build_user_prompt(
    question: str,
    retrieved_chunks: tuple[SearchResult, ...],
) -> str:
    """Build a user prompt that contains only the question and policy excerpts."""
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Policy excerpts:\n{_format_excerpts(retrieved_chunks)}"
    )


def _format_excerpts(retrieved_chunks: tuple[SearchResult, ...]) -> str:
    if not retrieved_chunks:
        return "(none)"

    return "\n\n".join(
        (
            f"chunk_id: {result.chunk.chunk_id}\n"
            f"section: {result.section}\n"
            f"{result.chunk.text}"
        )
        for result in retrieved_chunks
    )
