"""Single grounded prompt with evidence labels and refusal instruction."""

from __future__ import annotations

from expense_rag.models import SearchResult

SYSTEM_INSTRUCTION = """Answer the question using only the policy excerpts below.
Include the section that supports your answer.
If no excerpt contains a relevant rule, respond:
"The provided policy does not answer this question."

The excerpts answer the question if they contain the relevant rule,
even when the user uses different words.
You may:
- treat everyday synonyms as the same idea (food and meals; taxi and
  ground transportation; limousine and luxury vehicle upgrade)
- treat first-class, business-class, and other premium seating as
  non-economy airfare
- compare a dollar amount in the question to a limit in the excerpts
Do not refuse just because the question omits a detail that is already
in the excerpt (for example overnight travel).
Do not invent dollar amounts, approvers, or rules that are not written
in the excerpts.
If no excerpt contains a relevant rule, refuse with the canonical sentence.

Return structured output with these fields only:
- supported: true whenever you can cite a relevant excerpt, otherwise false
- answer: the grounded answer, or the refusal sentence when unsupported
- cited_chunk_id: the chunk_id of that relevant excerpt only when supported, otherwise null
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
