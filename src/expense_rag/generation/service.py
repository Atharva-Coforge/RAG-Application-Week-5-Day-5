"""Grounded answer generation with injected protocol dependencies."""

from __future__ import annotations

from pydantic import ValidationError

from expense_rag.generation.base import GenerationProvider
from expense_rag.generation.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from expense_rag.models import (
    REFUSAL_ANSWER,
    Citation,
    GenerationDecision,
    PolicyChunk,
    RagResponse,
    SearchResult,
)
from expense_rag.retrieval.cosine import MAX_RETRIEVAL_RESULTS


class GenerationContractError(ValueError):
    """Raised when generation input or provider output violates the contract."""


class GenerationService:
    """Turn a question and frozen search results into a public RagResponse."""

    def __init__(self, *, generation_provider: GenerationProvider) -> None:
        self._generation_provider = generation_provider

    def generate(
        self,
        question: str,
        retrieved_chunks: tuple[SearchResult, ...],
    ) -> RagResponse:
        """Allow citations only to retrieved chunks and canonicalize refusals."""
        if not question.strip():
            raise GenerationContractError("question must not be empty")
        if len(retrieved_chunks) > MAX_RETRIEVAL_RESULTS:
            raise GenerationContractError(
                f"generation accepts at most {MAX_RETRIEVAL_RESULTS} retrieved chunks"
            )

        if not retrieved_chunks:
            return _public_response(
                answer=REFUSAL_ANSWER,
                citation=None,
                retrieved_chunks=(),
            )

        try:
            raw_decision = self._generation_provider.generate(
                system_instruction=SYSTEM_INSTRUCTION,
                user_prompt=build_user_prompt(question, retrieved_chunks),
                response_schema=GenerationDecision.model_json_schema(),
            )
        except GenerationContractError:
            raise
        except Exception as error:
            raise GenerationContractError(
                "generation provider returned malformed output"
            ) from error

        decision = _decision_from_provider(raw_decision)
        if not decision.supported:
            return _public_response(
                answer=REFUSAL_ANSWER,
                citation=None,
                retrieved_chunks=retrieved_chunks,
            )

        cited_chunk = _chunk_for_id(retrieved_chunks, decision.cited_chunk_id)
        if cited_chunk is None:
            raise GenerationContractError(
                "cited chunk_id is not in the retrieved set"
            )

        return _public_response(
            answer=decision.answer,
            citation=Citation.from_chunk(cited_chunk),
            retrieved_chunks=retrieved_chunks,
        )


def _decision_from_provider(value: object) -> GenerationDecision:
    if isinstance(value, GenerationDecision):
        return value
    if isinstance(value, dict):
        try:
            return GenerationDecision.model_validate(value)
        except ValidationError as error:
            raise GenerationContractError(
                "generation provider returned malformed output"
            ) from error
    raise GenerationContractError(
        "generation provider must return structured output"
    )


def _chunk_for_id(
    retrieved_chunks: tuple[SearchResult, ...],
    chunk_id: str | None,
) -> PolicyChunk | None:
    if chunk_id is None:
        return None
    for result in retrieved_chunks:
        if result.chunk.chunk_id == chunk_id:
            return result.chunk
    return None


def _public_response(
    *,
    answer: str,
    citation: Citation | None,
    retrieved_chunks: tuple[SearchResult, ...],
) -> RagResponse:
    try:
        return RagResponse(
            answer=answer,
            citation=citation,
            retrieved_chunks=retrieved_chunks,
        )
    except ValidationError as error:
        raise GenerationContractError(
            "generation result violated the public response contract"
        ) from error
