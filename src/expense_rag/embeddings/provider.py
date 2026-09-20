"""Shared embedding workflow independent of any concrete model."""

from __future__ import annotations

from collections.abc import Sequence
from math import isclose, isfinite, sqrt

from expense_rag.config import Settings
from expense_rag.embeddings.base import EmbeddingProvider, EmbeddingVector
from expense_rag.models import PolicyChunk, PolicySection

_UNIT_VECTOR_TOLERANCE = 1e-6


class EmbeddingContractError(ValueError):
    """Raised when a provider violates the shared embedding contract."""


def format_section_for_embedding(section: PolicySection) -> str:
    """Build the canonical document text used by every embedding candidate."""
    return f"{section.section}\n{section.section_title}\n{section.text}"


def embed_sections(
    sections: Sequence[PolicySection],
    provider: EmbeddingProvider,
) -> tuple[PolicyChunk, ...]:
    """Embed parsed sections and preserve their metadata and original text."""
    if not sections:
        raise EmbeddingContractError("at least one policy section is required")

    _validate_provider_metadata(provider)
    texts = tuple(format_section_for_embedding(section) for section in sections)
    vectors = provider.embed_documents(texts)

    if len(vectors) != len(sections):
        raise EmbeddingContractError(
            "provider returned a different number of vectors than input texts"
        )

    chunks: list[PolicyChunk] = []
    for index, (section, vector) in enumerate(zip(sections, vectors, strict=True)):
        validated_vector = validate_embedding_vector(
            vector,
            expected_dimension=provider.dimension,
            context=f"document vector {index}",
        )
        chunks.append(
            PolicyChunk(
                **section.model_dump(),
                embedding=validated_vector,
            )
        )

    return tuple(chunks)


def embed_query(
    question: str,
    provider: EmbeddingProvider,
) -> EmbeddingVector:
    """Embed and validate one non-empty user question."""
    if not question.strip():
        raise EmbeddingContractError("question must not be empty")

    _validate_provider_metadata(provider)
    return validate_embedding_vector(
        provider.embed_query(question),
        expected_dimension=provider.dimension,
        context="query vector",
    )


def validate_embedding_vector(
    vector: Sequence[float],
    *,
    expected_dimension: int,
    context: str,
) -> EmbeddingVector:
    """Validate and freeze one finite unit-normalized embedding vector."""
    if expected_dimension <= 0:
        raise EmbeddingContractError("embedding dimension must be positive")
    if len(vector) != expected_dimension:
        raise EmbeddingContractError(
            f"{context} has dimension {len(vector)}; expected {expected_dimension}"
        )

    try:
        frozen = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as error:
        raise EmbeddingContractError(
            f"{context} contains a non-numeric value"
        ) from error

    if not all(isfinite(value) for value in frozen):
        raise EmbeddingContractError(f"{context} contains a non-finite value")

    norm = sqrt(sum(value * value for value in frozen))
    if not isclose(
        norm,
        1.0,
        rel_tol=_UNIT_VECTOR_TOLERANCE,
        abs_tol=_UNIT_VECTOR_TOLERANCE,
    ):
        raise EmbeddingContractError(
            f"{context} must be unit-normalized; received norm {norm:.8f}"
        )

    return frozen


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider without exposing SDK imports."""
    from expense_rag.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingProvider,
    )

    return SentenceTransformerEmbeddingProvider(settings.embedding_model)


def _validate_provider_metadata(provider: EmbeddingProvider) -> None:
    if not provider.model_name.strip():
        raise EmbeddingContractError("embedding model name must not be empty")
    if provider.dimension <= 0:
        raise EmbeddingContractError("embedding dimension must be positive")
