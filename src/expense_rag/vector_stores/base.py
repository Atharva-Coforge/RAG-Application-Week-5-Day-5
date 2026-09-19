"""Provider-neutral vector-store contract and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.embeddings.provider import (
    EmbeddingContractError,
    validate_embedding_vector,
)
from expense_rag.models import PolicyChunk, SearchResult
from expense_rag.retrieval.cosine import MAX_RETRIEVAL_RESULTS


class VectorStoreError(RuntimeError):
    """Base error for vector-store runtime failures."""


class VectorStoreClosedError(VectorStoreError):
    """Raised when an operation is attempted after a store is closed."""


class VectorStoreContractError(ValueError):
    """Raised when data violates the shared vector-store contract."""


@runtime_checkable
class VectorStore(Protocol):
    """Synchronous store scoped to one logical collection or index."""

    @property
    def collection_name(self) -> str:
        """Return the logical collection/index name."""
        ...

    @property
    def dimension(self) -> int:
        """Return the vector dimension required by this store."""
        ...

    def replace_all(self, chunks: Sequence[PolicyChunk]) -> None:
        """Atomically replace all records; an empty input clears the store."""
        ...

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        k: int = MAX_RETRIEVAL_RESULTS,
    ) -> tuple[SearchResult, ...]:
        """Return at most ``k`` unique chunks by ascending cosine distance."""
        ...

    def count(self) -> int:
        """Return the number of stored chunks."""
        ...

    def close(self) -> None:
        """Release resources; repeated calls must be safe."""
        ...

    def __enter__(self) -> Self:
        """Enter the synchronous store context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the store when leaving its context."""
        ...


def validate_store_configuration(
    *,
    collection_name: str,
    dimension: int,
) -> None:
    """Validate constructor values shared by every adapter."""
    if not collection_name.strip():
        raise VectorStoreContractError("collection name must not be empty")
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise VectorStoreContractError("vector dimension must be an integer")
    if dimension <= 0:
        raise VectorStoreContractError("vector dimension must be positive")


def validate_replacement(
    chunks: Sequence[PolicyChunk],
    *,
    expected_dimension: int,
) -> tuple[PolicyChunk, ...]:
    """Validate and freeze one complete replacement before any write occurs."""
    frozen = tuple(chunks)
    chunk_ids = [chunk.chunk_id for chunk in frozen]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorStoreContractError(
            "replacement chunks must have unique chunk IDs"
        )

    for index, chunk in enumerate(frozen):
        try:
            validate_embedding_vector(
                chunk.embedding,
                expected_dimension=expected_dimension,
                context=f"chunk {chunk.chunk_id!r} embedding at index {index}",
            )
        except EmbeddingContractError as error:
            raise VectorStoreContractError(str(error)) from error

    return frozen


def validate_search_request(
    query_embedding: EmbeddingVector,
    *,
    expected_dimension: int,
    k: int,
) -> EmbeddingVector:
    """Validate a query vector and the assignment's top-k boundary."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise VectorStoreContractError("k must be an integer")
    if not 1 <= k <= MAX_RETRIEVAL_RESULTS:
        raise VectorStoreContractError(
            f"k must be between 1 and {MAX_RETRIEVAL_RESULTS}"
        )

    try:
        return validate_embedding_vector(
            query_embedding,
            expected_dimension=expected_dimension,
            context="query embedding",
        )
    except EmbeddingContractError as error:
        raise VectorStoreContractError(str(error)) from error


def validate_search_results(
    results: Sequence[SearchResult],
    *,
    k: int,
) -> tuple[SearchResult, ...]:
    """Validate normalized adapter output before returning it to services."""
    frozen = tuple(results)
    if len(frozen) > k:
        raise VectorStoreContractError(
            f"search returned {len(frozen)} results for k={k}"
        )

    distances = [result.distance for result in frozen]
    if distances != sorted(distances):
        raise VectorStoreContractError(
            "search results must be sorted by ascending cosine distance"
        )

    chunk_ids = [result.chunk.chunk_id for result in frozen]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise VectorStoreContractError(
            "search results must contain unique chunk IDs"
        )

    return frozen
