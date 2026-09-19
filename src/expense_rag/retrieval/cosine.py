"""Exact cosine-distance reference implementation."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import PolicyChunk, SearchResult

MAX_RETRIEVAL_RESULTS = 3


class CosineSearchError(ValueError):
    """Raised when exact cosine search receives invalid input."""


def cosine_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return canonical cosine distance in the inclusive range [0, 2]."""
    if not left or not right:
        raise CosineSearchError("cosine distance requires non-empty vectors")
    if len(left) != len(right):
        raise CosineSearchError(
            f"vector dimensions differ: {len(left)} and {len(right)}"
        )
    if not all(isfinite(value) for value in (*left, *right)):
        raise CosineSearchError("cosine distance requires finite vectors")

    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise CosineSearchError("cosine distance is undefined for zero vectors")

    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    ) / (left_norm * right_norm)
    bounded_similarity = min(1.0, max(-1.0, similarity))
    return 1.0 - bounded_similarity


def cosine_search(
    query_embedding: EmbeddingVector,
    chunks: Sequence[PolicyChunk],
    *,
    k: int = MAX_RETRIEVAL_RESULTS,
) -> tuple[SearchResult, ...]:
    """Rank unique chunks by exact cosine distance with deterministic ties."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise CosineSearchError("k must be an integer")
    if not 1 <= k <= MAX_RETRIEVAL_RESULTS:
        raise CosineSearchError(
            f"k must be between 1 and {MAX_RETRIEVAL_RESULTS}"
        )
    if not chunks:
        raise CosineSearchError("at least one policy chunk is required")

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise CosineSearchError("policy chunks must have unique chunk IDs")

    ranked = [
        SearchResult(
            chunk=chunk,
            distance=cosine_distance(query_embedding, chunk.embedding),
        )
        for chunk in chunks
    ]
    ranked.sort(key=lambda result: (result.distance, result.chunk.chunk_id))
    return tuple(ranked[:k])
