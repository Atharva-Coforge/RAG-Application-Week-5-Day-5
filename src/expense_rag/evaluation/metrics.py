"""Retrieval-quality metrics for embedding-model comparison."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import GoldCase, PolicyChunk
from expense_rag.retrieval.cosine import cosine_distance, cosine_search


class QueryRetrievalEvaluation(BaseModel):
    """One gold question's exact retrieval result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    supported: bool
    expected_section: str | None
    top_sections: tuple[str, ...]
    distances: tuple[float, ...]
    expected_rank: int | None = Field(default=None, ge=1, le=3)
    distance_margin: float | None = None


class RetrievalQualityMetrics(BaseModel):
    """Aggregate supported-question metrics and per-question evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    mean_distance_margin: float
    query_results: tuple[QueryRetrievalEvaluation, ...]


def evaluate_retrieval_quality(
    cases: Sequence[GoldCase],
    query_embeddings: Sequence[EmbeddingVector],
    chunks: Sequence[PolicyChunk],
) -> RetrievalQualityMetrics:
    """Evaluate exact top-three retrieval against supported gold sections."""
    if len(cases) != len(query_embeddings):
        raise ValueError("gold cases and query embeddings must have equal lengths")
    if not cases:
        raise ValueError("at least one gold case is required")

    sections = {chunk.section for chunk in chunks}
    evaluations: list[QueryRetrievalEvaluation] = []
    supported_ranks: list[int | None] = []
    supported_margins: list[float] = []

    for case, query_embedding in zip(cases, query_embeddings, strict=True):
        results = cosine_search(query_embedding, chunks)
        top_sections = tuple(result.chunk.section for result in results)
        distances = tuple(result.distance for result in results)
        expected_rank: int | None = None
        distance_margin: float | None = None

        if case.supported:
            expected_section = case.expected_section
            if expected_section is None or expected_section not in sections:
                raise ValueError(
                    f"expected section {expected_section!r} is not in policy chunks"
                )

            try:
                expected_rank = top_sections.index(expected_section) + 1
            except ValueError:
                expected_rank = None

            all_distances = {
                chunk.section: cosine_distance(
                    query_embedding,
                    chunk.embedding,
                )
                for chunk in chunks
            }
            expected_distance = all_distances[expected_section]
            nearest_incorrect = min(
                distance
                for section, distance in all_distances.items()
                if section != expected_section
            )
            distance_margin = nearest_incorrect - expected_distance
            supported_ranks.append(expected_rank)
            supported_margins.append(distance_margin)

        evaluations.append(
            QueryRetrievalEvaluation(
                question=case.question,
                supported=case.supported,
                expected_section=case.expected_section,
                top_sections=top_sections,
                distances=distances,
                expected_rank=expected_rank,
                distance_margin=distance_margin,
            )
        )

    if not supported_ranks:
        raise ValueError("at least one supported gold case is required")

    return RetrievalQualityMetrics(
        hit_at_1=mean(rank == 1 for rank in supported_ranks),
        hit_at_3=mean(rank is not None and rank <= 3 for rank in supported_ranks),
        mean_reciprocal_rank=mean(
            0.0 if rank is None else 1.0 / rank for rank in supported_ranks
        ),
        mean_distance_margin=mean(supported_margins),
        query_results=tuple(evaluations),
    )
