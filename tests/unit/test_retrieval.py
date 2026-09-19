"""Tests for exact cosine distance and reference retrieval."""

from math import sqrt

import pytest

from expense_rag.models import PolicyChunk
from expense_rag.retrieval.cosine import (
    CosineSearchError,
    cosine_distance,
    cosine_search,
)


def test_cosine_distance_known_values() -> None:
    assert cosine_distance((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert cosine_distance((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)
    assert cosine_distance((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(2.0)


def test_cosine_distance_normalizes_inputs_during_calculation() -> None:
    assert cosine_distance((2.0, 0.0), (4.0, 0.0)) == pytest.approx(0.0)


def test_cosine_search_returns_top_three_in_ascending_order() -> None:
    diagonal = 1.0 / sqrt(2.0)
    chunks = (
        _chunk("4", (-1.0, 0.0)),
        _chunk("2", (diagonal, diagonal)),
        _chunk("3", (0.0, 1.0)),
        _chunk("1", (1.0, 0.0)),
    )

    results = cosine_search((1.0, 0.0), chunks)

    assert [result.chunk.section for result in results] == ["1", "2", "3"]
    assert [result.distance for result in results] == sorted(
        result.distance for result in results
    )
    assert all(isinstance(result.distance, float) for result in results)


def test_cosine_search_uses_chunk_id_to_break_distance_ties() -> None:
    results = cosine_search(
        (1.0, 0.0),
        (
            _chunk("2", (0.0, 1.0)),
            _chunk("1", (0.0, 1.0)),
        ),
        k=2,
    )

    assert [result.chunk.section for result in results] == ["1", "2"]


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ((), (1.0,), "non-empty"),
        ((1.0,), (1.0, 0.0), "dimensions differ"),
        ((0.0, 0.0), (1.0, 0.0), "zero vectors"),
        ((float("nan"), 0.0), (1.0, 0.0), "finite vectors"),
    ],
)
def test_cosine_distance_rejects_invalid_vectors(
    left: tuple[float, ...],
    right: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(CosineSearchError, match=message):
        cosine_distance(left, right)


@pytest.mark.parametrize("k", [0, 4, True, 1.5])
def test_cosine_search_rejects_invalid_k(k: object) -> None:
    with pytest.raises(CosineSearchError, match="k must"):
        cosine_search((1.0, 0.0), (_chunk("1", (1.0, 0.0)),), k=k)  # type: ignore[arg-type]


def test_cosine_search_rejects_empty_or_duplicate_chunks() -> None:
    with pytest.raises(CosineSearchError, match="at least one"):
        cosine_search((1.0, 0.0), ())

    duplicate = _chunk("1", (1.0, 0.0))
    with pytest.raises(CosineSearchError, match="unique chunk IDs"):
        cosine_search((1.0, 0.0), (duplicate, duplicate))


def _chunk(
    section: str,
    embedding: tuple[float, ...],
) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=f"Section {section}",
        text=f"Body for section {section}.",
        embedding=embedding,
    )
