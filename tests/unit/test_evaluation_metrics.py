"""Tests for embedding retrieval-quality metrics."""

from math import sqrt

import pytest

from expense_rag.evaluation.metrics import evaluate_retrieval_quality
from expense_rag.models import REFUSAL_ANSWER, GoldCase, PolicyChunk


def test_retrieval_metrics_use_supported_expected_sections() -> None:
    diagonal = 1.0 / sqrt(2.0)
    chunks = (
        _chunk("1", (1.0, 0.0)),
        _chunk("2", (0.0, 1.0)),
    )
    cases = (
        _supported_case("Question one", "1"),
        _supported_case("Question two", "2"),
        GoldCase(
            question="Unsupported question",
            supported=False,
            required_answer=REFUSAL_ANSWER,
            expected_section=None,
        ),
    )

    metrics = evaluate_retrieval_quality(
        cases,
        (
            (1.0, 0.0),
            (0.0, 1.0),
            (diagonal, diagonal),
        ),
        chunks,
    )

    assert metrics.hit_at_1 == 1.0
    assert metrics.hit_at_3 == 1.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.mean_distance_margin == pytest.approx(1.0)
    assert metrics.query_results[0].expected_rank == 1
    assert metrics.query_results[-1].expected_rank is None
    assert metrics.query_results[-1].distance_margin is None
    assert len(metrics.query_results[-1].top_sections) == 2


def test_retrieval_metrics_record_misses_and_negative_margin() -> None:
    chunks = (
        _chunk("1", (1.0, 0.0)),
        _chunk("2", (0.0, 1.0)),
        _chunk("3", (-1.0, 0.0)),
        _chunk("4", (0.0, -1.0)),
    )
    case = _supported_case("Question", "3")

    metrics = evaluate_retrieval_quality(
        (case,),
        ((1.0, 0.0),),
        chunks,
    )

    assert metrics.hit_at_1 == 0.0
    assert metrics.hit_at_3 == 0.0
    assert metrics.mean_reciprocal_rank == 0.0
    assert metrics.mean_distance_margin < 0.0
    assert metrics.query_results[0].expected_rank is None


def test_retrieval_metrics_reject_invalid_inputs() -> None:
    case = _supported_case("Question", "1")
    chunks = (_chunk("1", (1.0, 0.0)),)

    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_retrieval_quality((case,), (), chunks)
    with pytest.raises(ValueError, match="at least one gold"):
        evaluate_retrieval_quality((), (), chunks)
    with pytest.raises(ValueError, match="not in policy chunks"):
        evaluate_retrieval_quality(
            (_supported_case("Question", "2"),),
            ((1.0, 0.0),),
            chunks,
        )


def _supported_case(question: str, expected_section: str) -> GoldCase:
    return GoldCase(
        question=question,
        supported=True,
        required_answer="Expected answer.",
        expected_section=expected_section,
    )


def _chunk(section: str, embedding: tuple[float, ...]) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=f"Section {section}",
        text=f"Body for section {section}.",
        embedding=embedding,
    )
