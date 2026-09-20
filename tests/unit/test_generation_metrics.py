"""Tests for generation quality scoring."""

from expense_rag.evaluation.generation_metrics import (
    evaluate_generation_case,
    summarize_generation_quality,
)
from expense_rag.models import (
    REFUSAL_ANSWER,
    Citation,
    GoldCase,
    PolicyChunk,
    RagResponse,
    SearchResult,
)


def test_supported_citation_matches_expected_section() -> None:
    case = GoldCase(
        question="How much can I spend on food each day?",
        supported=True,
        required_answer="Employees may claim up to $65 per day.",
        expected_section="1",
    )
    response = RagResponse(
        answer="Employees may claim up to $65 per day.",
        citation=Citation(
            document="Employee Expense Policy",
            version="2.0",
            section="1. Meals",
        ),
        retrieved_chunks=(_result("1", "Meals"),),
    )

    evaluation = evaluate_generation_case(case, response, latency_ms=12.0)

    assert evaluation.valid_structured_output is True
    assert evaluation.citation_correct is True
    assert evaluation.refusal_correct is None
    assert evaluation.required_answer == case.required_answer
    assert evaluation.model_answer == response.answer


def test_gym_refusal_requires_canonical_answer_and_null_citation() -> None:
    case = GoldCase(
        question="Does the company reimburse gym memberships?",
        supported=False,
        required_answer=REFUSAL_ANSWER,
        expected_section=None,
    )
    response = RagResponse(
        answer=REFUSAL_ANSWER,
        citation=None,
        retrieved_chunks=(_result("4", "Ground Transportation"),),
    )

    evaluation = evaluate_generation_case(case, response, latency_ms=9.0)

    assert evaluation.refusal_correct is True
    assert evaluation.citation_correct is None


def test_malformed_generation_is_not_valid_output() -> None:
    case = GoldCase(
        question="How much can I spend on food each day?",
        supported=True,
        required_answer="Employees may claim up to $65 per day.",
        expected_section="1",
    )

    evaluation = evaluate_generation_case(
        case,
        None,
        error="generation provider returned malformed output",
        latency_ms=4.0,
    )

    assert evaluation.valid_structured_output is False
    assert evaluation.citation_correct is False
    assert evaluation.model_answer is None


def test_summarize_generation_quality_rates() -> None:
    supported = evaluate_generation_case(
        GoldCase(
            question="How much can I spend on food each day?",
            supported=True,
            required_answer="Employees may claim up to $65 per day.",
            expected_section="1",
        ),
        RagResponse(
            answer="Employees may claim up to $65 per day.",
            citation=Citation(
                document="Employee Expense Policy",
                version="2.0",
                section="1. Meals",
            ),
            retrieved_chunks=(_result("1", "Meals"),),
        ),
        latency_ms=10.0,
    )
    gym = evaluate_generation_case(
        GoldCase(
            question="Does the company reimburse gym memberships?",
            supported=False,
            required_answer=REFUSAL_ANSWER,
            expected_section=None,
        ),
        RagResponse(
            answer=REFUSAL_ANSWER,
            citation=None,
            retrieved_chunks=(_result("4", "Ground Transportation"),),
        ),
        latency_ms=8.0,
    )

    summary = summarize_generation_quality((supported, gym))

    assert summary.valid_structured_output_rate == 1.0
    assert summary.citation_correct_rate == 1.0
    assert summary.gym_refusal_correct is True


def _result(section: str, title: str) -> SearchResult:
    chunk = PolicyChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=f"Body for section {section}.",
        embedding=(1.0, 0.0),
    )
    return SearchResult(chunk=chunk, distance=0.1)
