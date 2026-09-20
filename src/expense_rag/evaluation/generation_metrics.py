"""Generation-quality metrics for the LLM comparison."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field

from expense_rag.models import REFUSAL_ANSWER, GoldCase, RagResponse


class GenerationCaseEvaluation(BaseModel):
    """One gold question's generation result, with answers side by side."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    supported: bool
    expected_section: str | None
    required_answer: str
    model_answer: str | None = None
    citation_section: str | None = None
    valid_structured_output: bool
    citation_correct: bool | None = None
    refusal_correct: bool | None = None
    error: str | None = None
    latency_ms: float = Field(ge=0.0)


class GenerationQualityMetrics(BaseModel):
    """Aggregate citation, refusal, and structured-output rates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid_structured_output_rate: float = Field(ge=0.0, le=1.0)
    citation_correct_rate: float = Field(ge=0.0, le=1.0)
    gym_refusal_correct: bool
    case_results: tuple[GenerationCaseEvaluation, ...]


def evaluate_generation_case(
    case: GoldCase,
    response: RagResponse | None,
    *,
    error: str | None = None,
    latency_ms: float,
) -> GenerationCaseEvaluation:
    """Score one gold question without judging answer wording."""
    valid = response is not None and error is None
    citation_section = (
        None if response is None or response.citation is None
        else response.citation.section
    )
    citation_correct: bool | None = None
    refusal_correct: bool | None = None

    if case.supported:
        citation_correct = bool(
            valid
            and response is not None
            and response.citation is not None
            and case.expected_section is not None
            and response.citation.section.startswith(f"{case.expected_section}.")
        )
    else:
        refusal_correct = bool(
            valid
            and response is not None
            and response.answer == REFUSAL_ANSWER
            and response.citation is None
        )

    return GenerationCaseEvaluation(
        question=case.question,
        supported=case.supported,
        expected_section=case.expected_section,
        required_answer=case.required_answer,
        model_answer=None if response is None else response.answer,
        citation_section=citation_section,
        valid_structured_output=valid,
        citation_correct=citation_correct,
        refusal_correct=refusal_correct,
        error=error,
        latency_ms=latency_ms,
    )


def summarize_generation_quality(
    evaluations: Sequence[GenerationCaseEvaluation],
) -> GenerationQualityMetrics:
    """Aggregate the six gold-case generation evaluations."""
    if not evaluations:
        raise ValueError("at least one generation evaluation is required")

    supported = [item for item in evaluations if item.supported]
    unsupported = [item for item in evaluations if not item.supported]
    if not supported:
        raise ValueError("at least one supported gold case is required")
    if len(unsupported) != 1:
        raise ValueError("exactly one unsupported gold case is required")

    return GenerationQualityMetrics(
        valid_structured_output_rate=mean(
            item.valid_structured_output for item in evaluations
        ),
        citation_correct_rate=mean(
            bool(item.citation_correct) for item in supported
        ),
        gym_refusal_correct=bool(unsupported[0].refusal_correct),
        case_results=tuple(evaluations),
    )
