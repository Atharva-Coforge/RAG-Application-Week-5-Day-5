"""Tests for provider-independent domain contracts."""

import json

import pytest
from pydantic import ValidationError

from expense_rag.models import (
    REFUSAL_ANSWER,
    Citation,
    GoldCase,
    PolicyChunk,
    PolicySection,
    RagResponse,
    SearchResult,
)


@pytest.fixture
def meals_chunk() -> PolicyChunk:
    """Return a complete embedded policy chunk."""
    return PolicyChunk(
        chunk_id="expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day for meals.",
        embedding=(0.12, -0.31, 0.44),
    )


def test_policy_section_represents_parser_output_without_embedding() -> None:
    section = PolicySection(
        chunk_id="expense-policy:v2.0:section-1",
        document="Employee Expense Policy",
        version="2.0",
        section="1",
        section_title="Meals",
        text="Employees may claim up to $65 per day for meals.",
    )

    assert "embedding" not in section.model_dump()
    assert "embedding" not in PolicySection.model_json_schema()["properties"]


def test_policy_chunk_supports_json_and_schema(meals_chunk: PolicyChunk) -> None:
    serialized = json.loads(meals_chunk.model_dump_json())

    assert serialized["embedding"] == [0.12, -0.31, 0.44]
    assert "embedding" in PolicyChunk.model_json_schema()["required"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", "Expense Policy"),
        ("version", "latest"),
        ("section", "0"),
        ("embedding", ()),
        ("embedding", (float("nan"),)),
        ("embedding", (float("inf"),)),
    ],
)
def test_policy_chunk_rejects_invalid_records(
    meals_chunk: PolicyChunk,
    field: str,
    value: object,
) -> None:
    invalid_data = meals_chunk.model_dump()
    invalid_data[field] = value

    with pytest.raises(ValidationError):
        PolicyChunk.model_validate(invalid_data)


def test_search_result_serializes_only_public_fields(
    meals_chunk: PolicyChunk,
) -> None:
    result = SearchResult(chunk=meals_chunk, distance=0.08)

    assert result.model_dump() == {
        "distance": 0.08,
        "section": "1. Meals",
    }


def test_supported_response_requires_grounded_citation(
    meals_chunk: PolicyChunk,
) -> None:
    result = SearchResult(chunk=meals_chunk, distance=0.08)
    response = RagResponse(
        answer="Employees may claim up to $65 per day for meals.",
        citation=Citation.from_chunk(meals_chunk),
        retrieved_chunks=(result,),
    )

    assert response.model_dump()["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }


def test_supported_response_rejects_unretrieved_citation(
    meals_chunk: PolicyChunk,
) -> None:
    with pytest.raises(ValidationError, match="retrieved chunk"):
        RagResponse(
            answer="A manager must approve this.",
            citation=Citation(
                document="Employee Expense Policy",
                version="2.0",
                section="2. Hotels",
            ),
            retrieved_chunks=(SearchResult(chunk=meals_chunk, distance=0.08),),
        )


def test_response_rejects_unsorted_or_duplicate_results(
    meals_chunk: PolicyChunk,
) -> None:
    with pytest.raises(ValidationError, match="ascending distance"):
        RagResponse(
            answer=REFUSAL_ANSWER,
            citation=None,
            retrieved_chunks=(
                SearchResult(chunk=meals_chunk, distance=0.2),
                SearchResult(chunk=meals_chunk, distance=0.1),
            ),
        )

    with pytest.raises(ValidationError, match="duplicates"):
        RagResponse(
            answer=REFUSAL_ANSWER,
            citation=None,
            retrieved_chunks=(
                SearchResult(chunk=meals_chunk, distance=0.1),
                SearchResult(chunk=meals_chunk, distance=0.2),
            ),
        )


def test_refusal_has_no_citation(meals_chunk: PolicyChunk) -> None:
    with pytest.raises(ValidationError, match="must not include a citation"):
        RagResponse(
            answer=REFUSAL_ANSWER,
            citation=Citation.from_chunk(meals_chunk),
            retrieved_chunks=(SearchResult(chunk=meals_chunk, distance=0.1),),
        )


def test_non_refusal_requires_citation(meals_chunk: PolicyChunk) -> None:
    with pytest.raises(ValidationError, match="must include a citation"):
        RagResponse(
            answer="Meals are reimbursable.",
            citation=None,
            retrieved_chunks=(SearchResult(chunk=meals_chunk, distance=0.1),),
        )


def test_gold_case_enforces_canonical_refusal() -> None:
    supported = GoldCase(
        question="How much can I spend on food?",
        supported=True,
        required_answer="$65 per day.",
        expected_section="1",
    )
    unsupported = GoldCase(
        question="Are gym memberships covered?",
        supported=False,
        required_answer=REFUSAL_ANSWER,
        expected_section=None,
    )

    assert supported.expected_section == "1"
    assert unsupported.required_answer == REFUSAL_ANSWER

    with pytest.raises(ValidationError, match="canonical refusal"):
        GoldCase(
            question="Are gym memberships covered?",
            supported=False,
            required_answer="No.",
            expected_section=None,
        )

    with pytest.raises(ValidationError, match="must define an expected section"):
        GoldCase(
            question="How much can I spend on food?",
            supported=True,
            required_answer="$65 per day.",
            expected_section=None,
        )

    with pytest.raises(ValidationError, match="must not define an expected section"):
        GoldCase(
            question="Are gym memberships covered?",
            supported=False,
            required_answer=REFUSAL_ANSWER,
            expected_section="1",
        )
