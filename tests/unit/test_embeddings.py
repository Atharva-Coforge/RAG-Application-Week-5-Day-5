"""Tests for the provider-neutral embedding boundary."""

from collections.abc import Sequence
from typing import cast

import pytest

from expense_rag.embeddings.base import EmbeddingProvider
from expense_rag.embeddings.provider import (
    EmbeddingContractError,
    embed_query,
    embed_sections,
    format_section_for_embedding,
    validate_embedding_vector,
)
from expense_rag.embeddings.sentence_transformer import (
    MODEL_INPUT_FORMATS,
    format_document_input,
    format_query_input,
)
from expense_rag.env import get_embedding_model
from expense_rag.models import PolicySection
from tests.fakes import FakeEmbeddingProvider


@pytest.fixture
def sections() -> tuple[PolicySection, ...]:
    return (
        PolicySection(
            chunk_id="expense-policy:v2.0:section-1",
            document="Employee Expense Policy",
            version="2.0",
            section="1",
            section_title="Meals",
            text="Employees may claim up to $65 per day.",
        ),
        PolicySection(
            chunk_id="expense-policy:v2.0:section-2",
            document="Employee Expense Policy",
            version="2.0",
            section="2",
            section_title="Hotels",
            text="Hotels are reimbursable up to $225 per night.",
        ),
    )


def test_embedding_provider_protocol_is_runtime_checkable() -> None:
    provider = FakeEmbeddingProvider(document_vectors=((1.0, 0.0),))

    assert isinstance(provider, EmbeddingProvider)


def test_section_embedding_text_has_selected_canonical_format(
    sections: tuple[PolicySection, ...],
) -> None:
    assert format_section_for_embedding(sections[0]) == (
        "1\n"
        "Meals\n"
        "Employees may claim up to $65 per day."
    )


def test_embed_sections_preserves_metadata_and_input_order(
    sections: tuple[PolicySection, ...],
) -> None:
    provider = FakeEmbeddingProvider(
        document_vectors=((1.0, 0.0), (0.0, 1.0))
    )

    chunks = embed_sections(sections, provider)

    assert provider.document_inputs == (
        "1\nMeals\nEmployees may claim up to $65 per day.",
        "2\nHotels\nHotels are reimbursable up to $225 per night.",
    )
    assert [chunk.chunk_id for chunk in chunks] == [
        section.chunk_id for section in sections
    ]
    assert [chunk.text for chunk in chunks] == [
        section.text for section in sections
    ]
    assert [chunk.embedding for chunk in chunks] == [
        (1.0, 0.0),
        (0.0, 1.0),
    ]


def test_embed_query_validates_and_preserves_question() -> None:
    provider = FakeEmbeddingProvider(
        document_vectors=((1.0, 0.0),),
        query_vector=(0.6, 0.8),
    )

    vector = embed_query("How much can I spend on food?", provider)

    assert vector == (0.6, 0.8)
    assert provider.query_input == "How much can I spend on food?"


@pytest.mark.parametrize(
    ("provider", "message"),
    [
        (
            FakeEmbeddingProvider(document_vectors=()),
            "different number of vectors",
        ),
        (
            FakeEmbeddingProvider(document_vectors=((1.0, 0.0, 0.0),)),
            "dimension 3; expected 2",
        ),
        (
            FakeEmbeddingProvider(document_vectors=((1.0, 1.0),)),
            "must be unit-normalized",
        ),
        (
            FakeEmbeddingProvider(document_vectors=((float("nan"), 0.0),)),
            "non-finite",
        ),
        (
            FakeEmbeddingProvider(
                document_vectors=((1.0, 0.0),),
                model_name=" ",
            ),
            "model name",
        ),
        (
            FakeEmbeddingProvider(
                document_vectors=((1.0, 0.0),),
                dimension=0,
            ),
            "dimension must be positive",
        ),
    ],
)
def test_embed_sections_rejects_provider_contract_violations(
    sections: tuple[PolicySection, ...],
    provider: FakeEmbeddingProvider,
    message: str,
) -> None:
    selected_sections = sections[:1]

    with pytest.raises(EmbeddingContractError, match=message):
        embed_sections(selected_sections, provider)


def test_embedding_workflow_rejects_empty_inputs(
    sections: tuple[PolicySection, ...],
) -> None:
    provider = FakeEmbeddingProvider(document_vectors=((1.0, 0.0),))

    with pytest.raises(EmbeddingContractError, match="policy section"):
        embed_sections((), provider)
    with pytest.raises(EmbeddingContractError, match="question"):
        embed_query(" ", provider)


@pytest.mark.parametrize("invalid_value", ["not-a-number", object()])
def test_embedding_validation_wraps_numeric_conversion_failures(
    invalid_value: object,
) -> None:
    malformed_vector = cast(Sequence[float], (invalid_value, 0.0))

    with pytest.raises(
        EmbeddingContractError,
        match="contains a non-numeric value",
    ):
        validate_embedding_vector(
            malformed_vector,
            expected_dimension=2,
            context="document vector 0",
        )


@pytest.mark.parametrize(
    ("model_name", "expected_query", "expected_document"),
    [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            "question",
            "document",
        ),
        (
            "BAAI/bge-small-en-v1.5",
            "Represent this sentence for searching relevant passages: question",
            "document",
        ),
        (
            "intfloat/e5-small-v2",
            "query: question",
            "passage: document",
        ),
    ],
)
def test_candidate_models_use_selected_documented_formats(
    model_name: str,
    expected_query: str,
    expected_document: str,
) -> None:
    assert format_query_input(model_name, "question") == expected_query
    assert format_document_input(model_name, "document") == expected_document


def test_selected_embedding_model_is_a_supported_candidate() -> None:
    model_name = get_embedding_model()
    assert model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert model_name in MODEL_INPUT_FORMATS
