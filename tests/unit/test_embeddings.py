"""Tests for the provider-neutral embedding boundary."""

from collections.abc import Sequence

import pytest

from expense_rag.embeddings.base import EmbeddingProvider, EmbeddingVector
from expense_rag.embeddings.provider import (
    EmbeddingContractError,
    embed_query,
    embed_sections,
    format_section_for_embedding,
)
from expense_rag.models import PolicySection


class FakeEmbeddingProvider:
    """Configurable synchronous provider used to test the shared contract."""

    def __init__(
        self,
        *,
        document_vectors: tuple[EmbeddingVector, ...],
        query_vector: EmbeddingVector = (1.0, 0.0),
        model_name: str = "fake-embedding-model",
        dimension: int = 2,
    ) -> None:
        self._document_vectors = document_vectors
        self._query_vector = query_vector
        self._model_name = model_name
        self._dimension = dimension
        self.document_inputs: tuple[str, ...] = ()
        self.query_input: str | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        self.document_inputs = tuple(texts)
        return self._document_vectors

    def embed_query(self, text: str) -> EmbeddingVector:
        self.query_input = text
        return self._query_vector


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
