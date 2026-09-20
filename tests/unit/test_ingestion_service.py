"""Unit tests for IngestionService with fake provider and in-memory store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import get_type_hints

from expense_rag.embeddings.base import EmbeddingProvider, EmbeddingVector
from expense_rag.env import project_root
from expense_rag.ingestion.parser import EXPECTED_SECTION_COUNT
from expense_rag.ingestion.service import IngestionService
from expense_rag.vector_stores.base import VectorStore
from tests.fakes import FakeEmbeddingProvider, InMemoryVectorStore

_POLICY_PATH = project_root() / "data" / "policy.md"
_EXPECTED_IDS = tuple(
    f"expense-policy:v2.0:section-{index}"
    for index in range(1, EXPECTED_SECTION_COUNT + 1)
)


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    """Record how many times document embedding is invoked."""

    def __init__(
        self,
        *,
        document_vectors: tuple[EmbeddingVector, ...],
        query_vector: EmbeddingVector = (1.0, 0.0),
        model_name: str = "fake-embedding-model",
        dimension: int = 2,
    ) -> None:
        super().__init__(
            document_vectors=document_vectors,
            query_vector=query_vector,
            model_name=model_name,
            dimension=dimension,
        )
        self.embed_documents_calls = 0

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        self.embed_documents_calls += 1
        return super().embed_documents(texts)


def _basis_vectors(count: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if index == axis else 0.0 for index in range(count))
        for axis in range(count)
    )


def test_ingestion_service_depends_only_on_protocols() -> None:
    hints = get_type_hints(IngestionService.__init__)

    assert hints["embedding_provider"] is EmbeddingProvider
    assert hints["vector_store"] is VectorStore


def test_ingest_stores_exactly_six_unique_stable_ids() -> None:
    provider = CountingEmbeddingProvider(
        document_vectors=_basis_vectors(EXPECTED_SECTION_COUNT),
        dimension=EXPECTED_SECTION_COUNT,
    )
    store = InMemoryVectorStore(
        collection_name="expense-policy",
        dimension=EXPECTED_SECTION_COUNT,
    )
    service = IngestionService(embedding_provider=provider, vector_store=store)

    chunks, summary = service.ingest(_POLICY_PATH)

    assert provider.embed_documents_calls == 1
    assert len(provider.document_inputs) == EXPECTED_SECTION_COUNT
    assert len(chunks) == EXPECTED_SECTION_COUNT
    assert summary.chunk_count == EXPECTED_SECTION_COUNT
    assert summary.chunk_ids == _EXPECTED_IDS
    assert tuple(chunk.chunk_id for chunk in chunks) == _EXPECTED_IDS
    assert store.count() == EXPECTED_SECTION_COUNT
    assert len({chunk.chunk_id for chunk in chunks}) == EXPECTED_SECTION_COUNT


def test_reingest_replaces_without_duplicates() -> None:
    provider = CountingEmbeddingProvider(
        document_vectors=_basis_vectors(EXPECTED_SECTION_COUNT),
        dimension=EXPECTED_SECTION_COUNT,
    )
    store = InMemoryVectorStore(
        collection_name="expense-policy",
        dimension=EXPECTED_SECTION_COUNT,
    )
    service = IngestionService(embedding_provider=provider, vector_store=store)

    first_chunks, _ = service.ingest(_POLICY_PATH)
    second_chunks, summary = service.ingest(_POLICY_PATH)

    assert provider.embed_documents_calls == 2
    assert store.count() == EXPECTED_SECTION_COUNT
    assert summary.chunk_ids == _EXPECTED_IDS
    assert tuple(chunk.chunk_id for chunk in first_chunks) == tuple(
        chunk.chunk_id for chunk in second_chunks
    )
