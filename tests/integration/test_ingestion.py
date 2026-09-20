"""pgvector ingestion round-trip gated by extras and .env configuration."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from expense_rag.embeddings.provider import EmbeddingContractError
from expense_rag.env import (
    MissingDatabaseUrlError,
    MissingVectorStoreError,
    get_database_url,
    get_vector_store,
    project_root,
)
from expense_rag.ingestion.parser import EXPECTED_SECTION_COUNT
from expense_rag.ingestion.service import IngestionService
from expense_rag.retrieval.service import RetrievalService
from expense_rag.vector_stores.base import VectorStore, VectorStoreError
from tests.fakes import FakeEmbeddingProvider

_POLICY_PATH = project_root() / "data" / "policy.md"
_COLLECTION_NAME = "expense-policy-ingestion-test"
_EXPECTED_IDS = tuple(
    f"expense-policy:v2.0:section-{index}"
    for index in range(1, EXPECTED_SECTION_COUNT + 1)
)


def _basis_vectors(count: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if index == axis else 0.0 for index in range(count))
        for axis in range(count)
    )


@pytest.fixture
def pgvector_store() -> Iterator[VectorStore]:
    try:
        backend = get_vector_store()
    except MissingVectorStoreError as error:
        pytest.skip(str(error))
    if backend != "pgvector":
        pytest.skip(
            "ingestion integration covers the selected pgvector pipeline only"
        )

    try:
        database_url = get_database_url()
    except MissingDatabaseUrlError as error:
        pytest.skip(str(error))

    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")

    from expense_rag.vector_stores.pgvector_store import PgVectorStore

    try:
        store = PgVectorStore(
            database_url=database_url,
            collection_name=_COLLECTION_NAME,
            dimension=EXPECTED_SECTION_COUNT,
        )
    except VectorStoreError as error:
        pytest.skip(f"PostgreSQL is not reachable: {error}")

    store.replace_all(())
    try:
        yield store
    finally:
        store.replace_all(())
        store.close()


def test_pgvector_ingest_and_retrieve_round_trip(pgvector_store: VectorStore) -> None:
    provider = FakeEmbeddingProvider(
        document_vectors=_basis_vectors(EXPECTED_SECTION_COUNT),
        query_vector=_basis_vectors(EXPECTED_SECTION_COUNT)[0],
        dimension=EXPECTED_SECTION_COUNT,
    )
    ingestion = IngestionService(
        embedding_provider=provider,
        vector_store=pgvector_store,
    )
    retrieval = RetrievalService(
        embedding_provider=provider,
        vector_store=pgvector_store,
    )

    chunks, summary = ingestion.ingest(_POLICY_PATH)
    results = retrieval.retrieve("What is the meal allowance?")

    assert summary.chunk_count == EXPECTED_SECTION_COUNT
    assert summary.chunk_ids == _EXPECTED_IDS
    assert tuple(chunk.chunk_id for chunk in chunks) == _EXPECTED_IDS
    assert pgvector_store.count() == EXPECTED_SECTION_COUNT
    assert len(results) <= 3
    assert results[0].chunk.chunk_id == _EXPECTED_IDS[0]
    assert [result.distance for result in results] == sorted(
        result.distance for result in results
    )

    ingestion.ingest(_POLICY_PATH)
    assert pgvector_store.count() == EXPECTED_SECTION_COUNT
    assert {chunk.chunk_id for chunk in chunks} == set(_EXPECTED_IDS)


def test_retrieve_still_rejects_empty_question_with_pgvector(
    pgvector_store: VectorStore,
) -> None:
    service = RetrievalService(
        embedding_provider=FakeEmbeddingProvider(
            document_vectors=_basis_vectors(EXPECTED_SECTION_COUNT),
            dimension=EXPECTED_SECTION_COUNT,
        ),
        vector_store=pgvector_store,
    )

    with pytest.raises(EmbeddingContractError, match="question must not be empty"):
        service.retrieve("")
