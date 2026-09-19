"""Shared contract tests for every vector-store adapter."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

import pytest

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import PolicyChunk, SearchResult
from expense_rag.retrieval.cosine import cosine_search
from expense_rag.vector_stores.base import (
    VectorStore,
    VectorStoreClosedError,
    VectorStoreContractError,
    validate_replacement,
    validate_search_request,
    validate_search_results,
    validate_store_configuration,
)
from expense_rag.vector_stores.factory import VectorStoreFactory


class InMemoryContractStore:
    """Test-only adapter proving the shared contract is implementable."""

    def __init__(self, *, collection_name: str, dimension: int) -> None:
        validate_store_configuration(
            collection_name=collection_name,
            dimension=dimension,
        )
        self._collection_name = collection_name
        self._dimension = dimension
        self._chunks: dict[str, PolicyChunk] = {}
        self._closed = False

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def replace_all(self, chunks: Sequence[PolicyChunk]) -> None:
        self._ensure_open()
        validated = validate_replacement(
            chunks,
            expected_dimension=self.dimension,
        )
        replacement = {chunk.chunk_id: chunk for chunk in validated}
        self._chunks = replacement

    def search(
        self,
        query_embedding: EmbeddingVector,
        *,
        k: int = 3,
    ) -> tuple[SearchResult, ...]:
        self._ensure_open()
        validated_query = validate_search_request(
            query_embedding,
            expected_dimension=self.dimension,
            k=k,
        )
        if not self._chunks:
            return ()

        results = cosine_search(
            validated_query,
            tuple(self._chunks.values()),
            k=k,
        )
        return validate_search_results(results, k=k)

    def count(self) -> int:
        self._ensure_open()
        return len(self._chunks)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreClosedError("vector store is closed")


@pytest.fixture
def store_factory() -> VectorStoreFactory:
    return InMemoryContractStore


@pytest.fixture
def store(store_factory: VectorStoreFactory) -> VectorStore:
    return store_factory(collection_name="expense-policy", dimension=3)


def test_store_implements_runtime_protocol(store: VectorStore) -> None:
    assert isinstance(store, VectorStore)
    assert store.collection_name == "expense-policy"
    assert store.dimension == 3


def test_replace_all_preserves_complete_records(store: VectorStore) -> None:
    chunks = (
        _chunk("1", (1.0, 0.0, 0.0)),
        _chunk("2", (0.0, 1.0, 0.0)),
    )

    store.replace_all(chunks)
    results = store.search((1.0, 0.0, 0.0), k=2)

    assert store.count() == 2
    assert results[0].chunk == chunks[0]
    assert results[1].chunk == chunks[1]


def test_replace_all_is_idempotent_and_removes_stale_chunks(
    store: VectorStore,
) -> None:
    original = (
        _chunk("1", (1.0, 0.0, 0.0)),
        _chunk("2", (0.0, 1.0, 0.0)),
    )
    replacement = (_chunk("2", (0.0, 1.0, 0.0)),)

    store.replace_all(original)
    store.replace_all(original)
    assert store.count() == 2

    store.replace_all(replacement)

    assert store.count() == 1
    assert store.search((0.0, 1.0, 0.0))[0].chunk.chunk_id == (
        "expense-policy:v2.0:section-2"
    )


def test_empty_replacement_clears_store_and_empty_search_returns_nothing(
    store: VectorStore,
) -> None:
    store.replace_all((_chunk("1", (1.0, 0.0, 0.0)),))

    store.replace_all(())

    assert store.count() == 0
    assert store.search((1.0, 0.0, 0.0)) == ()


def test_invalid_replacement_is_atomic(store: VectorStore) -> None:
    original = _chunk("1", (1.0, 0.0, 0.0))
    store.replace_all((original,))

    with pytest.raises(VectorStoreContractError, match="dimension 2"):
        store.replace_all(
            (
                _chunk("2", (0.0, 1.0, 0.0)),
                _chunk("3", (1.0, 0.0)),
            )
        )

    assert store.count() == 1
    assert store.search((1.0, 0.0, 0.0))[0].chunk == original


def test_replacement_rejects_duplicate_ids(store: VectorStore) -> None:
    chunk = _chunk("1", (1.0, 0.0, 0.0))

    with pytest.raises(VectorStoreContractError, match="unique chunk IDs"):
        store.replace_all((chunk, chunk))


def test_search_matches_exact_cosine_reference(store: VectorStore) -> None:
    chunks = (
        _chunk("3", (0.0, 0.0, 1.0)),
        _chunk("1", (1.0, 0.0, 0.0)),
        _chunk("2", (0.0, 1.0, 0.0)),
    )
    query = (1.0, 0.0, 0.0)
    store.replace_all(chunks)

    actual = store.search(query)
    expected = cosine_search(query, chunks)

    assert [result.chunk.chunk_id for result in actual] == [
        result.chunk.chunk_id for result in expected
    ]
    assert [result.distance for result in actual] == pytest.approx(
        [result.distance for result in expected],
        abs=1e-6,
    )


@pytest.mark.parametrize("k", [0, 4, True, 1.5])
def test_search_rejects_invalid_k(
    store: VectorStore,
    k: object,
) -> None:
    with pytest.raises(VectorStoreContractError, match="k must"):
        store.search((1.0, 0.0, 0.0), k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ((1.0, 0.0), "dimension 2"),
        ((1.0, 1.0, 0.0), "unit-normalized"),
        ((float("nan"), 0.0, 0.0), "non-finite"),
    ],
)
def test_search_rejects_invalid_query_vectors(
    store: VectorStore,
    query: EmbeddingVector,
    message: str,
) -> None:
    with pytest.raises(VectorStoreContractError, match=message):
        store.search(query)


def test_context_manager_and_close_are_safe(
    store_factory: VectorStoreFactory,
) -> None:
    store = store_factory(collection_name="expense-policy", dimension=3)

    with store as entered:
        assert entered is store
        assert entered.count() == 0

    store.close()
    with pytest.raises(VectorStoreClosedError, match="closed"):
        store.count()
    with pytest.raises(VectorStoreClosedError, match="closed"):
        store.replace_all(())
    with pytest.raises(VectorStoreClosedError, match="closed"):
        store.search((1.0, 0.0, 0.0))
    with pytest.raises(VectorStoreClosedError, match="closed"):
        store.__enter__()


@pytest.mark.parametrize(
    ("collection_name", "dimension", "message"),
    [
        ("", 3, "collection name"),
        ("expense-policy", 0, "positive"),
        ("expense-policy", True, "integer"),
        ("expense-policy", 3.5, "integer"),
    ],
)
def test_store_rejects_invalid_configuration(
    store_factory: VectorStoreFactory,
    collection_name: str,
    dimension: object,
    message: str,
) -> None:
    with pytest.raises(VectorStoreContractError, match=message):
        store_factory(
            collection_name=collection_name,
            dimension=dimension,  # type: ignore[arg-type]
        )


def _chunk(
    section: str,
    embedding: EmbeddingVector,
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
