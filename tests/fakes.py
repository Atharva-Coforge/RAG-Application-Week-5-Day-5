"""Shared test doubles for protocol-based service tests."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import PolicyChunk, SearchResult
from expense_rag.retrieval.cosine import cosine_search
from expense_rag.vector_stores.base import (
    VectorStoreClosedError,
    validate_replacement,
    validate_search_request,
    validate_search_results,
    validate_store_configuration,
)


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


class InMemoryVectorStore:
    """In-memory VectorStore used by service unit tests."""

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
        self._chunks = {chunk.chunk_id: chunk for chunk in validated}

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
