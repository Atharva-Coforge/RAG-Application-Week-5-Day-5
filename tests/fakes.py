"""Shared test doubles for protocol-based service tests."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Self

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.env import project_root
from expense_rag.evaluation.gold_loader import load_gold_cases
from expense_rag.models import (
    REFUSAL_ANSWER,
    GenerationDecision,
    PolicyChunk,
    SearchResult,
)
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


class MappedQueryEmbeddingProvider(FakeEmbeddingProvider):
    """Return a different query vector for each gold question."""

    def __init__(
        self,
        *,
        document_vectors: tuple[EmbeddingVector, ...],
        query_vectors_by_question: dict[str, EmbeddingVector],
        model_name: str = "fake-embedding-model",
        dimension: int = 2,
    ) -> None:
        super().__init__(
            document_vectors=document_vectors,
            query_vector=document_vectors[0],
            model_name=model_name,
            dimension=dimension,
        )
        self._query_vectors_by_question = query_vectors_by_question

    def embed_query(self, text: str) -> EmbeddingVector:
        self.query_input = text
        return self._query_vectors_by_question[text]


class MappedGenerationProvider:
    """Return a structured decision based on the prompted question."""

    def __init__(self, decisions: dict[str, GenerationDecision]) -> None:
        self._decisions = decisions
        self.call_count = 0

    def generate(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> GenerationDecision:
        del system_instruction, response_schema
        self.call_count += 1
        question = user_prompt.split("Question:\n", 1)[1].split("\n\n", 1)[0].strip()
        return self._decisions[question]


def basis_vectors(count: int) -> tuple[EmbeddingVector, ...]:
    """Return ``count`` orthogonal unit vectors."""
    return tuple(
        tuple(1.0 if index == axis else 0.0 for index in range(count))
        for axis in range(count)
    )


def gold_aware_embedding_provider() -> MappedQueryEmbeddingProvider:
    """Map each gold question to the vector of its expected section."""
    cases = load_gold_cases(project_root() / "data" / "gold" / "gold-data.md")
    vectors = basis_vectors(len(cases))
    query_vectors = {
        case.question: (
            vectors[int(case.expected_section) - 1]
            if case.expected_section is not None
            else vectors[0]
        )
        for case in cases
    }
    return MappedQueryEmbeddingProvider(
        document_vectors=vectors,
        query_vectors_by_question=query_vectors,
        dimension=len(cases),
    )


def gold_aware_generation_provider() -> MappedGenerationProvider:
    """Cite the expected section, or refuse the unsupported gym question."""
    cases = load_gold_cases(project_root() / "data" / "gold" / "gold-data.md")
    decisions = {
        case.question: (
            GenerationDecision(
                supported=True,
                answer=case.required_answer,
                cited_chunk_id=f"expense-policy:v2.0:section-{case.expected_section}",
            )
            if case.supported
            else GenerationDecision(
                supported=False,
                answer=REFUSAL_ANSWER,
                cited_chunk_id=None,
            )
        )
        for case in cases
    }
    return MappedGenerationProvider(decisions)
