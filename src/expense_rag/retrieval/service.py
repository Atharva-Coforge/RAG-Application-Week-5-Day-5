"""Question-to-evidence retrieval with injected protocol dependencies."""

from __future__ import annotations

from expense_rag.embeddings.base import EmbeddingProvider
from expense_rag.embeddings.provider import embed_query
from expense_rag.models import SearchResult
from expense_rag.retrieval.cosine import MAX_RETRIEVAL_RESULTS
from expense_rag.vector_stores.base import VectorStore


class RetrievalService:
    """Embed one question and return validated top-k store results."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = MAX_RETRIEVAL_RESULTS,
    ) -> tuple[SearchResult, ...]:
        """Return at most three results ordered by ascending cosine distance."""
        query_embedding = embed_query(question, self._embedding_provider)
        return self._vector_store.search(query_embedding, k=top_k)
