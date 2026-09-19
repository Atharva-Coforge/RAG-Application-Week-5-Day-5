"""Construction contract for vector-store adapters."""

from typing import Protocol

from expense_rag.vector_stores.base import VectorStore


class VectorStoreFactory(Protocol):
    """Build one store scoped to a logical collection and vector dimension."""

    def __call__(
        self,
        *,
        collection_name: str,
        dimension: int,
    ) -> VectorStore:
        """Create a configured vector store."""
        ...
