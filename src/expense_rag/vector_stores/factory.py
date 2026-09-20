"""Construction of configured vector-store adapters."""

from __future__ import annotations

from typing import Protocol

from expense_rag.config import Settings, VectorBackend
from expense_rag.vector_stores.base import VectorStore

SUPPORTED_VECTOR_STORES = tuple(backend.value for backend in VectorBackend)


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


def build_vector_store(settings: Settings) -> VectorStore:
    """Create the configured store without leaking adapter imports to callers."""
    if settings.vector_backend is VectorBackend.CHROMA:
        return _build_chroma_store(settings)
    if settings.vector_backend is VectorBackend.FAISS:
        return _build_faiss_store(settings)
    if settings.vector_backend is VectorBackend.PGVECTOR:
        return _build_pgvector_store(settings)
    raise ValueError(f"unsupported vector backend {settings.vector_backend}")


def _build_chroma_store(settings: Settings) -> VectorStore:
    try:
        from expense_rag.vector_stores.chroma_store import ChromaVectorStore
    except ImportError as error:
        raise RuntimeError(
            "Chroma comparison dependencies are not installed; "
            "run 'uv sync --extra chroma --group dev'"
        ) from error

    settings.chroma_directory.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(
        directory=settings.chroma_directory,
        collection_name=settings.collection_name,
        dimension=settings.embedding_dimension,
    )


def _build_faiss_store(settings: Settings) -> VectorStore:
    try:
        from expense_rag.vector_stores.faiss_store import FaissVectorStore
    except ImportError as error:
        raise RuntimeError(
            "FAISS comparison dependencies are not installed; "
            "run 'uv sync --extra faiss --group dev'"
        ) from error

    settings.faiss_directory.mkdir(parents=True, exist_ok=True)
    return FaissVectorStore(
        directory=settings.faiss_directory,
        collection_name=settings.collection_name,
        dimension=settings.embedding_dimension,
    )


def _build_pgvector_store(settings: Settings) -> VectorStore:
    try:
        from expense_rag.vector_stores.pgvector_store import PgVectorStore
    except ImportError as error:
        raise RuntimeError(
            "pgvector dependencies are not installed; "
            "run 'uv sync --extra pgvector --group dev'"
        ) from error

    if settings.database_url is None:
        from expense_rag.env import MissingDatabaseUrlError

        raise MissingDatabaseUrlError(
            "DATABASE_URL is required. "
            "Copy .env.example to .env and set the selected value."
        )

    return PgVectorStore(
        database_url=settings.database_url.get_secret_value(),
        collection_name=settings.collection_name,
        dimension=settings.embedding_dimension,
    )
