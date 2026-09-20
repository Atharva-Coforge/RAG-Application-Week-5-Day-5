"""Document-to-store ingestion with injected protocol dependencies."""

from __future__ import annotations

from pathlib import Path

from expense_rag.embeddings.base import EmbeddingProvider
from expense_rag.embeddings.provider import embed_sections
from expense_rag.ingestion.parser import load_policy
from expense_rag.models import IngestionSummary, PolicyChunk
from expense_rag.vector_stores.base import VectorStore


class IngestionService:
    """Parse, batch-embed, and replace one policy document in a store."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def ingest(
        self,
        policy_path: Path,
    ) -> tuple[tuple[PolicyChunk, ...], IngestionSummary]:
        """Persist exactly six unique chunks from one policy Markdown file."""
        sections = load_policy(policy_path)
        chunks = embed_sections(sections, self._embedding_provider)
        self._vector_store.replace_all(chunks)
        return chunks, IngestionSummary(
            chunk_count=len(chunks),
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        )
