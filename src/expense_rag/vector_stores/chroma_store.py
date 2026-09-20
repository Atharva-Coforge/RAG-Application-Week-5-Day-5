"""Persistent Chroma storage with application-side exact cosine ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import chromadb
import numpy as np

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import PolicyChunk, SearchResult
from expense_rag.retrieval.cosine import cosine_search
from expense_rag.vector_stores.base import (
    VectorStoreClosedError,
    VectorStoreError,
    validate_replacement,
    validate_search_request,
    validate_search_results,
    validate_store_configuration,
)


class ChromaVectorStore:
    """One persistent Chroma collection ranked by the exact reference scorer."""

    def __init__(
        self,
        *,
        directory: Path,
        collection_name: str,
        dimension: int,
    ) -> None:
        validate_store_configuration(
            collection_name=collection_name,
            dimension=dimension,
        )
        self._collection_name = collection_name
        self._dimension = dimension
        self._closed = False
        try:
            self._client = chromadb.PersistentClient(path=str(directory))
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError("failed to open Chroma collection") from error

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

        try:
            existing = self._collection.get()
            existing_ids = existing["ids"]
            if existing_ids:
                self._collection.delete(ids=existing_ids)

            if validated:
                embeddings = np.asarray(
                    [chunk.embedding for chunk in validated],
                    dtype=np.float32,
                )
                self._collection.add(
                    ids=[chunk.chunk_id for chunk in validated],
                    embeddings=embeddings,
                    documents=[chunk.text for chunk in validated],
                    metadatas=[
                        {
                            "document": chunk.document,
                            "version": chunk.version,
                            "section": chunk.section,
                            "section_title": chunk.section_title,
                        }
                        for chunk in validated
                    ],
                )
        except Exception as error:
            raise VectorStoreError("failed to replace Chroma records") from error

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
        chunks = self._load_all_chunks()
        if not chunks:
            return ()

        results = cosine_search(validated_query, chunks, k=k)
        return validate_search_results(results, k=k)

    def count(self) -> int:
        self._ensure_open()
        try:
            return int(self._collection.count())
        except Exception as error:
            raise VectorStoreError("failed to count Chroma records") from error

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

    def _load_all_chunks(self) -> tuple[PolicyChunk, ...]:
        try:
            payload = self._collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
        except Exception as error:
            raise VectorStoreError("failed to read Chroma records") from error

        ids = payload["ids"]
        embeddings = payload["embeddings"]
        documents = payload["documents"]
        metadatas = payload["metadatas"]
        if not ids:
            return ()
        if embeddings is None or documents is None or metadatas is None:
            raise VectorStoreError("Chroma returned incomplete stored records")
        if not (
            len(ids) == len(embeddings) == len(documents) == len(metadatas)
        ):
            raise VectorStoreError("Chroma returned inconsistent record counts")

        chunks = tuple(
            _chunk_from_chroma(
                chunk_id=chunk_id,
                embedding=embedding,
                text=text,
                metadata=metadata,
            )
            for chunk_id, embedding, text, metadata in zip(
                ids,
                embeddings,
                documents,
                metadatas,
                strict=True,
            )
        )
        validate_replacement(chunks, expected_dimension=self.dimension)
        return chunks

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreClosedError("vector store is closed")


def _chunk_from_chroma(
    *,
    chunk_id: str,
    embedding: Sequence[float],
    text: str | None,
    metadata: Mapping[str, Any] | None,
) -> PolicyChunk:
    if text is None or metadata is None:
        raise VectorStoreError("Chroma record is missing text or metadata")
    try:
        return PolicyChunk(
            chunk_id=chunk_id,
            document=str(metadata["document"]),
            version=str(metadata["version"]),
            section=str(metadata["section"]),
            section_title=str(metadata["section_title"]),
            text=text,
            embedding=tuple(float(value) for value in embedding),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VectorStoreError("Chroma record contains invalid metadata") from error
