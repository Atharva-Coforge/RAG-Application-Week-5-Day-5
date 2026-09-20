"""Persistent exact-cosine FAISS adapter with JSON metadata."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

import faiss
import numpy as np
from pydantic import TypeAdapter, ValidationError

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import PolicyChunk, SearchResult
from expense_rag.vector_stores.base import (
    VectorStoreClosedError,
    VectorStoreContractError,
    VectorStoreError,
    validate_replacement,
    validate_search_request,
    validate_search_results,
    validate_store_configuration,
)

_CHUNK_LIST_ADAPTER = TypeAdapter(list[PolicyChunk])


class FaissVectorStore:
    """One persistent FAISS ``IndexFlatIP`` and metadata sidecar."""

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
        self._directory = directory
        self._collection_name = collection_name
        self._dimension = dimension
        self._index_path = directory / f"{collection_name}.faiss"
        self._metadata_path = directory / f"{collection_name}.json"
        self._closed = False
        self._index: faiss.Index
        self._chunks: list[PolicyChunk]
        self._load_or_initialize()

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

        replacement_index = faiss.IndexFlatIP(self.dimension)
        if validated:
            vectors = np.asarray(
                [chunk.embedding for chunk in validated],
                dtype=np.float32,
            )
            replacement_index.add(vectors)

        self._persist(replacement_index, validated)
        self._index = replacement_index
        self._chunks = list(validated)

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

        result_count = min(k, len(self._chunks))
        scores, indexes = self._index.search(
            np.asarray([validated_query], dtype=np.float32),
            result_count,
        )
        results = [
            SearchResult(
                chunk=self._chunks[int(index)],
                distance=1.0 - min(1.0, max(-1.0, float(score))),
            )
            for score, index in zip(scores[0], indexes[0], strict=True)
            if index >= 0
        ]
        results.sort(key=lambda result: (result.distance, result.chunk.chunk_id))
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

    def _load_or_initialize(self) -> None:
        index_exists = self._index_path.exists()
        metadata_exists = self._metadata_path.exists()
        if index_exists != metadata_exists:
            raise VectorStoreError(
                "FAISS index and metadata sidecar must either both exist or "
                "both be absent"
            )
        if not index_exists:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._chunks = []
            return

        try:
            index = faiss.read_index(str(self._index_path))
            chunks = _CHUNK_LIST_ADAPTER.validate_json(
                self._metadata_path.read_text(encoding="utf-8")
            )
        except (RuntimeError, OSError, ValidationError) as error:
            raise VectorStoreError("failed to load FAISS persistence files") from error

        if index.d != self.dimension:
            raise VectorStoreContractError(
                f"persisted FAISS dimension is {index.d}; expected {self.dimension}"
            )
        if index.ntotal != len(chunks):
            raise VectorStoreError(
                "FAISS index count does not match metadata sidecar count"
            )
        validate_replacement(chunks, expected_dimension=self.dimension)
        self._index = index
        self._chunks = chunks

    def _persist(
        self,
        index: faiss.Index,
        chunks: Sequence[PolicyChunk],
    ) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        index_temp = _temporary_path(self._directory, ".faiss.tmp")
        metadata_temp = _temporary_path(self._directory, ".json.tmp")
        try:
            faiss.write_index(index, str(index_temp))
            metadata_temp.write_text(
                _CHUNK_LIST_ADAPTER.dump_json(list(chunks), indent=2).decode(),
                encoding="utf-8",
            )
            os.replace(index_temp, self._index_path)
            os.replace(metadata_temp, self._metadata_path)
        except (RuntimeError, OSError) as error:
            raise VectorStoreError("failed to persist FAISS replacement") from error
        finally:
            index_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreClosedError("vector store is closed")


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(descriptor)
    return Path(name)
