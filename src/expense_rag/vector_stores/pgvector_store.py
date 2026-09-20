"""Transactional PostgreSQL pgvector exact-cosine adapter."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from expense_rag.embeddings.base import EmbeddingVector
from expense_rag.models import PolicyChunk, SearchResult
from expense_rag.vector_stores.base import (
    VectorStoreClosedError,
    VectorStoreError,
    validate_replacement,
    validate_search_request,
    validate_search_results,
    validate_store_configuration,
)

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS policy_chunks (
    collection_name TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document TEXT NOT NULL,
    version TEXT NOT NULL,
    section TEXT NOT NULL,
    section_title TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    PRIMARY KEY (collection_name, chunk_id)
)
"""


class PgVectorStore:
    """One logical collection in a shared PostgreSQL policy-chunks table."""

    def __init__(
        self,
        *,
        database_url: str,
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
            self._connection = psycopg.connect(
                database_url,
                autocommit=True,
                row_factory=dict_row,
            )
            self._connection.execute(_CREATE_EXTENSION)
            register_vector(self._connection)
            self._connection.execute(_CREATE_TABLE)
        except psycopg.Error as error:
            raise VectorStoreError("failed to initialize pgvector store") from error

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
            with self._connection.transaction():
                self._connection.execute(
                    "DELETE FROM policy_chunks WHERE collection_name = %s",
                    (self.collection_name,),
                )
                if validated:
                    with self._connection.cursor() as cursor:
                        cursor.executemany(
                            """
                            INSERT INTO policy_chunks (
                                collection_name,
                                chunk_id,
                                document,
                                version,
                                section,
                                section_title,
                                text,
                                embedding
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            [
                                (
                                    self.collection_name,
                                    chunk.chunk_id,
                                    chunk.document,
                                    chunk.version,
                                    chunk.section,
                                    chunk.section_title,
                                    chunk.text,
                                    Vector(list(chunk.embedding)),
                                )
                                for chunk in validated
                            ],
                        )
        except psycopg.Error as error:
            raise VectorStoreError("failed to replace pgvector records") from error

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
        try:
            rows = self._connection.execute(
                """
                SELECT
                    chunk_id,
                    document,
                    version,
                    section,
                    section_title,
                    text,
                    embedding,
                    embedding <=> %s AS distance
                FROM policy_chunks
                WHERE collection_name = %s
                ORDER BY distance ASC, chunk_id ASC
                LIMIT %s
                """,
                (
                    Vector(list(validated_query)),
                    self.collection_name,
                    k,
                ),
            ).fetchall()
        except psycopg.Error as error:
            raise VectorStoreError("failed to search pgvector records") from error

        results = tuple(_result_from_row(row) for row in rows)
        return validate_search_results(results, k=k)

    def count(self) -> int:
        self._ensure_open()
        try:
            row = self._connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM policy_chunks
                WHERE collection_name = %s
                """,
                (self.collection_name,),
            ).fetchone()
        except psycopg.Error as error:
            raise VectorStoreError("failed to count pgvector records") from error
        if row is None:
            raise VectorStoreError("pgvector count query returned no row")
        return int(row["count"])

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
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


def _result_from_row(row: dict[str, Any]) -> SearchResult:
    try:
        chunk = PolicyChunk(
            chunk_id=str(row["chunk_id"]),
            document=str(row["document"]),
            version=str(row["version"]),
            section=str(row["section"]),
            section_title=str(row["section_title"]),
            text=str(row["text"]),
            embedding=_embedding_from_row(row["embedding"]),
        )
        return SearchResult(
            chunk=chunk,
            distance=float(row["distance"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise VectorStoreError("pgvector returned an invalid record") from error


def _embedding_from_row(value: Any) -> tuple[float, ...]:
    if hasattr(value, "to_list"):
        return tuple(float(item) for item in value.to_list())
    if isinstance(value, (list, tuple)):
        return tuple(float(item) for item in value)
    raise TypeError(f"unsupported embedding type: {type(value)!r}")
