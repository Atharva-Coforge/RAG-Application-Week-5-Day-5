"""Sentence-transformers adapter used by the embedding comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from expense_rag.embeddings.base import EmbeddingVector

if TYPE_CHECKING:
    from sentence_transformers import (
        SentenceTransformer,
    )


@dataclass(frozen=True, slots=True)
class ModelInputFormat:
    """Documented query and passage prefixes for one candidate model."""

    query_prefix: str
    document_prefix: str


MODEL_INPUT_FORMATS: dict[str, ModelInputFormat] = {
    "sentence-transformers/all-MiniLM-L6-v2": ModelInputFormat(
        query_prefix="",
        document_prefix="",
    ),
    "BAAI/bge-small-en-v1.5": ModelInputFormat(
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",
    ),
    "intfloat/e5-small-v2": ModelInputFormat(
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
}


def format_document_input(model_name: str, text: str) -> str:
    """Apply the candidate model's documented passage formatting."""
    return f"{_input_format(model_name).document_prefix}{text}"


def format_query_input(model_name: str, text: str) -> str:
    """Apply the candidate model's documented query formatting."""
    return f"{_input_format(model_name).query_prefix}{text}"


class SentenceTransformerEmbeddingProvider:
    """CPU-only, synchronous provider with unit-normalized output."""

    def __init__(self, model_name: str) -> None:
        _input_format(model_name)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "embedding comparison dependencies are not installed; "
                "run 'uv sync --extra embeddings --group dev'"
            ) from error

        self._model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(
            model_name,
            device="cpu",
        )
        dimension = self._model.get_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError(
                f"model {model_name!r} did not report a valid embedding dimension"
            )
        self._dimension = dimension

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
        prepared = [
            format_document_input(self.model_name, text) for text in texts
        ]
        return self._encode(prepared)

    def embed_query(self, text: str) -> EmbeddingVector:
        prepared = format_query_input(self.model_name, text)
        vectors = self._encode([prepared])
        return vectors[0]

    def _encode(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if not texts:
            return ()

        encoded = self._model.encode(
            list(texts),
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return tuple(
            tuple(float(value) for value in vector)
            for vector in encoded
        )


def _input_format(model_name: str) -> ModelInputFormat:
    try:
        return MODEL_INPUT_FORMATS[model_name]
    except KeyError as error:
        supported = ", ".join(MODEL_INPUT_FORMATS)
        raise ValueError(
            f"unsupported comparison model {model_name!r}; choose one of {supported}"
        ) from error
