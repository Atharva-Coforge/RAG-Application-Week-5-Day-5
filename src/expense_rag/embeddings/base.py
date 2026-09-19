"""Provider-neutral embedding contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

type EmbeddingVector = tuple[float, ...]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Synchronous provider that returns finite unit-normalized vectors."""

    @property
    def model_name(self) -> str:
        """Return the stable provider/model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the number of values in every embedding."""
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        """Embed document texts in input order."""
        ...

    def embed_query(self, text: str) -> EmbeddingVector:
        """Embed one user query."""
        ...
