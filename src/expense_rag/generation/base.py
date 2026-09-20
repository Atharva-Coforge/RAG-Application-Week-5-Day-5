"""Provider-neutral grounded-generation contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from expense_rag.models import GenerationDecision


@runtime_checkable
class GenerationProvider(Protocol):
    """Synchronous provider that returns a structured generation decision."""

    def generate(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> GenerationDecision:
        """Return a validated decision from system text, user text, and schema."""
        ...
