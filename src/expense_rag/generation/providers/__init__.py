"""Generation provider adapters and the selected-provider factory."""

from expense_rag.config import Settings
from expense_rag.generation.base import GenerationProvider


def build_generation_provider(settings: Settings) -> GenerationProvider:
    """Build the configured generation provider without exposing SDK imports."""
    from expense_rag.generation.providers.ollama_provider import (
        OllamaGenerationProvider,
    )

    return OllamaGenerationProvider(
        model_name=settings.generation_model,
        host=settings.ollama_host,
    )
