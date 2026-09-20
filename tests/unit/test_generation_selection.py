"""Tests for the selected default generation model."""

from expense_rag.env import get_generation_model
from expense_rag.generation.providers.ollama_provider import CANDIDATE_OLLAMA_MODELS


def test_selected_generation_model_is_qwen3() -> None:
    model_name = get_generation_model()
    assert model_name == "qwen3:8b"
    assert model_name in CANDIDATE_OLLAMA_MODELS
