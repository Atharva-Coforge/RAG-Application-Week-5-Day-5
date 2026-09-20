"""Tests for the Ollama generation adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from expense_rag.generation.base import GenerationProvider
from expense_rag.generation.providers.ollama_provider import (
    GENERATION_TEMPERATURE,
    OllamaGenerationProvider,
)
from expense_rag.generation.service import GenerationContractError
from expense_rag.models import GenerationDecision


def test_ollama_provider_sends_chat_schema_and_zero_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    decision = GenerationDecision(
        supported=True,
        answer="Employees may claim up to $65 per day.",
        cited_chunk_id="expense-policy:v2.0:section-1",
    )

    class FakeClient:
        def __init__(self, host: str) -> None:
            captured["host"] = host

        def chat(self, **kwargs: object) -> dict[str, object]:
            captured["kwargs"] = kwargs
            return {"message": {"content": decision.model_dump_json()}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    provider = OllamaGenerationProvider(
        model_name="mistral:7b",
        host="http://host.docker.internal:11434",
    )
    schema = GenerationDecision.model_json_schema()

    result = provider.generate(
        system_instruction="system",
        user_prompt="user",
        response_schema=schema,
    )

    assert isinstance(provider, GenerationProvider)
    assert result == decision
    assert captured["host"] == "http://host.docker.internal:11434"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["model"] == "mistral:7b"
    assert kwargs["format"] == schema
    assert kwargs["think"] is False
    assert kwargs["options"] == {"temperature": GENERATION_TEMPERATURE}
    assert kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_ollama_provider_rejects_unstructured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClient:
        def __init__(self, host: str) -> None:
            return None

        def chat(self, **kwargs: object) -> dict[str, object]:
            return {"message": {"content": "The meal limit is $65."}}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeClient))
    provider = OllamaGenerationProvider(
        model_name="qwen3:8b",
        host="http://host.docker.internal:11434",
    )

    with pytest.raises(GenerationContractError, match="malformed output"):
        provider.generate(
            system_instruction="system",
            user_prompt="user",
            response_schema=GenerationDecision.model_json_schema(),
        )


def test_ollama_provider_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unsupported comparison model"):
        OllamaGenerationProvider(
            model_name="llama3:8b",
            host="http://host.docker.internal:11434",
        )
