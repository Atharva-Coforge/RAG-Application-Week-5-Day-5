"""Ollama adapter for the grounded GenerationProvider contract."""

from __future__ import annotations

import json
from typing import Any

from expense_rag.generation.service import GenerationContractError
from expense_rag.models import GenerationDecision

CANDIDATE_OLLAMA_MODELS = ("mistral:7b", "qwen3:8b")
GENERATION_TEMPERATURE = 0.0


class OllamaGenerationProvider:
    """Local Ollama chat adapter with schema-constrained JSON output."""

    def __init__(
        self,
        *,
        model_name: str,
        host: str,
        temperature: float = GENERATION_TEMPERATURE,
    ) -> None:
        if model_name not in CANDIDATE_OLLAMA_MODELS:
            supported = ", ".join(CANDIDATE_OLLAMA_MODELS)
            raise ValueError(
                f"unsupported comparison model {model_name!r}; choose one of {supported}"
            )
        if not host.strip():
            raise ValueError("Ollama host must not be empty")
        try:
            from ollama import Client
        except ImportError as error:
            raise RuntimeError(
                "generation comparison dependencies are not installed; "
                "run 'uv sync --extra generation --group dev'"
            ) from error

        self._model_name = model_name
        self._temperature = temperature
        self._client = Client(host=host)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> GenerationDecision:
        try:
            response = self._client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt},
                ],
                think=False,
                format=response_schema,
                options={"temperature": self._temperature},
            )
        except GenerationContractError:
            raise
        except Exception as error:
            raise GenerationContractError(
                "generation provider returned malformed output"
            ) from error

        return _decision_from_content(_message_content(response))


def _message_content(response: object) -> str:
    message: Any
    if isinstance(response, dict):
        message = response.get("message")
    else:
        message = getattr(response, "message", None)

    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise GenerationContractError(
            "generation provider returned malformed output"
        )
    return content


def _decision_from_content(content: str) -> GenerationDecision:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise GenerationContractError(
            "generation provider returned malformed output"
        ) from error
    try:
        return GenerationDecision.model_validate(payload)
    except Exception as error:
        raise GenerationContractError(
            "generation provider returned malformed output"
        ) from error
