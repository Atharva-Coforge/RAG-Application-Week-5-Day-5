"""Tests for grounded answer generation."""

from __future__ import annotations

from typing import get_type_hints

import pytest
from pydantic import ValidationError

from expense_rag.generation.base import GenerationProvider
from expense_rag.generation.prompts import SYSTEM_INSTRUCTION, build_user_prompt
from expense_rag.generation.service import GenerationContractError, GenerationService
from expense_rag.models import (
    REFUSAL_ANSWER,
    GenerationDecision,
    PolicyChunk,
    SearchResult,
)


class FakeGenerationProvider:
    """Return a preconfigured structured decision or raise."""

    def __init__(self, result: GenerationDecision | dict[str, object] | Exception) -> None:
        self._result = result
        self.system_instruction: str | None = None
        self.user_prompt: str | None = None
        self.response_schema: dict[str, object] | None = None
        self.call_count = 0

    def generate(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        response_schema: dict[str, object],
    ) -> GenerationDecision:
        self.call_count += 1
        self.system_instruction = system_instruction
        self.user_prompt = user_prompt
        self.response_schema = response_schema
        if isinstance(self._result, Exception):
            raise self._result
        if isinstance(self._result, dict):
            return self._result  # type: ignore[return-value]
        return self._result


def test_generation_provider_protocol_is_runtime_checkable() -> None:
    provider = FakeGenerationProvider(
        GenerationDecision(
            supported=True,
            answer="Employees may claim up to $65 per day.",
            cited_chunk_id="expense-policy:v2.0:section-1",
        )
    )

    assert isinstance(provider, GenerationProvider)


def test_generation_service_depends_only_on_protocol() -> None:
    hints = get_type_hints(GenerationService.__init__)

    assert hints["generation_provider"] is GenerationProvider


def test_system_instruction_uses_assignment_text_and_structured_fields() -> None:
    assert "Answer the question using only the policy excerpts below." in SYSTEM_INSTRUCTION
    assert "The provided policy does not answer this question." in SYSTEM_INSTRUCTION
    assert "even when the user uses different words" in SYSTEM_INSTRUCTION
    assert "treat everyday synonyms as the same idea" in SYSTEM_INSTRUCTION
    assert "supported: true whenever you can cite a relevant excerpt" in SYSTEM_INSTRUCTION
    assert "cited_chunk_id: the chunk_id of that relevant excerpt only" in SYSTEM_INSTRUCTION


def test_user_prompt_includes_only_question_and_labeled_excerpts() -> None:
    prompt = build_user_prompt(
        "How much can I spend on food each day?",
        (_result("1", "Meals", "Employees may claim up to $65 per day."),),
    )

    assert "How much can I spend on food each day?" in prompt
    assert "chunk_id: expense-policy:v2.0:section-1" in prompt
    assert "section: 1. Meals" in prompt
    assert "Employees may claim up to $65 per day." in prompt
    assert "distance" not in prompt
    assert "0.08" not in prompt


def test_supported_decision_builds_cited_rag_response() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    hotels = _result("2", "Hotels", "Hotels are reimbursable up to $225 per night.")
    provider = FakeGenerationProvider(
        GenerationDecision(
            supported=True,
            answer="Employees may claim up to $65 per day for meals while traveling overnight.",
            cited_chunk_id=meals.chunk.chunk_id,
        )
    )
    service = GenerationService(generation_provider=provider)

    response = service.generate(
        "How much can I spend on food each day?",
        (meals, hotels),
    )

    assert provider.call_count == 1
    assert provider.system_instruction == SYSTEM_INSTRUCTION
    assert provider.response_schema == GenerationDecision.model_json_schema()
    assert meals.chunk.chunk_id in (provider.user_prompt or "")
    assert response.answer == (
        "Employees may claim up to $65 per day for meals while traveling overnight."
    )
    assert response.citation is not None
    assert response.citation.document == "Employee Expense Policy"
    assert response.citation.version == "2.0"
    assert response.citation.section == "1. Meals"
    assert response.retrieved_chunks == (meals, hotels)


def test_unsupported_decision_becomes_canonical_refusal() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    provider = FakeGenerationProvider(
        GenerationDecision(
            supported=False,
            answer="Gym memberships are not mentioned, so I will guess no.",
            cited_chunk_id=None,
        )
    )
    service = GenerationService(generation_provider=provider)

    response = service.generate(
        "Does the company reimburse gym memberships?",
        (meals,),
    )

    assert response.answer == REFUSAL_ANSWER
    assert response.citation is None
    assert response.retrieved_chunks == (meals,)


def test_empty_retrieval_refuses_without_calling_provider() -> None:
    provider = FakeGenerationProvider(
        GenerationDecision(
            supported=True,
            answer="should not be used",
            cited_chunk_id="expense-policy:v2.0:section-1",
        )
    )
    service = GenerationService(generation_provider=provider)

    response = service.generate("Does the company reimburse gym memberships?", ())

    assert provider.call_count == 0
    assert response.answer == REFUSAL_ANSWER
    assert response.citation is None
    assert response.retrieved_chunks == ()


def test_supported_decision_with_unknown_chunk_id_fails_closed() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    service = GenerationService(
        generation_provider=FakeGenerationProvider(
            GenerationDecision(
                supported=True,
                answer="Employees may claim up to $65 per day.",
                cited_chunk_id="expense-policy:v2.0:section-9",
            )
        )
    )

    with pytest.raises(GenerationContractError, match="retrieved set"):
        service.generate("How much can I spend on food each day?", (meals,))


def test_supported_decision_without_citation_fails_closed() -> None:
    with pytest.raises(ValidationError, match="cite a chunk_id"):
        GenerationDecision(
            supported=True,
            answer="Employees may claim up to $65 per day.",
            cited_chunk_id=None,
        )


def test_unsupported_decision_with_citation_fails_closed() -> None:
    with pytest.raises(ValidationError, match="must not cite a chunk_id"):
        GenerationDecision(
            supported=False,
            answer=REFUSAL_ANSWER,
            cited_chunk_id="expense-policy:v2.0:section-1",
        )


def test_empty_question_fails_closed() -> None:
    service = GenerationService(
        generation_provider=FakeGenerationProvider(
            GenerationDecision(
                supported=False,
                answer=REFUSAL_ANSWER,
                cited_chunk_id=None,
            )
        )
    )

    with pytest.raises(GenerationContractError, match="question must not be empty"):
        service.generate(
            "   ",
            (_result("1", "Meals", "Employees may claim up to $65 per day."),),
        )


def test_malformed_provider_output_fails_closed() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    service = GenerationService(
        generation_provider=FakeGenerationProvider(ValueError("not json"))
    )

    with pytest.raises(GenerationContractError, match="malformed output"):
        service.generate("How much can I spend on food each day?", (meals,))


def test_unstructured_provider_output_fails_closed() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    service = GenerationService(
        generation_provider=FakeGenerationProvider("The meal limit is $65.")  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationContractError, match="structured output"):
        service.generate("How much can I spend on food each day?", (meals,))


def test_structured_dict_output_is_accepted() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    service = GenerationService(
        generation_provider=FakeGenerationProvider(
            {
                "supported": True,
                "answer": "Employees may claim up to $65 per day.",
                "cited_chunk_id": meals.chunk.chunk_id,
            }
        )
    )

    response = service.generate("How much can I spend on food each day?", (meals,))

    assert response.citation is not None
    assert response.citation.section == "1. Meals"


def test_invalid_structured_dict_fails_closed() -> None:
    meals = _result("1", "Meals", "Employees may claim up to $65 per day.")
    service = GenerationService(
        generation_provider=FakeGenerationProvider(
            {"supported": True, "answer": "Employees may claim up to $65 per day."}
        )
    )

    with pytest.raises(GenerationContractError, match="malformed output"):
        service.generate("How much can I spend on food each day?", (meals,))


def _result(section: str, title: str, text: str) -> SearchResult:
    chunk = PolicyChunk(
        chunk_id=f"expense-policy:v2.0:section-{section}",
        document="Employee Expense Policy",
        version="2.0",
        section=section,
        section_title=title,
        text=text,
        embedding=(1.0, 0.0),
    )
    return SearchResult(chunk=chunk, distance=0.08 if section == "1" else 0.2)
