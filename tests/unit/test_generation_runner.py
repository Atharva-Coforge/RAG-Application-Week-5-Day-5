"""Tests for frozen-context generation comparison without calling Ollama."""

from __future__ import annotations

from expense_rag.embeddings.provider import embed_sections
from expense_rag.env import project_root
from expense_rag.evaluation.generation_runner import (
    GenerationComparisonRunner,
    freeze_generation_cases,
)
from expense_rag.evaluation.gold_loader import load_gold_cases
from expense_rag.generation.base import GenerationProvider
from expense_rag.ingestion.parser import EXPECTED_SECTION_COUNT, load_policy
from expense_rag.models import REFUSAL_ANSWER, GenerationDecision
from tests.fakes import FakeEmbeddingProvider
from tests.unit.test_generation import FakeGenerationProvider


def _basis_vectors(count: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if index == axis else 0.0 for index in range(count))
        for axis in range(count)
    )


def test_freeze_generation_cases_keeps_top_three_chunks() -> None:
    root = project_root()
    cases = load_gold_cases(root / "data" / "gold" / "gold-data.md")
    sections = load_policy(root / "data" / "policy.md")
    provider = FakeEmbeddingProvider(
        document_vectors=_basis_vectors(EXPECTED_SECTION_COUNT),
        query_vector=_basis_vectors(EXPECTED_SECTION_COUNT)[0],
        dimension=EXPECTED_SECTION_COUNT,
    )
    chunks = embed_sections(sections, provider)

    frozen = freeze_generation_cases(cases, chunks, provider)

    assert len(frozen) == 6
    assert all(len(item.retrieved_chunks) == 3 for item in frozen)
    assert frozen[0].question == cases[0].question
    assert frozen[0].required_answer == cases[0].required_answer


def test_comparison_runner_records_selected_qwen() -> None:
    def factory(_model_name: str) -> GenerationProvider:
        return FakeGenerationProvider(
            GenerationDecision(
                supported=False,
                answer=REFUSAL_ANSWER,
                cited_chunk_id=None,
            )
        )

    embedder = FakeEmbeddingProvider(
        document_vectors=_basis_vectors(EXPECTED_SECTION_COUNT),
        query_vector=_basis_vectors(EXPECTED_SECTION_COUNT)[0],
        dimension=EXPECTED_SECTION_COUNT,
        model_name="fake-embedding-model",
    )
    runner = GenerationComparisonRunner(
        project_root_path=project_root(),
        ollama_host="http://host.docker.internal:11434",
        latency_rounds=1,
    )

    report = runner.run(
        ("mistral:7b", "qwen3:8b"),
        embedding_provider=embedder,
        generation_factory=factory,
    )

    assert report.selection_status == "Selected by user"
    assert report.selected_model == "qwen3:8b"
    assert [result.model_name for result in report.results] == [
        "mistral:7b",
        "qwen3:8b",
    ]
    assert report.results[0].timing.latency_rounds == 1
    assert report.results[0].quality.gym_refusal_correct is True
    assert all(
        case.required_answer and case.model_answer == REFUSAL_ANSWER
        for result in report.results
        for case in result.quality.case_results
    )
