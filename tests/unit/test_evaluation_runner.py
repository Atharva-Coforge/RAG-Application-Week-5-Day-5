"""Acceptance scoring for the six gold cases using protocol fakes."""

from __future__ import annotations

import pytest

from expense_rag.config import Settings, VectorBackend
from expense_rag.env import project_root
from expense_rag.evaluation.gold_loader import load_gold_cases
from expense_rag.evaluation.runner import EvaluationRunner
from expense_rag.ingestion.parser import EXPECTED_SECTION_COUNT
from expense_rag.ingestion.service import IngestionService
from expense_rag.models import REFUSAL_ANSWER, GenerationDecision
from tests.fakes import (
    InMemoryVectorStore,
    MappedGenerationProvider,
    gold_aware_embedding_provider,
    gold_aware_generation_provider,
)


def _settings() -> Settings:
    root = project_root()
    return Settings(
        project_root=root,
        policy_path=root / "data" / "policy.md",
        gold_path=root / "data" / "gold" / "gold-data.md",
        artifacts_dir=root / "data" / "artifacts",
        chroma_directory=root / "data" / "artifacts" / "chroma",
        faiss_directory=root / "data" / "artifacts" / "faiss",
        collection_name="expense-policy",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_backend=VectorBackend.CHROMA,
        generation_provider="ollama",
        generation_model="qwen3:8b",
        ollama_host="http://127.0.0.1:11434",
        top_k=3,
    )


def _ingested_store() -> InMemoryVectorStore:
    embedder = gold_aware_embedding_provider()
    store = InMemoryVectorStore(
        collection_name="expense-policy",
        dimension=EXPECTED_SECTION_COUNT,
    )
    IngestionService(embedding_provider=embedder, vector_store=store).ingest(
        project_root() / "data" / "policy.md"
    )
    return store


def test_evaluation_runner_accepts_grounded_gold_set() -> None:
    embedder = gold_aware_embedding_provider()
    store = _ingested_store()
    runner = EvaluationRunner(
        settings=_settings(),
        embedding_provider=embedder,
        vector_store=store,
        generation_provider=gold_aware_generation_provider(),
    )

    report = runner.run(load_gold_cases(project_root() / "data" / "gold" / "gold-data.md"))

    assert report.stored_chunk_count == EXPECTED_SECTION_COUNT
    assert report.six_chunks_stored is True
    assert report.retrieval_shape_valid is True
    assert report.complete_records is True
    assert report.supported_hit_at_3_count == 5
    assert report.supported_hit_at_3_passes is True
    assert report.all_supported_grounded_and_cited is True
    assert report.gym_refusal_correct is True
    assert report.accepted is True
    assert report.case_results[-1].response.answer == REFUSAL_ANSWER
    assert report.case_results[-1].response.citation is None
    assert all(len(item.distances) <= 3 for item in report.case_results)
    assert all(
        item.distances == tuple(sorted(item.distances)) for item in report.case_results
    )


def test_evaluation_runner_fails_when_gym_is_not_refused() -> None:
    cases = load_gold_cases(project_root() / "data" / "gold" / "gold-data.md")
    decisions = {
        case.question: GenerationDecision(
            supported=True,
            answer="Gym memberships are reimbursable.",
            cited_chunk_id="expense-policy:v2.0:section-1",
        )
        for case in cases
    }
    runner = EvaluationRunner(
        settings=_settings(),
        embedding_provider=gold_aware_embedding_provider(),
        vector_store=_ingested_store(),
        generation_provider=MappedGenerationProvider(decisions),
    )

    report = runner.run(cases)

    assert report.gym_refusal_correct is False
    assert report.accepted is False


def test_evaluation_runner_requires_ingested_policy() -> None:
    store = InMemoryVectorStore(
        collection_name="expense-policy",
        dimension=EXPECTED_SECTION_COUNT,
    )
    runner = EvaluationRunner(
        settings=_settings(),
        embedding_provider=gold_aware_embedding_provider(),
        vector_store=store,
        generation_provider=gold_aware_generation_provider(),
    )

    with pytest.raises(FileNotFoundError, match="not been ingested"):
        runner.run(load_gold_cases(project_root() / "data" / "gold" / "gold-data.md"))
