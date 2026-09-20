"""Acceptance evaluation for the six gold questions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from math import isfinite
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from expense_rag.config import Settings
from expense_rag.embeddings.base import EmbeddingProvider
from expense_rag.evaluation.generation_metrics import evaluate_generation_case
from expense_rag.evaluation.gold_loader import EXPECTED_CASE_COUNT
from expense_rag.generation.base import GenerationProvider
from expense_rag.generation.service import GenerationService
from expense_rag.ingestion.parser import EXPECTED_SECTION_COUNT
from expense_rag.models import GoldCase, PolicyChunk, RagResponse
from expense_rag.retrieval.cosine import MAX_RETRIEVAL_RESULTS
from expense_rag.retrieval.service import RetrievalService
from expense_rag.vector_stores.base import VectorStore

MIN_SUPPORTED_SECTION_HITS = 5


class EvaluationCaseResult(BaseModel):
    """One gold question's retrieval, answer, citation, and timing result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    supported: bool
    expected_section: str | None
    required_answer: str
    response: RagResponse
    retrieved_sections: tuple[str, ...]
    distances: tuple[float, ...]
    expected_section_retrieved: bool | None
    citation_correct: bool | None
    refusal_correct: bool | None
    retrieval_shape_valid: bool
    complete_records: bool
    latency_ms: float = Field(ge=0.0)


class EvaluationReport(BaseModel):
    """Acceptance outcome for one configured application run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime
    embedding_model: str
    embedding_dimension: int
    vector_backend: str
    generation_provider: str
    generation_model: str
    top_k: int
    stored_chunk_count: int
    six_chunks_stored: bool
    retrieval_shape_valid: bool
    complete_records: bool
    supported_hit_at_3_count: int = Field(ge=0)
    supported_hit_at_3_required: int = Field(ge=1)
    supported_hit_at_3_passes: bool
    all_supported_grounded_and_cited: bool
    gym_refusal_correct: bool
    accepted: bool
    case_results: tuple[EvaluationCaseResult, ...]


class EvaluationRunner:
    """Run the six gold cases through one assembled application."""

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        generation_provider: GenerationProvider,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._retrieval = RetrievalService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )
        self._generation = GenerationService(
            generation_provider=generation_provider,
        )

    def run(self, cases: Sequence[GoldCase]) -> EvaluationReport:
        """Score retrieval, grounding, citation, and refusal for every case."""
        frozen_cases = tuple(cases)
        if len(frozen_cases) != EXPECTED_CASE_COUNT:
            raise ValueError(
                f"evaluation requires exactly {EXPECTED_CASE_COUNT} gold cases"
            )
        stored_chunk_count = self._vector_store.count()
        if stored_chunk_count == 0:
            raise FileNotFoundError("policy has not been ingested")

        case_results = tuple(
            self._evaluate_case(case) for case in frozen_cases
        )
        supported = [item for item in case_results if item.supported]
        unsupported = [item for item in case_results if not item.supported]
        if len(supported) != EXPECTED_CASE_COUNT - 1 or len(unsupported) != 1:
            raise ValueError("gold set must contain five supported cases and one refusal")

        six_chunks_stored = stored_chunk_count == EXPECTED_SECTION_COUNT
        retrieval_shape_valid = all(item.retrieval_shape_valid for item in case_results)
        complete_records = all(item.complete_records for item in case_results)
        supported_hit_at_3_count = sum(
            1 for item in supported if item.expected_section_retrieved
        )
        supported_hit_at_3_passes = (
            supported_hit_at_3_count >= MIN_SUPPORTED_SECTION_HITS
        )
        all_supported_grounded_and_cited = all(
            item.citation_correct for item in supported
        )
        gym_refusal_correct = bool(unsupported[0].refusal_correct)
        accepted = (
            six_chunks_stored
            and retrieval_shape_valid
            and complete_records
            and supported_hit_at_3_passes
            and all_supported_grounded_and_cited
            and gym_refusal_correct
        )

        return EvaluationReport(
            generated_at=datetime.now(UTC),
            embedding_model=self._settings.embedding_model,
            embedding_dimension=self._embedding_provider.dimension,
            vector_backend=self._settings.vector_backend.value,
            generation_provider=self._settings.generation_provider,
            generation_model=self._settings.generation_model,
            top_k=self._settings.top_k,
            stored_chunk_count=stored_chunk_count,
            six_chunks_stored=six_chunks_stored,
            retrieval_shape_valid=retrieval_shape_valid,
            complete_records=complete_records,
            supported_hit_at_3_count=supported_hit_at_3_count,
            supported_hit_at_3_required=MIN_SUPPORTED_SECTION_HITS,
            supported_hit_at_3_passes=supported_hit_at_3_passes,
            all_supported_grounded_and_cited=all_supported_grounded_and_cited,
            gym_refusal_correct=gym_refusal_correct,
            accepted=accepted,
            case_results=case_results,
        )

    def _evaluate_case(self, case: GoldCase) -> EvaluationCaseResult:
        started = perf_counter()
        retrieved = self._retrieval.retrieve(
            case.question,
            top_k=self._settings.top_k,
        )
        response = self._generation.generate(case.question, retrieved)
        latency_ms = (perf_counter() - started) * 1000.0
        scored = evaluate_generation_case(case, response, latency_ms=latency_ms)
        distances = tuple(result.distance for result in retrieved)
        retrieval_shape_valid = (
            1 <= len(retrieved) <= MAX_RETRIEVAL_RESULTS
            and distances == tuple(sorted(distances))
            and all(isfinite(distance) for distance in distances)
        )
        expected_section_retrieved: bool | None = None
        if case.supported:
            expected_section_retrieved = any(
                result.chunk.section == case.expected_section for result in retrieved
            )

        return EvaluationCaseResult(
            question=case.question,
            supported=case.supported,
            expected_section=case.expected_section,
            required_answer=case.required_answer,
            response=response,
            retrieved_sections=tuple(result.section for result in retrieved),
            distances=distances,
            expected_section_retrieved=expected_section_retrieved,
            citation_correct=scored.citation_correct,
            refusal_correct=scored.refusal_correct,
            retrieval_shape_valid=retrieval_shape_valid,
            complete_records=all(
                _complete_record(result.chunk) for result in retrieved
            ),
            latency_ms=latency_ms,
        )


def _complete_record(chunk: PolicyChunk) -> bool:
    return bool(
        chunk.chunk_id
        and chunk.document
        and chunk.version
        and chunk.section
        and chunk.section_title
        and chunk.text
        and chunk.embedding
    )
