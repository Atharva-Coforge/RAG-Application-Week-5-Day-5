"""LLM comparison using frozen MiniLM retrieval and local Ollama models."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from expense_rag.embeddings.base import EmbeddingProvider
from expense_rag.embeddings.provider import embed_query, embed_sections
from expense_rag.env import get_embedding_model, get_ollama_host, project_root
from expense_rag.evaluation.generation_metrics import (
    GenerationCaseEvaluation,
    GenerationQualityMetrics,
    evaluate_generation_case,
    summarize_generation_quality,
)
from expense_rag.evaluation.gold_loader import load_gold_cases
from expense_rag.generation.base import GenerationProvider
from expense_rag.generation.providers.ollama_provider import (
    CANDIDATE_OLLAMA_MODELS,
    GENERATION_TEMPERATURE,
    OllamaGenerationProvider,
)
from expense_rag.generation.service import GenerationContractError, GenerationService
from expense_rag.ingestion.parser import load_policy
from expense_rag.models import GoldCase, PolicyChunk, RagResponse, SearchResult
from expense_rag.retrieval.cosine import cosine_search

DEFAULT_LATENCY_ROUNDS = 3


class FrozenRetrievedChunk(BaseModel):
    """Search evidence that keeps the full chunk for later generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: PolicyChunk
    distance: float = Field(ge=0.0, le=2.0)

    def to_search_result(self) -> SearchResult:
        return SearchResult(chunk=self.chunk, distance=self.distance)


class FrozenGenerationCase(BaseModel):
    """One gold question with frozen top-three evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    supported: bool
    required_answer: str
    expected_section: str | None
    retrieved_chunks: tuple[FrozenRetrievedChunk, ...]

    def gold_case(self) -> GoldCase:
        return GoldCase(
            question=self.question,
            supported=self.supported,
            required_answer=self.required_answer,
            expected_section=self.expected_section,
        )

    def search_results(self) -> tuple[SearchResult, ...]:
        return tuple(item.to_search_result() for item in self.retrieved_chunks)


class GenerationTimingMetrics(BaseModel):
    """Latency from the measured generation rounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    quality_pass_total_ms: float = Field(ge=0.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_rounds: int = Field(ge=1)


class GenerationBenchmarkResult(BaseModel):
    """Quality and timing evidence for one Ollama model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    quality: GenerationQualityMetrics
    timing: GenerationTimingMetrics


class GenerationComparisonReport(BaseModel):
    """Comparison evidence without selecting a winner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str
    ollama_host: str
    temperature: float
    latency_rounds: int
    environment: dict[str, str]
    frozen_cases: tuple[FrozenGenerationCase, ...]
    results: tuple[GenerationBenchmarkResult, ...]
    selection_status: str


def freeze_generation_cases(
    cases: Sequence[GoldCase],
    chunks: Sequence[PolicyChunk],
    embedding_provider: EmbeddingProvider,
) -> tuple[FrozenGenerationCase, ...]:
    """Retrieve top-three evidence once with exact cosine search."""
    frozen: list[FrozenGenerationCase] = []
    for case in cases:
        results = cosine_search(
            embed_query(case.question, embedding_provider),
            chunks,
        )
        frozen.append(
            FrozenGenerationCase(
                question=case.question,
                supported=case.supported,
                required_answer=case.required_answer,
                expected_section=case.expected_section,
                retrieved_chunks=tuple(
                    FrozenRetrievedChunk(
                        chunk=result.chunk,
                        distance=result.distance,
                    )
                    for result in results
                ),
            )
        )
    return tuple(frozen)


class GenerationComparisonRunner:
    """Benchmark Ollama candidates against identical frozen contexts."""

    def __init__(
        self,
        *,
        project_root_path: Path,
        ollama_host: str,
        latency_rounds: int = DEFAULT_LATENCY_ROUNDS,
    ) -> None:
        if latency_rounds <= 0:
            raise ValueError("latency rounds must be positive")
        self._project_root = project_root_path
        self._ollama_host = ollama_host
        self._latency_rounds = latency_rounds

    def run(
        self,
        models: Sequence[str] = CANDIDATE_OLLAMA_MODELS,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        generation_factory: Callable[[str], GenerationProvider] | None = None,
    ) -> GenerationComparisonReport:
        """Freeze retrieval once, then score each generation model."""
        invalid = set(models) - set(CANDIDATE_OLLAMA_MODELS)
        if invalid:
            raise ValueError(f"unsupported generation models: {sorted(invalid)}")
        if not models:
            raise ValueError("at least one generation model is required")

        cases = load_gold_cases(self._project_root / "data" / "gold" / "gold-data.md")
        sections = load_policy(self._project_root / "data" / "policy.md")
        embedder = embedding_provider or _load_selected_embedder()
        chunks = embed_sections(sections, embedder)
        frozen_cases = freeze_generation_cases(cases, chunks, embedder)
        factory = generation_factory or _default_generation_factory(self._ollama_host)

        results = tuple(
            self._benchmark_model(model_name, frozen_cases, factory)
            for model_name in models
        )
        return GenerationComparisonReport(
            embedding_model=embedder.model_name,
            ollama_host=self._ollama_host,
            temperature=GENERATION_TEMPERATURE,
            latency_rounds=self._latency_rounds,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": "local Ollama",
            },
            frozen_cases=frozen_cases,
            results=results,
            selection_status="Pending user decision",
        )

    def _benchmark_model(
        self,
        model_name: str,
        frozen_cases: Sequence[FrozenGenerationCase],
        factory: Callable[[str], GenerationProvider],
    ) -> GenerationBenchmarkResult:
        provider = factory(model_name)
        service = GenerationService(generation_provider=provider)
        quality_started = perf_counter()
        quality_evaluations = tuple(
            _run_quality_case(service, frozen_case) for frozen_case in frozen_cases
        )
        quality_pass_total_ms = _elapsed_ms(quality_started)
        quality = summarize_generation_quality(quality_evaluations)

        latency_samples: list[float] = []
        for _ in range(self._latency_rounds):
            for frozen_case in frozen_cases:
                started = perf_counter()
                try:
                    service.generate(
                        frozen_case.question,
                        frozen_case.search_results(),
                    )
                except GenerationContractError:
                    pass
                latency_samples.append(_elapsed_ms(started))

        return GenerationBenchmarkResult(
            model_name=model_name,
            quality=quality,
            timing=GenerationTimingMetrics(
                quality_pass_total_ms=quality_pass_total_ms,
                latency_p50_ms=_percentile(latency_samples, 0.50),
                latency_p95_ms=_percentile(latency_samples, 0.95),
                latency_rounds=self._latency_rounds,
            ),
        )


def _run_quality_case(
    service: GenerationService,
    frozen_case: FrozenGenerationCase,
) -> GenerationCaseEvaluation:
    started = perf_counter()
    response: RagResponse | None = None
    error: str | None = None
    try:
        response = service.generate(
            frozen_case.question,
            frozen_case.search_results(),
        )
    except GenerationContractError as exc:
        error = str(exc)
    return evaluate_generation_case(
        frozen_case.gold_case(),
        response,
        error=error,
        latency_ms=_elapsed_ms(started),
    )


def _load_selected_embedder() -> EmbeddingProvider:
    from expense_rag.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingProvider,
    )

    return SentenceTransformerEmbeddingProvider(get_embedding_model())


def _default_generation_factory(
    host: str,
) -> Callable[[str], GenerationProvider]:
    def factory(model_name: str) -> GenerationProvider:
        return OllamaGenerationProvider(model_name=model_name, host=host)

    return factory


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument(
        "--models",
        default=",".join(CANDIDATE_OLLAMA_MODELS),
        help="Comma-separated Ollama model tags.",
    )
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/evaluation/generation-comparison.json"),
    )
    parser.add_argument(
        "--latency-rounds",
        type=int,
        default=DEFAULT_LATENCY_ROUNDS,
    )
    return parser.parse_args()


def main() -> None:
    """Run the local Ollama comparison and write the raw report."""
    args = _parse_args()
    models = tuple(
        model.strip() for model in args.models.split(",") if model.strip()
    )
    host = args.ollama_host or get_ollama_host()
    runner = GenerationComparisonRunner(
        project_root_path=args.project_root.resolve(),
        ollama_host=host,
        latency_rounds=args.latency_rounds,
    )
    report = runner.run(models)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    print(f"Raw report: {output_path}")


if __name__ == "__main__":
    main()
