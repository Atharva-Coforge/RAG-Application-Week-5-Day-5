"""Reproducible embedding-model comparison runner."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from time import perf_counter

from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field

from expense_rag.embeddings.provider import (
    embed_query,
    embed_sections,
    format_section_for_embedding,
)
from expense_rag.embeddings.sentence_transformer import (
    MODEL_INPUT_FORMATS,
    SELECTED_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from expense_rag.evaluation.gold_loader import load_gold_cases
from expense_rag.evaluation.metrics import (
    RetrievalQualityMetrics,
    evaluate_retrieval_quality,
)
from expense_rag.ingestion.parser import load_policy
from expense_rag.models import GoldCase, PolicySection
from expense_rag.retrieval.cosine import cosine_search

DEFAULT_WARMUP_ROUNDS = 2
DEFAULT_MEASURED_ROUNDS = 20


class TimingMetrics(BaseModel):
    """Measured model and exact-retrieval timings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_load_ms: float = Field(ge=0.0)
    document_embedding_p50_ms: float = Field(ge=0.0)
    document_embedding_p95_ms: float = Field(ge=0.0)
    query_embedding_p50_ms: float = Field(ge=0.0)
    query_embedding_p95_ms: float = Field(ge=0.0)
    retrieval_p50_ms: float = Field(ge=0.0)
    retrieval_p95_ms: float = Field(ge=0.0)
    query_throughput_per_second: float = Field(ge=0.0)


class ResourceMetrics(BaseModel):
    """Disk and process-memory observations from an isolated model process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    peak_process_ram_mb: float = Field(ge=0.0)
    peak_ram_increase_mb: float = Field(ge=0.0)
    downloaded_model_size_mb: float = Field(ge=0.0)


class EmbeddingBenchmarkResult(BaseModel):
    """Quality, timing, and resource evidence for one candidate model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    dimension: int = Field(gt=0)
    quality: RetrievalQualityMetrics
    timing: TimingMetrics
    resources: ResourceMetrics


class EmbeddingComparisonReport(BaseModel):
    """Complete comparison evidence without making the final user decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    warmup_rounds: int = Field(gt=0)
    measured_rounds: int = Field(gt=0)
    canonical_document_format: str
    model_specific_formatting: dict[str, dict[str, str]]
    environment: dict[str, str]
    accuracy_first_order: tuple[str, ...]
    results: tuple[EmbeddingBenchmarkResult, ...]
    selected_model: str
    selection_status: str


class EmbeddingComparisonRunner:
    """Benchmark one model inside an isolated process."""

    def __init__(
        self,
        *,
        warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
        measured_rounds: int = DEFAULT_MEASURED_ROUNDS,
    ) -> None:
        if warmup_rounds <= 0 or measured_rounds <= 0:
            raise ValueError("benchmark round counts must be positive")
        self._warmup_rounds = warmup_rounds
        self._measured_rounds = measured_rounds

    def benchmark_model(
        self,
        model_name: str,
        sections: Sequence[PolicySection],
        cases: Sequence[GoldCase],
    ) -> EmbeddingBenchmarkResult:
        """Run one candidate after all inputs have been parsed and validated."""
        baseline_peak_mb = _peak_process_ram_mb()
        load_started = perf_counter()
        provider = SentenceTransformerEmbeddingProvider(model_name)
        model_load_ms = _elapsed_ms(load_started)

        document_texts = tuple(
            format_section_for_embedding(section) for section in sections
        )
        questions = tuple(case.question for case in cases)

        for _ in range(self._warmup_rounds):
            provider.embed_documents(document_texts)
            for question in questions:
                provider.embed_query(question)

        document_times: list[float] = []
        query_times: list[float] = []
        for _ in range(self._measured_rounds):
            started = perf_counter()
            provider.embed_documents(document_texts)
            document_times.append(_elapsed_ms(started))

            for question in questions:
                started = perf_counter()
                provider.embed_query(question)
                query_times.append(_elapsed_ms(started))

        chunks = embed_sections(sections, provider)
        query_embeddings = tuple(
            embed_query(case.question, provider) for case in cases
        )

        retrieval_times: list[float] = []
        for _ in range(self._measured_rounds):
            for query_embedding in query_embeddings:
                started = perf_counter()
                cosine_search(query_embedding, chunks)
                retrieval_times.append(_elapsed_ms(started))

        quality = evaluate_retrieval_quality(
            cases,
            query_embeddings,
            chunks,
        )
        total_query_seconds = sum(query_times) / 1000.0
        throughput = (
            len(query_times) / total_query_seconds
            if total_query_seconds > 0.0
            else 0.0
        )
        peak_ram_mb = _peak_process_ram_mb()

        return EmbeddingBenchmarkResult(
            model_name=model_name,
            dimension=provider.dimension,
            quality=quality,
            timing=TimingMetrics(
                model_load_ms=model_load_ms,
                document_embedding_p50_ms=median(document_times),
                document_embedding_p95_ms=_percentile(document_times, 0.95),
                query_embedding_p50_ms=median(query_times),
                query_embedding_p95_ms=_percentile(query_times, 0.95),
                retrieval_p50_ms=median(retrieval_times),
                retrieval_p95_ms=_percentile(retrieval_times, 0.95),
                query_throughput_per_second=throughput,
            ),
            resources=ResourceMetrics(
                peak_process_ram_mb=peak_ram_mb,
                peak_ram_increase_mb=max(0.0, peak_ram_mb - baseline_peak_mb),
                downloaded_model_size_mb=_downloaded_model_size_mb(model_name),
            ),
        )


def run_comparison(
    *,
    project_root: Path,
    output_path: Path,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    measured_rounds: int = DEFAULT_MEASURED_ROUNDS,
) -> EmbeddingComparisonReport:
    """Run every model in a clean subprocess and combine their evidence."""
    results: list[EmbeddingBenchmarkResult] = []
    with tempfile.TemporaryDirectory(prefix="embedding-comparison-") as temp_dir:
        for index, model_name in enumerate(MODEL_INPUT_FORMATS):
            result_path = Path(temp_dir) / f"model-{index}.json"
            command = [
                sys.executable,
                "-m",
                "expense_rag.evaluation.embedding_runner",
                "--single-model",
                model_name,
                "--project-root",
                str(project_root),
                "--output",
                str(result_path),
                "--warmup-rounds",
                str(warmup_rounds),
                "--measured-rounds",
                str(measured_rounds),
            ]
            subprocess.run(command, check=True)
            results.append(
                EmbeddingBenchmarkResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
            )

    report = EmbeddingComparisonReport(
        warmup_rounds=warmup_rounds,
        measured_rounds=measured_rounds,
        canonical_document_format="section number\\nsection title\\nsection body",
        model_specific_formatting={
            name: {
                "query_prefix": input_format.query_prefix,
                "document_prefix": input_format.document_prefix,
            }
            for name, input_format in MODEL_INPUT_FORMATS.items()
        },
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
            "device": "CPU",
        },
        accuracy_first_order=tuple(
            result.model_name
            for result in sorted(results, key=_accuracy_first_key, reverse=True)
        ),
        results=tuple(results),
        selected_model=SELECTED_EMBEDDING_MODEL,
        selection_status="Selected by user",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return report


def _accuracy_first_key(
    result: EmbeddingBenchmarkResult,
) -> tuple[bool, float, float, float, float, float]:
    quality = result.quality
    return (
        quality.hit_at_3 == 1.0,
        quality.hit_at_1,
        quality.mean_reciprocal_rank,
        quality.mean_distance_margin,
        -result.timing.query_embedding_p95_ms,
        -result.resources.peak_ram_increase_mb,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile without values")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + (
        ordered[upper_index] - ordered[lower_index]
    ) * fraction


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000.0


def _peak_process_ram_mb() -> float:
    # Linux reports ru_maxrss in KiB. The project and comparison run on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _downloaded_model_size_mb(model_name: str) -> float:
    snapshot_path = Path(
        snapshot_download(
            repo_id=model_name,
            local_files_only=True,
        )
    )
    size_bytes = sum(
        path.stat().st_size for path in snapshot_path.rglob("*") if path.is_file()
    )
    return size_bytes / (1024.0 * 1024.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/evaluation/embedding-comparison.json"),
    )
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument(
        "--measured-rounds",
        type=int,
        default=DEFAULT_MEASURED_ROUNDS,
    )
    parser.add_argument("--single-model", choices=tuple(MODEL_INPUT_FORMATS))
    return parser.parse_args()


def main() -> None:
    """Run one isolated model or the complete three-model comparison."""
    args = _parse_args()
    project_root = args.project_root.resolve()
    output_path = args.output.resolve()

    if args.single_model is not None:
        runner = EmbeddingComparisonRunner(
            warmup_rounds=args.warmup_rounds,
            measured_rounds=args.measured_rounds,
        )
        result = runner.benchmark_model(
            args.single_model,
            load_policy(project_root / "data" / "policy.md"),
            load_gold_cases(project_root / "data" / "gold" / "gold-data.md"),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return

    report = run_comparison(
        project_root=project_root,
        output_path=output_path,
        warmup_rounds=args.warmup_rounds,
        measured_rounds=args.measured_rounds,
    )
    print(json.dumps(_summary(report), indent=2))
    print(f"Raw report: {output_path}")


def _summary(report: EmbeddingComparisonReport) -> dict[str, object]:
    return {
        "selection_status": report.selection_status,
        "selected_model": report.selected_model,
        "accuracy_first_order": report.accuracy_first_order,
        "models": [
            {
                "model": result.model_name,
                "hit_at_1": result.quality.hit_at_1,
                "hit_at_3": result.quality.hit_at_3,
                "mrr": result.quality.mean_reciprocal_rank,
                "mean_margin": result.quality.mean_distance_margin,
                "query_p95_ms": result.timing.query_embedding_p95_ms,
                "peak_ram_mb": result.resources.peak_process_ram_mb,
            }
            for result in report.results
        ],
    }


if __name__ == "__main__":
    main()
