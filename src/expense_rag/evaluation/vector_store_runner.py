"""Persistent vector-store comparison using frozen MiniLM embeddings."""

from __future__ import annotations

import argparse
import platform
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from expense_rag.embeddings.provider import embed_query, embed_sections
from expense_rag.embeddings.sentence_transformer import (
    SELECTED_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from expense_rag.env import get_database_url
from expense_rag.evaluation.gold_loader import load_gold_cases
from expense_rag.ingestion.parser import load_policy
from expense_rag.models import GoldCase, PolicyChunk, SearchResult
from expense_rag.retrieval.cosine import cosine_search
from expense_rag.vector_stores.base import VectorStore
from expense_rag.vector_stores.factory import SELECTED_VECTOR_STORE

DEFAULT_REPLACEMENT_ROUNDS = 20
DEFAULT_QUERY_ROUNDS = 200
_BACKENDS = ("chroma", "faiss", "pgvector")


class StoreQueryResult(BaseModel):
    """One backend's result for one gold question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    expected_section: str | None
    top_sections: tuple[str, ...]
    distances: tuple[float, ...]
    expected_rank: int | None
    exact_reference_parity: bool


class StoreTimingMetrics(BaseModel):
    """Replacement and query timings for one backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    replacement_p50_ms: float = Field(ge=0.0)
    replacement_p95_ms: float = Field(ge=0.0)
    query_p50_ms: float = Field(ge=0.0)
    query_p95_ms: float = Field(ge=0.0)


class VectorStoreBenchmarkResult(BaseModel):
    """Correctness, persistence, and timing evidence for one backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    exact_reference_parity: bool
    reopen_count: int = Field(ge=0)
    artifact_size_mb: float | None = Field(default=None, ge=0.0)
    timing: StoreTimingMetrics
    query_results: tuple[StoreQueryResult, ...]


class VectorStoreComparisonReport(BaseModel):
    """Combined evidence without making the final backend decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding_model: str
    dimension: int
    replacement_rounds: int
    query_rounds: int
    environment: dict[str, str]
    results: tuple[VectorStoreBenchmarkResult, ...]
    selected_backend: str
    selection_status: str


class VectorStoreComparisonRunner:
    """Benchmark persistent stores with identical chunks and query vectors."""

    def __init__(
        self,
        *,
        project_root: Path,
        database_url: str,
        replacement_rounds: int = DEFAULT_REPLACEMENT_ROUNDS,
        query_rounds: int = DEFAULT_QUERY_ROUNDS,
    ) -> None:
        if replacement_rounds <= 0 or query_rounds <= 0:
            raise ValueError("comparison round counts must be positive")
        self._project_root = project_root
        self._database_url = database_url
        self._replacement_rounds = replacement_rounds
        self._query_rounds = query_rounds

    def run(self, backends: Sequence[str]) -> VectorStoreComparisonReport:
        """Run selected backends and retain their persistence artifacts."""
        invalid = set(backends) - set(_BACKENDS)
        if invalid:
            raise ValueError(f"unsupported vector backends: {sorted(invalid)}")
        if not backends:
            raise ValueError("at least one vector backend is required")

        sections = load_policy(self._project_root / "data" / "policy.md")
        cases = load_gold_cases(
            self._project_root / "data" / "gold" / "gold-data.md"
        )
        provider = SentenceTransformerEmbeddingProvider(
            SELECTED_EMBEDDING_MODEL
        )
        chunks = embed_sections(sections, provider)
        query_embeddings = tuple(
            embed_query(case.question, provider) for case in cases
        )

        results = tuple(
            self._benchmark_backend(
                backend,
                chunks,
                cases,
                query_embeddings,
            )
            for backend in backends
        )
        return VectorStoreComparisonReport(
            embedding_model=SELECTED_EMBEDDING_MODEL,
            dimension=provider.dimension,
            replacement_rounds=self._replacement_rounds,
            query_rounds=self._query_rounds,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": "CPU",
            },
            results=results,
            selected_backend=SELECTED_VECTOR_STORE,
            selection_status="Selected by user",
        )

    def _benchmark_backend(
        self,
        backend: str,
        chunks: tuple[PolicyChunk, ...],
        cases: tuple[GoldCase, ...],
        query_embeddings: tuple[tuple[float, ...], ...],
    ) -> VectorStoreBenchmarkResult:
        factory, artifact_path = self._factory(backend, len(chunks[0].embedding))
        store = factory()
        replacement_times: list[float] = []
        query_times: list[float] = []
        try:
            store.replace_all(chunks)
            for _ in range(self._replacement_rounds):
                started = perf_counter()
                store.replace_all(chunks)
                replacement_times.append(_elapsed_ms(started))

            for _ in range(self._query_rounds):
                for query_embedding in query_embeddings:
                    started = perf_counter()
                    store.search(query_embedding)
                    query_times.append(_elapsed_ms(started))

            query_results = tuple(
                _evaluate_query(store, case, query_embedding, chunks)
                for case, query_embedding in zip(
                    cases,
                    query_embeddings,
                    strict=True,
                )
            )
        finally:
            store.close()

        reopened = factory()
        try:
            reopen_count = reopened.count()
            if reopen_count != len(chunks):
                raise RuntimeError(
                    f"{backend} reopened with {reopen_count} chunks; "
                    f"expected {len(chunks)}"
                )
        finally:
            reopened.close()

        supported_ranks = [
            result.expected_rank
            for result in query_results
            if result.expected_section is not None
        ]
        return VectorStoreBenchmarkResult(
            backend=backend,
            hit_at_1=mean(rank == 1 for rank in supported_ranks),
            hit_at_3=mean(
                rank is not None and rank <= 3 for rank in supported_ranks
            ),
            mean_reciprocal_rank=mean(
                0.0 if rank is None else 1.0 / rank
                for rank in supported_ranks
            ),
            exact_reference_parity=all(
                result.exact_reference_parity for result in query_results
            ),
            reopen_count=reopen_count,
            artifact_size_mb=(
                _directory_size_mb(artifact_path)
                if artifact_path is not None
                else None
            ),
            timing=StoreTimingMetrics(
                replacement_p50_ms=median(replacement_times),
                replacement_p95_ms=_percentile(replacement_times, 0.95),
                query_p50_ms=median(query_times),
                query_p95_ms=_percentile(query_times, 0.95),
            ),
            query_results=query_results,
        )

    def _factory(
        self,
        backend: str,
        dimension: int,
    ) -> tuple[Callable[[], VectorStore], Path | None]:
        collection_name = "expense-policy-comparison"
        if backend == "chroma":
            from expense_rag.vector_stores.chroma_store import ChromaVectorStore

            path = self._project_root / "data" / "artifacts" / "chroma"
            return (
                lambda: ChromaVectorStore(
                    directory=path,
                    collection_name=collection_name,
                    dimension=dimension,
                ),
                path,
            )
        if backend == "faiss":
            from expense_rag.vector_stores.faiss_store import FaissVectorStore

            path = self._project_root / "data" / "artifacts" / "faiss"
            return (
                lambda: FaissVectorStore(
                    directory=path,
                    collection_name=collection_name,
                    dimension=dimension,
                ),
                path,
            )

        from expense_rag.vector_stores.pgvector_store import PgVectorStore

        return (
            lambda: PgVectorStore(
                database_url=self._database_url,
                collection_name=collection_name,
                dimension=dimension,
            ),
            None,
        )


def _evaluate_query(
    store: VectorStore,
    case: GoldCase,
    query_embedding: tuple[float, ...],
    chunks: tuple[PolicyChunk, ...],
) -> StoreQueryResult:
    actual = store.search(query_embedding)
    exact = cosine_search(query_embedding, chunks)
    top_sections = tuple(result.chunk.section for result in actual)
    expected_rank: int | None = None
    if case.expected_section is not None:
        try:
            expected_rank = top_sections.index(case.expected_section) + 1
        except ValueError:
            expected_rank = None

    return StoreQueryResult(
        question=case.question,
        expected_section=case.expected_section,
        top_sections=top_sections,
        distances=tuple(result.distance for result in actual),
        expected_rank=expected_rank,
        exact_reference_parity=_results_match(actual, exact),
    )


def _results_match(
    actual: Sequence[SearchResult],
    expected: Sequence[SearchResult],
) -> bool:
    return (
        [result.chunk.chunk_id for result in actual]
        == [result.chunk.chunk_id for result in expected]
        and all(
            abs(actual_result.distance - expected_result.distance) <= 1e-5
            for actual_result, expected_result in zip(
                actual,
                expected,
                strict=True,
            )
        )
    )


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


def _directory_size_mb(path: Path) -> float:
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file()
    ) / (1024.0 * 1024.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--backends",
        default=",".join(_BACKENDS),
        help="Comma-separated subset of chroma, faiss, pgvector.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="PostgreSQL URL. Defaults to DATABASE_URL from the environment or .env.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/evaluation/vector-store-comparison.json"),
    )
    parser.add_argument(
        "--replacement-rounds",
        type=int,
        default=DEFAULT_REPLACEMENT_ROUNDS,
    )
    parser.add_argument(
        "--query-rounds",
        type=int,
        default=DEFAULT_QUERY_ROUNDS,
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested persistent backend comparison."""
    args = _parse_args()
    project_root = args.project_root.resolve()
    backends = tuple(
        backend.strip()
        for backend in args.backends.split(",")
        if backend.strip()
    )
    database_url = args.database_url
    if database_url is None and "pgvector" in backends:
        database_url = get_database_url()
    runner = VectorStoreComparisonRunner(
        project_root=project_root,
        database_url=database_url or "",
        replacement_rounds=args.replacement_rounds,
        query_rounds=args.query_rounds,
    )
    report = runner.run(backends)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(report.model_dump_json(indent=2))
    print(f"Raw report: {output_path}")


if __name__ == "__main__":
    main()
