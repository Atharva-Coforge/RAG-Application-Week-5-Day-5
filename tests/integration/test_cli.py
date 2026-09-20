"""CLI assembly tests using protocol fakes, not live models or stores."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from expense_rag.cli import main
from expense_rag.config import VectorBackend
from expense_rag.env import MissingDatabaseUrlError, project_root
from expense_rag.generation.base import GenerationProvider
from expense_rag.ingestion.parser import EXPECTED_SECTION_COUNT
from tests.fakes import (
    InMemoryVectorStore,
    MappedGenerationProvider,
    gold_aware_embedding_provider,
    gold_aware_generation_provider,
)


@pytest.fixture
def fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[InMemoryVectorStore, MappedGenerationProvider]:
    store = InMemoryVectorStore(
        collection_name="expense-policy",
        dimension=EXPECTED_SECTION_COUNT,
    )
    store.close = lambda: None  # type: ignore[method-assign]
    embedder = gold_aware_embedding_provider()
    generator = gold_aware_generation_provider()
    captured: dict[str, object] = {}

    def build_store(settings: object) -> InMemoryVectorStore:
        captured["settings"] = settings
        return store

    monkeypatch.setattr(
        "expense_rag.cli.build_embedding_provider",
        lambda settings: embedder,
    )
    monkeypatch.setattr("expense_rag.cli.build_vector_store", build_store)
    monkeypatch.setattr(
        "expense_rag.cli.build_generation_provider",
        lambda settings: generator,
    )
    store.captured = captured  # type: ignore[attr-defined]
    return store, generator


def test_cli_imports_factories_not_provider_sdks() -> None:
    source = Path(inspect.getfile(main)).read_text(encoding="utf-8")

    assert "build_embedding_provider" in source
    assert "build_vector_store" in source
    assert "build_generation_provider" in source
    assert "chromadb" not in source
    assert "faiss" not in source
    assert "psycopg" not in source
    assert "sentence_transformers" not in source
    assert "from ollama" not in source


def test_ingest_prints_six_chunk_summary(
    fake_runtime: tuple[InMemoryVectorStore, MappedGenerationProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    store, _ = fake_runtime

    assert main(["ingest"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert store.count() == EXPECTED_SECTION_COUNT
    assert payload["chunk_count"] == EXPECTED_SECTION_COUNT
    assert payload["chunk_ids"] == [
        f"expense-policy:v2.0:section-{index}"
        for index in range(1, EXPECTED_SECTION_COUNT + 1)
    ]


def test_ingest_missing_policy_returns_nonzero(
    fake_runtime: tuple[InMemoryVectorStore, MappedGenerationProvider],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fake_runtime

    assert main(["ingest", "--policy", str(tmp_path / "missing.md")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "policy file not found" in captured.err


def test_ask_prints_assignment_response_shape(
    fake_runtime: tuple[InMemoryVectorStore, MappedGenerationProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["ingest"]) == 0
    capsys.readouterr()

    assert main(["ask", "How much can I spend on food each day?"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert "$65" in payload["answer"]
    assert payload["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }
    assert 1 <= len(payload["retrieved_chunks"]) <= 3
    distances = [chunk["distance"] for chunk in payload["retrieved_chunks"]]
    assert distances == sorted(distances)
    assert all(isinstance(distance, (int, float)) for distance in distances)
    assert {"section", "distance"} <= set(payload["retrieved_chunks"][0])


def test_ask_without_ingestion_returns_nonzero(
    fake_runtime: tuple[InMemoryVectorStore, MappedGenerationProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    del fake_runtime

    assert main(["ask", "How much can I spend on food each day?"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not been ingested" in captured.err


def test_evaluate_prints_acceptance_report(
    fake_runtime: tuple[InMemoryVectorStore, MappedGenerationProvider],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, generator = fake_runtime
    evaluation_dir = project_root() / "data" / "artifacts" / "evaluation"
    before = set(evaluation_dir.glob("acceptance-*.json")) if evaluation_dir.exists() else set()
    assert main(["ingest"]) == 0
    capsys.readouterr()

    assert main(["evaluate"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    created = set(evaluation_dir.glob("acceptance-*.json")) - before
    for path in created:
        path.unlink()

    assert payload["accepted"] is True
    assert payload["stored_chunk_count"] == EXPECTED_SECTION_COUNT
    assert payload["supported_hit_at_3_count"] == 5
    assert payload["gym_refusal_correct"] is True
    assert len(payload["case_results"]) == 6
    assert generator.call_count == 6
    first = payload["case_results"][0]
    assert {"answer", "citation", "retrieved_chunks"} <= set(first["response"])
    assert "acceptance passed" in captured.err
    assert "wrote " in captured.err


def test_backend_override_reaches_the_store_factory(
    fake_runtime: tuple[InMemoryVectorStore, MappedGenerationProvider],
) -> None:
    store, _ = fake_runtime

    assert main(["ingest", "--backend", "chroma"]) == 0
    settings = store.captured["settings"]  # type: ignore[attr-defined]
    assert settings.vector_backend is VectorBackend.CHROMA
    assert settings.database_url is None


def test_missing_credentials_return_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing(*, root: Path | None = None) -> object:
        del root
        raise MissingDatabaseUrlError(
            "DATABASE_URL is required. "
            "Copy .env.example to .env and set the selected value."
        )

    monkeypatch.setattr(
        "expense_rag.cli.Settings.load",
        classmethod(lambda cls, *, root=None: missing(root=root)),
    )

    assert main(["ingest"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "DATABASE_URL" in captured.err


def test_cli_generation_provider_is_protocol_compatible() -> None:
    provider = gold_aware_generation_provider()
    assert isinstance(provider, GenerationProvider)
