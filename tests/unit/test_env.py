"""Tests for gitignored .env loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from expense_rag.env import (
    MissingDatabaseUrlError,
    MissingEmbeddingModelError,
    MissingVectorStoreError,
    UnsupportedVectorStoreError,
    get_database_url,
    get_embedding_model,
    get_vector_store,
    load_project_env,
    project_root,
)
from expense_rag.vector_stores.factory import SUPPORTED_VECTOR_STORES


def test_project_root_contains_pyproject() -> None:
    assert (project_root() / "pyproject.toml").is_file()


def test_load_project_env_reads_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://test:test@localhost:5432/test\n",
        encoding="utf-8",
    )

    load_project_env(root=tmp_path)

    assert get_database_url(root=tmp_path) == (
        "postgresql://test:test@localhost:5432/test"
    )


def test_exported_database_url_wins_over_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://exported:secret@db:5432/app")
    (tmp_path / ".env").write_text(
        "DATABASE_URL=postgresql://file:secret@db:5432/app\n",
        encoding="utf-8",
    )

    assert get_database_url(root=tmp_path) == (
        "postgresql://exported:secret@db:5432/app"
    )


def test_missing_database_url_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    (tmp_path / ".env").write_text("VECTOR_STORE=pgvector\n", encoding="utf-8")

    with pytest.raises(MissingDatabaseUrlError, match="DATABASE_URL"):
        get_database_url(root=tmp_path)


def test_load_project_env_reads_embedding_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2\n",
        encoding="utf-8",
    )

    assert get_embedding_model(root=tmp_path) == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_exported_embedding_model_wins_over_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    (tmp_path / ".env").write_text(
        "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2\n",
        encoding="utf-8",
    )

    assert get_embedding_model(root=tmp_path) == "BAAI/bge-small-en-v1.5"


def test_missing_embedding_model_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    (tmp_path / ".env").write_text("VECTOR_STORE=pgvector\n", encoding="utf-8")

    with pytest.raises(MissingEmbeddingModelError, match="EMBEDDING_MODEL"):
        get_embedding_model(root=tmp_path)


def test_load_project_env_reads_vector_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    (tmp_path / ".env").write_text("VECTOR_STORE=faiss\n", encoding="utf-8")

    assert get_vector_store(root=tmp_path) == "faiss"


def test_exported_vector_store_wins_over_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    (tmp_path / ".env").write_text("VECTOR_STORE=pgvector\n", encoding="utf-8")

    assert get_vector_store(root=tmp_path) == "chroma"


def test_missing_vector_store_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    (tmp_path / ".env").write_text(
        "EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2\n",
        encoding="utf-8",
    )

    with pytest.raises(MissingVectorStoreError, match="VECTOR_STORE"):
        get_vector_store(root=tmp_path)


def test_unsupported_vector_store_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    (tmp_path / ".env").write_text("VECTOR_STORE=pinecone\n", encoding="utf-8")

    with pytest.raises(UnsupportedVectorStoreError, match="pinecone"):
        get_vector_store(root=tmp_path)


def test_env_example_defines_runtime_knobs() -> None:
    values = _dotenv_values(project_root() / ".env.example")

    assert values["EMBEDDING_MODEL"] == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert values["VECTOR_STORE"] == "pgvector"
    assert values["VECTOR_STORE"] in SUPPORTED_VECTOR_STORES


def _dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def test_dotenv_file_is_gitignored() -> None:
    gitignore = (project_root() / ".gitignore").read_text(encoding="utf-8")
    ignored = {
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".env" in ignored
