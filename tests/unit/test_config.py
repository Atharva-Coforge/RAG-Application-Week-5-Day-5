"""Tests for the finalized runtime Settings object."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from expense_rag.config import (
    SELECTED_EMBEDDING_DIMENSION,
    SELECTED_GENERATION_PROVIDER,
    Settings,
    VectorBackend,
)
from expense_rag.env import MissingDatabaseUrlError, project_root
from expense_rag.retrieval.cosine import MAX_RETRIEVAL_RESULTS
from expense_rag.vector_stores.factory import SUPPORTED_VECTOR_STORES


def _write_env(path: Path, **values: str) -> None:
    body = "\n".join(f"{key}={value}" for key, value in values.items())
    (path / ".env").write_text(body + "\n", encoding="utf-8")


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EMBEDDING_MODEL",
        "VECTOR_STORE",
        "GENERATION_MODEL",
        "OLLAMA_HOST",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _valid_settings(**overrides: object) -> Settings:
    root = Path(overrides.pop("project_root", Path("/tmp/expense-rag-settings")))
    values: dict[str, object] = {
        "project_root": root,
        "policy_path": root / "data" / "policy.md",
        "gold_path": root / "data" / "gold" / "gold-data.md",
        "artifacts_dir": root / "data" / "artifacts",
        "chroma_directory": root / "data" / "artifacts" / "chroma",
        "faiss_directory": root / "data" / "artifacts" / "faiss",
        "collection_name": "expense-policy",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": SELECTED_EMBEDDING_DIMENSION,
        "vector_backend": VectorBackend.PGVECTOR,
        "generation_provider": SELECTED_GENERATION_PROVIDER,
        "generation_model": "qwen3:8b",
        "ollama_host": "http://127.0.0.1:11434",
        "database_url": SecretStr("postgresql://user:secret-password@db:5432/app"),
        "top_k": MAX_RETRIEVAL_RESULTS,
    }
    values.update(overrides)
    return Settings.model_validate(values)


def test_vector_backend_matches_supported_stores() -> None:
    assert {backend.value for backend in VectorBackend} == set(
        SUPPORTED_VECTOR_STORES
    )


def test_settings_load_uses_project_root_not_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings.load()
    root = project_root()

    assert settings.project_root == root.resolve()
    assert settings.policy_path == (root / "data" / "policy.md").resolve()
    assert settings.gold_path == (root / "data" / "gold" / "gold-data.md").resolve()
    assert settings.policy_path.is_absolute()
    assert settings.gold_path.is_absolute()
    assert settings.top_k == MAX_RETRIEVAL_RESULTS
    assert settings.embedding_dimension == SELECTED_EMBEDDING_DIMENSION
    assert settings.generation_provider == SELECTED_GENERATION_PROVIDER
    assert settings.collection_name == "expense-policy"


def test_settings_load_reads_env_from_provided_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_env(
        tmp_path,
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        VECTOR_STORE="pgvector",
        GENERATION_MODEL="qwen3:8b",
        OLLAMA_HOST="http://host.docker.internal:11434",
        DATABASE_URL="postgresql://test:test@localhost:5432/test",
    )

    settings = Settings.load(root=tmp_path)

    assert settings.vector_backend is VectorBackend.PGVECTOR
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.generation_model == "qwen3:8b"
    assert settings.ollama_host == "http://host.docker.internal:11434"
    assert settings.database_url is not None
    assert (
        settings.database_url.get_secret_value()
        == "postgresql://test:test@localhost:5432/test"
    )


def test_settings_chroma_does_not_require_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_env(
        tmp_path,
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        VECTOR_STORE="chroma",
        GENERATION_MODEL="qwen3:8b",
        OLLAMA_HOST="http://127.0.0.1:11434",
    )

    settings = Settings.load(root=tmp_path)

    assert settings.vector_backend is VectorBackend.CHROMA
    assert settings.database_url is None


def test_settings_pgvector_requires_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_env(monkeypatch)
    _write_env(
        tmp_path,
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        VECTOR_STORE="pgvector",
        GENERATION_MODEL="qwen3:8b",
        OLLAMA_HOST="http://127.0.0.1:11434",
    )

    with pytest.raises(MissingDatabaseUrlError, match="DATABASE_URL"):
        Settings.load(root=tmp_path)


def test_settings_rejects_top_k_above_three() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        _valid_settings(top_k=4)


def test_settings_rejects_top_k_below_one() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        _valid_settings(top_k=0)


def test_settings_rejects_placeholder_model_name() -> None:
    with pytest.raises(ValidationError, match="placeholder"):
        _valid_settings(embedding_model="TBD")


def test_settings_rejects_path_outside_project_root() -> None:
    with pytest.raises(ValidationError, match="project root"):
        _valid_settings(policy_path=Path("/etc/passwd"))


def test_settings_repr_does_not_print_database_secret() -> None:
    secret = "super-secret-password"
    settings = _valid_settings(
        database_url=SecretStr(f"postgresql://user:{secret}@db:5432/app")
    )

    assert secret not in repr(settings)
    assert secret not in str(settings)
