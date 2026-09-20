"""Load gitignored project secrets without printing them."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from expense_rag.generation.providers.ollama_provider import CANDIDATE_OLLAMA_MODELS
from expense_rag.vector_stores.factory import SUPPORTED_VECTOR_STORES


class MissingDatabaseUrlError(ValueError):
    """Raised when pgvector needs DATABASE_URL and none is configured."""


class MissingEmbeddingModelError(ValueError):
    """Raised when EMBEDDING_MODEL is missing from the environment or `.env`."""


class MissingVectorStoreError(ValueError):
    """Raised when VECTOR_STORE is missing from the environment or `.env`."""


class MissingOllamaHostError(ValueError):
    """Raised when Ollama generation needs OLLAMA_HOST and none is configured."""


class MissingGenerationModelError(ValueError):
    """Raised when GENERATION_MODEL is missing from the environment or `.env`."""


class UnsupportedGenerationModelError(ValueError):
    """Raised when GENERATION_MODEL is not one of the compared Ollama models."""


class UnsupportedVectorStoreError(ValueError):
    """Raised when VECTOR_STORE is not one of the implemented adapters."""


def project_root() -> Path:
    """Return the repository root that contains pyproject.toml."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("could not locate project root")


def load_project_env(*, root: Path | None = None) -> Path:
    """Load `.env` from the project root without overriding exported values."""
    env_path = (root or project_root()) / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
    return env_path


def get_database_url(*, root: Path | None = None) -> str:
    """Return DATABASE_URL from the process environment or project `.env`."""
    return _required_env("DATABASE_URL", MissingDatabaseUrlError, root=root)


def get_embedding_model(*, root: Path | None = None) -> str:
    """Return EMBEDDING_MODEL from the process environment or project `.env`."""
    return _required_env(
        "EMBEDDING_MODEL",
        MissingEmbeddingModelError,
        root=root,
    )


def get_vector_store(*, root: Path | None = None) -> str:
    """Return a supported VECTOR_STORE from the environment or project `.env`."""
    value = _required_env(
        "VECTOR_STORE",
        MissingVectorStoreError,
        root=root,
    ).lower()
    if value not in SUPPORTED_VECTOR_STORES:
        supported = ", ".join(SUPPORTED_VECTOR_STORES)
        raise UnsupportedVectorStoreError(
            f"unsupported VECTOR_STORE {value!r}; choose one of {supported}"
        )
    return value


def get_ollama_host(*, root: Path | None = None) -> str:
    """Return OLLAMA_HOST from the process environment or project `.env`."""
    return _required_env("OLLAMA_HOST", MissingOllamaHostError, root=root)


def get_generation_model(*, root: Path | None = None) -> str:
    """Return a compared Ollama model from the environment or project `.env`."""
    value = _required_env(
        "GENERATION_MODEL",
        MissingGenerationModelError,
        root=root,
    )
    if value not in CANDIDATE_OLLAMA_MODELS:
        supported = ", ".join(CANDIDATE_OLLAMA_MODELS)
        raise UnsupportedGenerationModelError(
            f"unsupported GENERATION_MODEL {value!r}; choose one of {supported}"
        )
    return value


def _required_env(
    name: str,
    error_type: type[ValueError],
    *,
    root: Path | None = None,
) -> str:
    load_project_env(root=root)
    value = os.environ.get(name, "").strip()
    if not value:
        raise error_type(
            f"{name} is required. "
            "Copy .env.example to .env and set the selected value."
        )
    return value
