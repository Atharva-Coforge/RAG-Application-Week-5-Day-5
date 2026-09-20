"""Validated runtime settings after all technology decisions are closed."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from expense_rag.retrieval.cosine import MAX_RETRIEVAL_RESULTS

SELECTED_EMBEDDING_DIMENSION = 384
SELECTED_GENERATION_PROVIDER: Literal["ollama"] = "ollama"
DEFAULT_COLLECTION_NAME = "expense-policy"
POLICY_RELATIVE_PATH = Path("data") / "policy.md"
GOLD_RELATIVE_PATH = Path("data") / "gold" / "gold-data.md"
ARTIFACTS_RELATIVE_PATH = Path("data") / "artifacts"
CHROMA_RELATIVE_PATH = Path("data") / "artifacts" / "chroma"
FAISS_RELATIVE_PATH = Path("data") / "artifacts" / "faiss"
_PLACEHOLDER_VALUES = frozenset({"TBD", "TODO", "CHANGEME"})


class VectorBackend(StrEnum):
    """Allowed vector-store backends after the Step 6 comparison."""

    CHROMA = "chroma"
    FAISS = "faiss"
    PGVECTOR = "pgvector"


class Settings(BaseModel):
    """Complete runtime configuration with no placeholder values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_root: Path
    policy_path: Path
    gold_path: Path
    artifacts_dir: Path
    chroma_directory: Path
    faiss_directory: Path
    collection_name: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1)
    vector_backend: VectorBackend
    generation_provider: Literal["ollama"]
    generation_model: str = Field(min_length=1)
    ollama_host: str = Field(min_length=1)
    database_url: SecretStr | None = None
    top_k: int = Field(default=MAX_RETRIEVAL_RESULTS, ge=1, le=MAX_RETRIEVAL_RESULTS)

    @classmethod
    def load(cls, *, root: Path | None = None) -> Self:
        """Load settings from the environment or project `.env` file."""
        from expense_rag.env import (
            get_database_url,
            get_embedding_model,
            get_generation_model,
            get_ollama_host,
            get_vector_store,
            project_root,
        )

        project = (root or project_root()).resolve()
        backend = VectorBackend(get_vector_store(root=project))
        database_url = None
        if backend is VectorBackend.PGVECTOR:
            database_url = SecretStr(get_database_url(root=project))

        return cls(
            project_root=project,
            policy_path=_resolve_under(project, POLICY_RELATIVE_PATH),
            gold_path=_resolve_under(project, GOLD_RELATIVE_PATH),
            artifacts_dir=_resolve_under(project, ARTIFACTS_RELATIVE_PATH),
            chroma_directory=_resolve_under(project, CHROMA_RELATIVE_PATH),
            faiss_directory=_resolve_under(project, FAISS_RELATIVE_PATH),
            collection_name=DEFAULT_COLLECTION_NAME,
            embedding_model=get_embedding_model(root=project),
            embedding_dimension=SELECTED_EMBEDDING_DIMENSION,
            vector_backend=backend,
            generation_provider=SELECTED_GENERATION_PROVIDER,
            generation_model=get_generation_model(root=project),
            ollama_host=get_ollama_host(root=project),
            database_url=database_url,
            top_k=MAX_RETRIEVAL_RESULTS,
        )

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> Self:
        """Reject incomplete, placeholder, or out-of-root configuration."""
        for name in (
            "collection_name",
            "embedding_model",
            "generation_provider",
            "generation_model",
            "ollama_host",
        ):
            _reject_placeholder(name, getattr(self, name))

        if self.embedding_dimension != SELECTED_EMBEDDING_DIMENSION:
            raise ValueError(
                "embedding_dimension must be "
                f"{SELECTED_EMBEDDING_DIMENSION} for the selected MiniLM model"
            )

        if self.vector_backend is VectorBackend.PGVECTOR:
            if self.database_url is None:
                raise ValueError("DATABASE_URL is required when VECTOR_STORE=pgvector")
            _reject_placeholder("database_url", self.database_url.get_secret_value())
        elif self.database_url is not None:
            _reject_placeholder("database_url", self.database_url.get_secret_value())

        project = self.project_root.resolve()
        for path in (
            self.policy_path,
            self.gold_path,
            self.artifacts_dir,
            self.chroma_directory,
            self.faiss_directory,
        ):
            resolved = path if path.is_absolute() else project / path
            try:
                resolved.resolve().relative_to(project)
            except ValueError as error:
                raise ValueError(
                    f"path {path} must resolve under the project root"
                ) from error

        return self


def _resolve_under(root: Path, relative: Path) -> Path:
    return (root / relative).resolve()


def _reject_placeholder(name: str, value: str) -> None:
    stripped = value.strip()
    if not stripped or stripped.upper() in _PLACEHOLDER_VALUES:
        raise ValueError(f"{name} is missing or still a placeholder")
