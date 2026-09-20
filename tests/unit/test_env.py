"""Tests for gitignored .env loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from expense_rag.env import (
    MissingDatabaseUrlError,
    get_database_url,
    load_project_env,
    project_root,
)


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


def test_dotenv_file_is_gitignored() -> None:
    gitignore = (project_root() / ".gitignore").read_text(encoding="utf-8")
    ignored = {
        line.strip()
        for line in gitignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".env" in ignored
