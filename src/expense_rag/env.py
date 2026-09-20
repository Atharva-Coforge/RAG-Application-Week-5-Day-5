"""Load gitignored project secrets without printing them."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class MissingDatabaseUrlError(ValueError):
    """Raised when pgvector needs DATABASE_URL and none is configured."""


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
    load_project_env(root=root)
    value = os.environ.get("DATABASE_URL", "").strip()
    if not value:
        raise MissingDatabaseUrlError(
            "DATABASE_URL is required for pgvector. "
            "Copy .env.example to .env and set the real connection string."
        )
    return value
