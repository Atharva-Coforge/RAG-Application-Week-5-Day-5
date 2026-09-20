"""Pytest options for explicit vector-store integration runs."""

from __future__ import annotations

import pytest

from expense_rag.env import load_project_env

load_project_env()

_KNOWN_VECTOR_BACKENDS = {"memory", "chroma", "faiss", "pgvector"}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vector-backends",
        default="memory",
        help=(
            "Comma-separated contract backends: memory, chroma, faiss, pgvector. "
            "Real backends require their optional dependencies; pgvector also "
            "requires a running PostgreSQL service."
        ),
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend_name" not in metafunc.fixturenames:
        return

    raw_value = str(metafunc.config.getoption("--vector-backends"))
    backend_names = tuple(
        name.strip() for name in raw_value.split(",") if name.strip()
    )
    if not backend_names:
        raise pytest.UsageError("--vector-backends must contain at least one backend")

    unknown = set(backend_names) - _KNOWN_VECTOR_BACKENDS
    if unknown:
        choices = ", ".join(sorted(_KNOWN_VECTOR_BACKENDS))
        invalid = ", ".join(sorted(unknown))
        raise pytest.UsageError(
            f"unknown vector backends: {invalid}; choose from {choices}"
        )

    metafunc.parametrize("backend_name", backend_names, ids=backend_names)
