"""Tests for the selected default vector-store backend."""

from expense_rag.env import get_vector_store
from expense_rag.vector_stores.factory import SUPPORTED_VECTOR_STORES


def test_selected_vector_store_comes_from_env() -> None:
    backend = get_vector_store()
    assert backend == "pgvector"
    assert backend in SUPPORTED_VECTOR_STORES
