"""Tests for the selected default vector-store backend."""

from expense_rag.vector_stores.factory import (
    SELECTED_VECTOR_STORE,
    SUPPORTED_VECTOR_STORES,
)


def test_selected_vector_store_is_pgvector() -> None:
    assert SELECTED_VECTOR_STORE == "pgvector"
    assert SELECTED_VECTOR_STORE in SUPPORTED_VECTOR_STORES
