"""Tests for the timestamped acceptance report writer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from expense_rag.evaluation.report import write_evaluation_report
from expense_rag.evaluation.runner import EvaluationReport


def test_write_evaluation_report_uses_utc_timestamp(tmp_path: Path) -> None:
    report = EvaluationReport(
        generated_at=datetime(2026, 9, 20, 5, 4, 12, tzinfo=UTC),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension=384,
        vector_backend="pgvector",
        generation_provider="ollama",
        generation_model="qwen3:8b",
        top_k=3,
        stored_chunk_count=6,
        six_chunks_stored=True,
        retrieval_shape_valid=True,
        complete_records=True,
        supported_hit_at_3_count=5,
        supported_hit_at_3_required=5,
        supported_hit_at_3_passes=True,
        all_supported_grounded_and_cited=True,
        gym_refusal_correct=True,
        accepted=True,
        case_results=(),
    )

    path = write_evaluation_report(report, directory=tmp_path)

    assert path.name == "acceptance-20260920T050412Z.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["accepted"] is True
    assert payload["generation_model"] == "qwen3:8b"
    assert payload["gym_refusal_correct"] is True
    assert "database_url" not in payload
    assert "PASSWORD" not in path.read_text(encoding="utf-8")
