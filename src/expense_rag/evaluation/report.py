"""Write timestamped acceptance reports for manual review."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

from expense_rag.evaluation.runner import EvaluationReport


def write_evaluation_report(
    report: EvaluationReport,
    *,
    directory: Path,
) -> Path:
    """Write one JSON artifact under ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"acceptance-{stamp}.json"
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
