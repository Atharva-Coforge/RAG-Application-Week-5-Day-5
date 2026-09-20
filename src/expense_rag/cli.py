"""Assemble tested services into ingest, ask, and evaluate commands."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from expense_rag.config import Settings, VectorBackend
from expense_rag.embeddings.base import EmbeddingProvider
from expense_rag.embeddings.provider import (
    EmbeddingContractError,
    build_embedding_provider,
)
from expense_rag.env import (
    MissingDatabaseUrlError,
    MissingEmbeddingModelError,
    MissingGenerationModelError,
    MissingOllamaHostError,
    MissingVectorStoreError,
    UnsupportedGenerationModelError,
    UnsupportedVectorStoreError,
)
from expense_rag.evaluation.gold_loader import GoldDataParseError, load_gold_cases
from expense_rag.generation.base import GenerationProvider
from expense_rag.generation.providers import build_generation_provider
from expense_rag.generation.service import GenerationContractError, GenerationService
from expense_rag.ingestion.service import IngestionService
from expense_rag.models import RagResponse
from expense_rag.retrieval.service import RetrievalService
from expense_rag.vector_stores.base import VectorStore, VectorStoreError
from expense_rag.vector_stores.factory import build_vector_store

_USER_ERRORS = (
    MissingDatabaseUrlError,
    MissingEmbeddingModelError,
    MissingGenerationModelError,
    MissingOllamaHostError,
    MissingVectorStoreError,
    UnsupportedGenerationModelError,
    UnsupportedVectorStoreError,
    VectorStoreError,
    EmbeddingContractError,
    GenerationContractError,
    GoldDataParseError,
    FileNotFoundError,
    ValueError,
    RuntimeError,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, assemble dependencies, and return a process exit code."""
    args = _build_parser().parse_args(None if argv is None else list(argv))
    try:
        settings = _settings_from_args(args)
        return int(args.handler(args, settings))
    except _USER_ERRORS as error:
        print(error, file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="expense-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest",
        help="Parse, embed, and persist the expense policy.",
    )
    ingest.add_argument("--policy", type=Path, help="Path to policy Markdown.")
    ingest.add_argument(
        "--backend",
        choices=tuple(backend.value for backend in VectorBackend),
        help="Override VECTOR_STORE for this command.",
    )
    ingest.set_defaults(handler=_ingest_command)

    ask = subparsers.add_parser(
        "ask",
        help="Retrieve evidence and generate one grounded answer.",
    )
    ask.add_argument("question", help="User question to answer from the policy.")
    ask.add_argument(
        "--backend",
        choices=tuple(backend.value for backend in VectorBackend),
        help="Override VECTOR_STORE for this command.",
    )
    ask.set_defaults(handler=_ask_command)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Ask the six gold questions and print a JSON artifact.",
    )
    evaluate.add_argument("--gold", type=Path, help="Path to gold-data Markdown.")
    evaluate.add_argument(
        "--backend",
        choices=tuple(backend.value for backend in VectorBackend),
        help="Override VECTOR_STORE for this command.",
    )
    evaluate.add_argument(
        "--model",
        help="Override GENERATION_MODEL for this command.",
    )
    evaluate.set_defaults(handler=_evaluate_command)
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.load()
    backend = getattr(args, "backend", None)
    if backend is not None:
        settings = settings.with_backend(VectorBackend(backend))
    model_name = getattr(args, "model", None)
    if model_name is not None:
        settings = settings.with_generation_model(model_name)
    return settings


def _ingest_command(args: argparse.Namespace, settings: Settings) -> int:
    policy_path = args.policy or settings.policy_path
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy file not found: {policy_path}")

    embedding_provider = build_embedding_provider(settings)
    with build_vector_store(settings) as store:
        _, summary = IngestionService(
            embedding_provider=embedding_provider,
            vector_store=store,
        ).ingest(policy_path)

    _print_json(summary.model_dump(mode="json"))
    return 0


def _ask_command(args: argparse.Namespace, settings: Settings) -> int:
    embedding_provider = build_embedding_provider(settings)
    generation_provider = build_generation_provider(settings)
    with build_vector_store(settings) as store:
        response = _answer_question(
            settings,
            args.question,
            embedding_provider=embedding_provider,
            generation_provider=generation_provider,
            vector_store=store,
        )
    _print_json(response.model_dump(mode="json"))
    return 0


def _evaluate_command(args: argparse.Namespace, settings: Settings) -> int:
    gold_path = args.gold or settings.gold_path
    if not gold_path.is_file():
        raise FileNotFoundError(f"gold file not found: {gold_path}")

    cases = load_gold_cases(gold_path)
    embedding_provider = build_embedding_provider(settings)
    generation_provider = build_generation_provider(settings)
    results: list[dict[str, object]] = []
    with build_vector_store(settings) as store:
        for case in cases:
            response = _answer_question(
                settings,
                case.question,
                embedding_provider=embedding_provider,
                generation_provider=generation_provider,
                vector_store=store,
            )
            results.append(
                {
                    "question": case.question,
                    "supported": case.supported,
                    "response": response.model_dump(mode="json"),
                }
            )

    _print_json(
        {
            "backend": settings.vector_backend.value,
            "generation_model": settings.generation_model,
            "case_count": len(results),
            "results": results,
        }
    )
    print(
        f"evaluated {len(results)} cases using "
        f"{settings.vector_backend.value} and {settings.generation_model}",
        file=sys.stderr,
    )
    return 0


def _answer_question(
    settings: Settings,
    question: str,
    *,
    embedding_provider: EmbeddingProvider,
    generation_provider: GenerationProvider,
    vector_store: VectorStore,
) -> RagResponse:
    if vector_store.count() == 0:
        raise FileNotFoundError("policy has not been ingested")

    retrieved = RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    ).retrieve(question, top_k=settings.top_k)
    return GenerationService(
        generation_provider=generation_provider,
    ).generate(question, retrieved)


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
