"""Tests for the Markdown gold-evaluation data loader."""

from pathlib import Path

import pytest

from expense_rag.evaluation.gold_loader import (
    GoldDataParseError,
    load_gold_cases,
    parse_gold_cases,
)
from expense_rag.models import REFUSAL_ANSWER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "gold-data.md"


def _gold_text() -> str:
    return GOLD_PATH.read_text(encoding="utf-8")


def _gold_body() -> str:
    return _gold_text().split("\n", maxsplit=1)[1]


def test_load_gold_cases_produces_six_valid_cases() -> None:
    cases = load_gold_cases(GOLD_PATH)

    assert len(cases) == 6
    assert [case.expected_section for case in cases] == [
        "1",
        "3",
        "2",
        "5",
        "4",
        None,
    ]
    assert [case.supported for case in cases] == [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert cases[-1].required_answer == REFUSAL_ANSWER


def test_gold_loader_joins_wrapped_answer_lines() -> None:
    first_case = load_gold_cases(GOLD_PATH)[0]

    assert first_case.required_answer == (
        "Employees may claim up to $65 per day for meals while "
        "traveling overnight."
    )


@pytest.mark.parametrize(
    ("gold_data", "message"),
    [
        ("", "empty"),
        (
            "# Evaluation Data\n" + _gold_body(),
            "first heading",
        ),
        (
            _gold_text().replace(
                "- `supported`: true",
                "- `supported`: yes",
                1,
            ),
            "must be true or false",
        ),
        (
            _gold_text().replace(
                "- `expected_section`: 1",
                "",
                1,
            ),
            "missing fields: expected_section",
        ),
        (
            _gold_text().replace("## Case 2", "## Case 1", 1),
            "duplicate case 1",
        ),
        (
            _gold_text().split("\n## Case 6", maxsplit=1)[0],
            "expected cases",
        ),
        (
            _gold_text().replace(
                "- `expected_section`: 1",
                "- `expected_section`: null",
                1,
            ),
            "must define an expected section",
        ),
        (
            _gold_text().replace(
                "- `supported`: true",
                "- `answer_style`: concise",
                1,
            ),
            "unrecognized content",
        ),
    ],
)
def test_gold_loader_rejects_malformed_data(
    gold_data: str,
    message: str,
) -> None:
    with pytest.raises(GoldDataParseError, match=message):
        parse_gold_cases(gold_data)
