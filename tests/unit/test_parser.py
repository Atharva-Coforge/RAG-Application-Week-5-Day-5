"""Tests for structural policy parsing."""

from pathlib import Path
from textwrap import dedent

import pytest

from expense_rag.ingestion.parser import (
    PolicyParseError,
    load_policy,
    parse_policy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _valid_policy(
    section_one_body: str = "Body for section 1.",
) -> str:
    indented_body = section_one_body.replace("\n", "\n        ")
    return dedent(
        f"""\
        # Employee Expense Policy — Version 2.0

        ## 1. Meals
        {indented_body}

        ## 2. Hotels
        Body for section 2.

        ## 3. Airfare
        Body for section 3.

        ## 4. Ground Transportation
        Body for section 4.

        ## 5. Receipts
        Body for section 5.

        ## 6. Submission Deadline
        Body for section 6.
        """
    )


def test_load_policy_produces_six_ordered_sections() -> None:
    sections = load_policy(PROJECT_ROOT / "data" / "policy.md")

    assert len(sections) == 6
    assert [section.section for section in sections] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert [section.section_title for section in sections] == [
        "Meals",
        "Hotels",
        "Airfare",
        "Ground Transportation",
        "Receipts",
        "Submission Deadline",
    ]
    assert sections[0].chunk_id == "expense-policy:v2.0:section-1"
    assert sections[-1].chunk_id == "expense-policy:v2.0:section-6"
    assert all(section.document == "Employee Expense Policy" for section in sections)
    assert all(section.version == "2.0" for section in sections)


def test_policy_text_contains_body_only_and_preserves_paragraphs() -> None:
    policy = _valid_policy(
        section_one_body=(
            "First paragraph remains intact.\n\nSecond paragraph remains intact."
        )
    )

    section = parse_policy(policy)[0]

    assert section.text == (
        "First paragraph remains intact.\n\nSecond paragraph remains intact."
    )
    assert "## 1. Meals" not in section.text


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("", "empty"),
        ("# Employee Expense Policy\n", "first heading"),
        (
            _valid_policy().replace("## 2. Hotels", "## 2 Hotels"),
            "malformed section heading",
        ),
        (
            _valid_policy().replace(
                "## 3. Airfare",
                "## 2. Duplicate Hotels",
            ),
            "duplicate section 2",
        ),
        (
            _valid_policy().replace(
                "## 3. Airfare\nBody for section 3.",
                "## 3. Airfare",
            ),
            "section 3 has no body text",
        ),
        (
            _valid_policy().replace(
                "## 6. Submission Deadline\nBody for section 6.",
                "",
            ),
            "expected sections",
        ),
    ],
)
def test_policy_parser_rejects_malformed_documents(
    policy: str,
    message: str,
) -> None:
    with pytest.raises(PolicyParseError, match=message):
        parse_policy(policy)


def test_policy_parser_rejects_content_before_first_section() -> None:
    policy = _valid_policy().replace(
        "\n\n## 1. Meals",
        "\n\nUnexpected introduction.\n\n## 1. Meals",
    )

    with pytest.raises(PolicyParseError, match="before the first numbered section"):
        parse_policy(policy)
