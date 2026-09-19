"""Parser for the project's fixed Markdown gold-evaluation dataset."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import ValidationError

from expense_rag.models import GoldCase

EXPECTED_CASE_COUNT = 6

_DATASET_HEADING = "# Gold Evaluation Dataset"
_CASE_HEADING = re.compile(r"^##\s+Case\s+(?P<number>[1-9]\d*)\s*$")
_FIELD = re.compile(
    r"^-\s+`(?P<name>question|supported|required_answer|expected_section)`:"
    r"\s*(?P<value>.*)$"
)
_EXPECTED_FIELDS = {
    "question",
    "supported",
    "required_answer",
    "expected_section",
}


class GoldDataParseError(ValueError):
    """Raised when the gold Markdown violates the required six-case format."""


def load_gold_cases(path: Path) -> tuple[GoldCase, ...]:
    """Read and parse a UTF-8 gold-data Markdown file."""
    return parse_gold_cases(path.read_text(encoding="utf-8"))


def parse_gold_cases(markdown: str) -> tuple[GoldCase, ...]:
    """Parse the project's simple case-heading and field-list Markdown format."""
    lines = markdown.removeprefix("\ufeff").splitlines()
    heading_index = _first_non_empty_line(lines)
    if heading_index is None:
        raise GoldDataParseError("gold dataset is empty")
    if lines[heading_index].strip() != _DATASET_HEADING:
        raise GoldDataParseError(f"first heading must be {_DATASET_HEADING!r}")

    cases: list[GoldCase] = []
    case_numbers: list[int] = []
    current_case: int | None = None
    fields: dict[str, str] = {}
    current_field: str | None = None

    def finish_current_case() -> None:
        if current_case is None:
            return

        missing = _EXPECTED_FIELDS - fields.keys()
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise GoldDataParseError(
                f"case {current_case} is missing fields: {missing_list}"
            )

        try:
            cases.append(
                GoldCase(
                    question=fields["question"],
                    supported=_parse_supported(fields["supported"], current_case),
                    required_answer=fields["required_answer"],
                    expected_section=_parse_expected_section(
                        fields["expected_section"]
                    ),
                )
            )
        except ValidationError as error:
            raise GoldDataParseError(
                f"case {current_case} is invalid: {error.errors()[0]['msg']}"
            ) from error

    for line in lines[heading_index + 1 :]:
        stripped = line.strip()
        case_match = _CASE_HEADING.fullmatch(stripped)
        if case_match is not None:
            finish_current_case()

            case_number = int(case_match.group("number"))
            if case_number in case_numbers:
                raise GoldDataParseError(f"duplicate case {case_number}")

            case_numbers.append(case_number)
            current_case = case_number
            fields = {}
            current_field = None
            continue

        if stripped.startswith("##"):
            raise GoldDataParseError(f"malformed case heading: {stripped!r}")

        if not stripped:
            continue

        if current_case is None:
            raise GoldDataParseError("content before the first case is not allowed")

        field_match = _FIELD.fullmatch(stripped)
        if field_match is not None:
            field_name = field_match.group("name")
            if field_name in fields:
                raise GoldDataParseError(
                    f"case {current_case} repeats field {field_name!r}"
                )

            value = field_match.group("value").strip()
            if not value:
                raise GoldDataParseError(
                    f"case {current_case} field {field_name!r} is empty"
                )

            fields[field_name] = value
            current_field = field_name
            continue

        if line.startswith((" ", "\t")) and current_field is not None:
            fields[current_field] = f"{fields[current_field]} {stripped}"
            continue

        raise GoldDataParseError(
            f"unrecognized content in case {current_case}: {stripped!r}"
        )

    finish_current_case()
    _validate_case_sequence(case_numbers)
    return tuple(cases)


def _first_non_empty_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip():
            return index
    return None


def _parse_supported(value: str, case_number: int) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise GoldDataParseError(
        f"case {case_number} field 'supported' must be true or false"
    )


def _parse_expected_section(value: str) -> str | None:
    return None if value == "null" else value


def _validate_case_sequence(case_numbers: list[int]) -> None:
    expected = list(range(1, EXPECTED_CASE_COUNT + 1))
    if case_numbers != expected:
        raise GoldDataParseError(
            f"expected cases {expected} in order, received {case_numbers}"
        )
