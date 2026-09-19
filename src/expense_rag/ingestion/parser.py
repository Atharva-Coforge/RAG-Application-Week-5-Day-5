"""Deterministic parser for the structural expense-policy Markdown document."""

from __future__ import annotations

import re
from pathlib import Path

from expense_rag.models import PolicySection

EXPECTED_SECTION_COUNT = 6

_DOCUMENT_HEADING = re.compile(
    r"^#\s+(?P<document>.+?)\s+[—-]\s+Version\s+"
    r"(?P<version>\d+(?:\.\d+)*)\s*$"
)
_SECTION_HEADING = re.compile(
    r"^##\s+(?P<section>[1-9]\d*)\.\s+(?P<title>\S.*?)\s*$"
)


class PolicyParseError(ValueError):
    """Raised when policy Markdown violates the required structure."""


def load_policy(path: Path) -> tuple[PolicySection, ...]:
    """Read and parse a UTF-8 policy Markdown file."""
    return parse_policy(path.read_text(encoding="utf-8"))


def parse_policy(markdown: str) -> tuple[PolicySection, ...]:
    """Parse one numbered Markdown section into each ``PolicySection``."""
    lines = markdown.removeprefix("\ufeff").splitlines()
    heading_index = _first_non_empty_line(lines)
    if heading_index is None:
        raise PolicyParseError("policy document is empty")

    document_match = _DOCUMENT_HEADING.fullmatch(lines[heading_index].strip())
    if document_match is None:
        raise PolicyParseError(
            "first heading must match '# <document> — Version <number>'"
        )

    document = document_match.group("document").strip()
    version = document_match.group("version")
    sections: list[PolicySection] = []
    seen_sections: set[str] = set()
    current_number: str | None = None
    current_title: str | None = None
    body_lines: list[str] = []

    def finish_current_section() -> None:
        if current_number is None or current_title is None:
            return

        text = "\n".join(body_lines).strip()
        if not text:
            raise PolicyParseError(f"section {current_number} has no body text")

        sections.append(
            PolicySection(
                chunk_id=f"expense-policy:v{version}:section-{current_number}",
                document=document,
                version=version,
                section=current_number,
                section_title=current_title,
                text=text,
            )
        )

    for line in lines[heading_index + 1 :]:
        section_match = _SECTION_HEADING.fullmatch(line.strip())
        if section_match is not None:
            finish_current_section()

            section_number = section_match.group("section")
            if section_number in seen_sections:
                raise PolicyParseError(f"duplicate section {section_number}")

            seen_sections.add(section_number)
            current_number = section_number
            current_title = section_match.group("title").strip()
            body_lines = []
            continue

        if line.lstrip().startswith("##"):
            raise PolicyParseError(f"malformed section heading: {line.strip()!r}")

        if current_number is None:
            if line.strip():
                raise PolicyParseError(
                    "content before the first numbered section is not allowed"
                )
            continue

        body_lines.append(line)

    finish_current_section()
    _validate_section_sequence(sections)
    return tuple(sections)


def _first_non_empty_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip():
            return index
    return None


def _validate_section_sequence(sections: list[PolicySection]) -> None:
    expected = [str(number) for number in range(1, EXPECTED_SECTION_COUNT + 1)]
    actual = [section.section for section in sections]

    if actual != expected:
        raise PolicyParseError(
            f"expected sections {expected} in order, received {actual}"
        )
