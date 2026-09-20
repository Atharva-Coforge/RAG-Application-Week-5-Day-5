"""Provider-independent domain models for the expense-policy RAG application."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

REFUSAL_ANSWER = "The provided policy does not answer this question."
EmbeddingValue = Annotated[float, Field(allow_inf_nan=False)]


class DomainModel(BaseModel):
    """Base configuration shared by all domain models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class PolicySection(DomainModel):
    """A parsed policy section before embedding."""

    chunk_id: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9._:-]*$",
    )
    document: str = Field(min_length=1)
    version: str = Field(
        min_length=1,
        pattern=r"^\d+(?:\.\d+)*$",
    )
    section: str = Field(
        min_length=1,
        pattern=r"^[1-9]\d*$",
    )
    section_title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class PolicyChunk(PolicySection):
    """A complete embedded policy section ready for vector storage."""

    embedding: tuple[EmbeddingValue, ...] = Field(min_length=1)


class IngestionSummary(DomainModel):
    """Minimal record of one successful policy ingestion."""

    chunk_count: int = Field(ge=6, le=6)
    chunk_ids: tuple[str, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        """Keep the reported count aligned with unique persisted IDs."""
        if self.chunk_count != len(self.chunk_ids):
            raise ValueError("chunk_count must match the number of chunk_ids")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("chunk_ids must be unique")
        return self


class SearchResult(DomainModel):
    """A retrieved chunk and its canonical cosine distance."""

    chunk: PolicyChunk = Field(exclude=True)
    distance: float = Field(ge=0.0, le=2.0, allow_inf_nan=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def section(self) -> str:
        """Return the public section label required by the response contract."""
        return f"{self.chunk.section}. {self.chunk.section_title}"


class GenerationDecision(DomainModel):
    """Private structured model output before the public response is built."""

    supported: bool
    answer: str = Field(min_length=1)
    cited_chunk_id: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9._:-]*$",
    )

    @model_validator(mode="after")
    def validate_supported_citation(self) -> Self:
        """Require a chunk ID whenever the model claims the question is supported."""
        if self.supported and self.cited_chunk_id is None:
            raise ValueError("supported output must cite a chunk_id")
        return self


class Citation(DomainModel):
    """A citation in the application's public response format."""

    document: str = Field(min_length=1)
    version: str = Field(
        min_length=1,
        pattern=r"^\d+(?:\.\d+)*$",
    )
    section: str = Field(
        min_length=1,
        pattern=r"^[1-9]\d*\.\s+\S.*$",
    )

    @classmethod
    def from_chunk(cls, chunk: PolicyChunk) -> Self:
        """Create a citation from a retrieved policy chunk."""
        return cls(
            document=chunk.document,
            version=chunk.version,
            section=f"{chunk.section}. {chunk.section_title}",
        )


class RagResponse(DomainModel):
    """The structured answer returned by the application."""

    answer: str = Field(min_length=1)
    citation: Citation | None
    retrieved_chunks: tuple[SearchResult, ...] = Field(max_length=3)

    @model_validator(mode="after")
    def validate_grounding_contract(self) -> Self:
        """Enforce ordering, citation, and refusal invariants."""
        distances = [result.distance for result in self.retrieved_chunks]
        if distances != sorted(distances):
            raise ValueError("retrieved chunks must be sorted by ascending distance")

        chunk_ids = [result.chunk.chunk_id for result in self.retrieved_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("retrieved chunks must not contain duplicates")

        if self.answer == REFUSAL_ANSWER:
            if self.citation is not None:
                raise ValueError("unsupported answers must not include a citation")
            return self

        if self.citation is None:
            raise ValueError("supported answers must include a citation")

        valid_citations = {
            (
                result.chunk.document,
                result.chunk.version,
                result.section,
            )
            for result in self.retrieved_chunks
        }
        citation_key = (
            self.citation.document,
            self.citation.version,
            self.citation.section,
        )
        if citation_key not in valid_citations:
            raise ValueError("citation must refer to a retrieved chunk")

        return self


class GoldCase(DomainModel):
    """One expected question and answer used by the evaluation runner."""

    question: str = Field(min_length=1)
    supported: bool
    required_answer: str = Field(min_length=1)
    expected_section: str | None = Field(
        default=None,
        pattern=r"^[1-9]\d*$",
    )

    @model_validator(mode="after")
    def validate_expected_answer(self) -> Self:
        """Keep support labels, answers, and expected sections consistent."""
        if self.supported and self.required_answer == REFUSAL_ANSWER:
            raise ValueError("a supported case cannot require the refusal answer")
        if self.supported and self.expected_section is None:
            raise ValueError("a supported case must define an expected section")
        if not self.supported and self.required_answer != REFUSAL_ANSWER:
            raise ValueError(
                "an unsupported case must require the canonical refusal answer"
            )
        if not self.supported and self.expected_section is not None:
            raise ValueError("an unsupported case must not define an expected section")
        return self
