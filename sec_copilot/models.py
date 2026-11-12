"""Domain models for the SEC filing RAG service.

These are the only shapes that cross component boundaries. Loaders, chunkers,
embedders, vector stores and generators are all swappable precisely because they
speak in terms of these objects and nothing else.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Ingestion-side models
# --------------------------------------------------------------------------- #


class BlockType(str, Enum):
    """What kind of content a parsed block holds.

    The distinction matters: TABLE blocks are never split mid-row and are
    serialised with their header repeated, which is the whole reason financial
    figures survive chunking intact.
    """

    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


@dataclass(frozen=True)
class Table:
    """A parsed tabular region, kept as rows rather than flattened text."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        if self.header:
            return len(self.header)
        return len(self.rows[0]) if self.rows else 0

    def to_markdown(self, rows: Sequence[tuple[str, ...]] | None = None) -> str:
        """Render as a GitHub-flavoured markdown table.

        Markdown is used deliberately: it keeps every cell bound to its column
        label on a single line, so a model reading a balance sheet cannot
        mistake which fiscal year a figure belongs to.
        """
        body = self.rows if rows is None else tuple(rows)
        width = self.n_cols
        header = self.header or tuple(f"col_{i + 1}" for i in range(width))
        header = _pad(header, width)

        lines = [
            "| " + " | ".join(_clean_cell(c) for c in header) + " |",
            "| " + " | ".join("---" for _ in range(width)) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(_clean_cell(c) for c in _pad(row, width)) + " |")

        rendered = "\n".join(lines)
        if self.caption:
            rendered = f"**{self.caption.strip()}**\n\n{rendered}"
        return rendered


@dataclass(frozen=True)
class Block:
    """One layout-atomic region of a source document."""

    type: BlockType
    page: int
    text: str = ""
    table: Table | None = None
    section: str | None = None
    order: int = 0

    def __post_init__(self) -> None:
        if self.type is BlockType.TABLE and self.table is None:
            raise ValueError("TABLE blocks require a Table payload")


@dataclass
class Document:
    """A source filing, plus the metadata analysts filter on."""

    doc_id: str
    source_path: str
    blocks: list[Block] = field(default_factory=list)
    title: str | None = None
    ticker: str | None = None
    company: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    period_end: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_pages(self) -> int:
        return max((b.page for b in self.blocks), default=0)

    def filter_metadata(self) -> dict[str, Any]:
        """The subset of metadata that propagates onto every chunk."""
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "title": self.title,
            "ticker": self.ticker,
            "company": self.company,
            "form_type": self.form_type,
            "fiscal_year": self.fiscal_year,
            "period_end": self.period_end,
            **self.metadata,
        }


# --------------------------------------------------------------------------- #
# Retrieval-side models
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    """An embeddable, citable unit of a filing.

    `page` is carried end to end so the generator can be forced to cite it, and
    `content_type` lets the retriever bias toward tables for numeric questions.
    """

    chunk_id: str
    doc_id: str
    text: str
    page: int
    content_type: BlockType = BlockType.TEXT
    section: str | None = None
    order: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, page: int, order: int, text: str) -> str:
        digest = hashlib.sha1(f"{doc_id}:{page}:{order}:{text}".encode()).hexdigest()[:12]
        return f"{doc_id}:p{page}:c{order}:{digest}"

    @property
    def is_table(self) -> bool:
        return self.content_type is BlockType.TABLE

    def citation_label(self) -> str:
        bits: list[Any] = [self.metadata.get("form_type") or "filing", f"p.{self.page}"]
        if self.section:
            bits.append(self.section)
        return " - ".join(str(b) for b in bits if b)


@dataclass
class ScoredChunk:
    """A chunk with its retrieval provenance attached."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


# --------------------------------------------------------------------------- #
# Generation-side models. These are pydantic because the API enforces them as a
# structured-output schema - a citation-free answer is not representable.
# --------------------------------------------------------------------------- #

#: Longest quote a citation may carry. Long enough for a full table row.
_MAX_SNIPPET = 400


class Citation(BaseModel):
    """A single grounded reference. Every field here is machine-verifiable."""

    @field_validator("snippet", mode="before")
    @classmethod
    def _clamp_snippet(cls, value: Any) -> Any:
        """Truncate an over-long quote rather than rejecting the whole answer.

        Only some providers can enforce `max_length` at generation time -
        Gemini's schema subset drops length keywords - so a model that quotes a
        whole paragraph would otherwise fail validation and discard an
        otherwise correct, well-cited answer. A prefix of a verbatim quote is
        still verbatim, so the citation check is unaffected by the truncation.
        """
        if isinstance(value, str) and len(value) > _MAX_SNIPPET:
            return value[:_MAX_SNIPPET]
        return value

    chunk_id: str = Field(
        description="The exact chunk_id of the source passage, copied from the context header."
    )
    page: int = Field(description="Page number of the filing the snippet appears on.")
    snippet: str = Field(
        description=(
            "A verbatim quote of 5-400 characters copied character-for-character "
            "from the cited chunk. Do not paraphrase, reformat or reflow it."
        ),
        min_length=3,
        max_length=_MAX_SNIPPET,
    )


class Claim(BaseModel):
    """One assertion in the answer, with the evidence that supports it."""

    text: str = Field(description="A single factual statement drawn from the filings.")
    citations: list[Citation] = Field(
        description="At least one citation supporting this exact statement.",
        min_length=1,
    )
    derived_from: list[str] = Field(
        default_factory=list,
        description=(
            "If this claim states a figure you calculated (a growth rate, a margin, "
            "a sum) rather than one printed in the filing, list every input figure "
            "exactly as it appears in the source. Leave empty for figures quoted "
            "directly. A calculated figure whose inputs are not listed here is "
            "rejected by the citation verifier."
        ),
    )


class GroundedAnswer(BaseModel):
    """The only response shape the generator is allowed to emit."""

    answerable: bool = Field(
        description=(
            "False if the provided context does not contain the information needed. "
            "When false, claims must be empty and refusal_reason must be set."
        )
    )
    answer: str = Field(
        default="",
        description="The analyst-facing answer. Empty when answerable is false.",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Every factual assertion in `answer`, decomposed and cited.",
    )
    refusal_reason: str = Field(
        default="",
        description="Why the question cannot be answered from the provided filings.",
    )


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    SNIPPET_NOT_FOUND = "snippet_not_found"
    UNKNOWN_CHUNK = "unknown_chunk"
    PAGE_MISMATCH = "page_mismatch"


@dataclass
class CitationCheck:
    citation: Citation
    status: VerificationStatus
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is VerificationStatus.VERIFIED


class NumericStatus(str, Enum):
    """Why a figure passed or failed grounding."""

    GROUNDED = "grounded"
    DERIVED = "derived"
    ABSENT = "absent"
    #: The figure is real, but belongs to a different period than the claim
    #: states - a column slip, and the most likely genuine error in a filing.
    PERIOD_MISMATCH = "period_mismatch"


@dataclass
class NumericCheck:
    """Result of checking one numeric literal from the answer against context."""

    value: str
    grounded: bool
    detail: str = ""
    status: NumericStatus = NumericStatus.GROUNDED


@dataclass
class VerificationReport:
    """The guardrail verdict attached to every answer the service returns."""

    citation_checks: list[CitationCheck] = field(default_factory=list)
    numeric_checks: list[NumericCheck] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)

    @property
    def failed_citations(self) -> list[CitationCheck]:
        return [c for c in self.citation_checks if not c.ok]

    @property
    def ungrounded_numbers(self) -> list[NumericCheck]:
        return [n for n in self.numeric_checks if not n.grounded]

    @property
    def period_mismatches(self) -> list[NumericCheck]:
        return [
            n for n in self.numeric_checks if n.status is NumericStatus.PERIOD_MISMATCH
        ]

    @property
    def passed(self) -> bool:
        return not (self.failed_citations or self.ungrounded_numbers or self.uncited_claims)

    def summary(self) -> str:
        if self.passed:
            return (
                f"{len(self.citation_checks)} citation(s) verified verbatim; "
                f"{len(self.numeric_checks)} figure(s) traced to source."
            )
        parts = []
        if self.failed_citations:
            parts.append(f"{len(self.failed_citations)} unverifiable citation(s)")
        absent = [n for n in self.ungrounded_numbers if n.status is NumericStatus.ABSENT]
        if absent:
            parts.append("figures absent from context: " + ", ".join(n.value for n in absent[:5]))
        if self.period_mismatches:
            parts.append(
                "figures attributed to the wrong period: "
                + ", ".join(f"{n.value} ({n.detail})" for n in self.period_mismatches[:3])
            )
        if self.uncited_claims:
            parts.append(f"{len(self.uncited_claims)} uncited claim(s)")
        return "; ".join(parts)


@dataclass
class AnswerResult:
    """What the pipeline hands back to the caller."""

    question: str
    answer: GroundedAnswer
    verification: VerificationReport
    retrieved: list[ScoredChunk] = field(default_factory=list)
    blocked: bool = False
    blocked_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answerable": self.answer.answerable and not self.blocked,
            "answer": "" if self.blocked else self.answer.answer,
            "refusal_reason": self.blocked_reason or self.answer.refusal_reason,
            "claims": [
                {
                    "text": c.text,
                    "citations": [
                        {
                            "chunk_id": ct.chunk_id,
                            "page": ct.page,
                            "snippet": ct.snippet,
                            "label": self._label_for(ct.chunk_id),
                        }
                        for ct in c.citations
                    ],
                }
                for c in self.answer.claims
            ],
            "verification": {
                "passed": self.verification.passed,
                "summary": self.verification.summary(),
                "citation_checks": [
                    {
                        "chunk_id": c.citation.chunk_id,
                        "status": c.status.value,
                        "detail": c.detail,
                    }
                    for c in self.verification.citation_checks
                ],
                "ungrounded_numbers": [
                    {"value": n.value, "status": n.status.value, "detail": n.detail}
                    for n in self.verification.ungrounded_numbers
                ],
                "uncited_claims": self.verification.uncited_claims,
            },
            "sources": [
                {
                    "chunk_id": sc.chunk_id,
                    "page": sc.chunk.page,
                    "score": round(sc.score, 5),
                    "content_type": sc.chunk.content_type.value,
                    "section": sc.chunk.section,
                    "label": sc.chunk.citation_label(),
                }
                for sc in self.retrieved
            ],
            "usage": self.usage,
        }

    def _label_for(self, chunk_id: str) -> str:
        for sc in self.retrieved:
            if sc.chunk_id == chunk_id:
                return sc.chunk.citation_label()
        return chunk_id


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")


def _clean_cell(value: Any) -> str:
    """Flatten a cell to one line, escaping the markdown column delimiter."""
    if value is None:
        return ""
    return _WS.sub(" ", str(value)).replace("|", "\\|").strip()


def _pad(row: Iterable[Any], width: int) -> tuple[Any, ...]:
    items = list(row)
    if len(items) < width:
        items.extend([""] * (width - len(items)))
    return tuple(items[:width])
