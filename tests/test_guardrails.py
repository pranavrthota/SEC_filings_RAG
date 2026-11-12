"""Tests for the citation and numeric-grounding guardrails.

These are the tests that matter most: they assert that a fabricated citation or
an invented figure is *caught*, not merely discouraged by the prompt.
"""

from __future__ import annotations

import pytest

from sec_copilot.config import GuardrailConfig
from sec_copilot.generation.verifier import CitationVerifier, normalize_number
from sec_copilot.models import (
    BlockType,
    Chunk,
    Citation,
    Claim,
    GroundedAnswer,
    ScoredChunk,
    VerificationStatus,
)

TABLE_TEXT = (
    "[Acme Industries, Inc. | 10-K | Item 7. MD&A]\n"
    "**CONSOLIDATED STATEMENTS OF OPERATIONS (in millions)**\n\n"
    "| Line item | FY2023 | FY2022 |\n"
    "| --- | --- |\n"
    "| Net sales | 18,742 | 17,109 |\n"
    "| Operating income | 2,879 | 2,113 |\n"
)


@pytest.fixture
def sources() -> list[ScoredChunk]:
    chunk = Chunk(
        chunk_id="acme:p4:c1:abc123",
        doc_id="acme",
        text=TABLE_TEXT,
        page=4,
        content_type=BlockType.TABLE,
        section="Item 7. MD&A",
    )
    return [ScoredChunk(chunk=chunk, score=1.0)]


@pytest.fixture
def verifier() -> CitationVerifier:
    return CitationVerifier(GuardrailConfig())


def _answer(text: str, snippet: str, *, chunk_id: str = "acme:p4:c1:abc123", page: int = 4, derived=()):
    return GroundedAnswer(
        answerable=True,
        answer=text,
        claims=[
            Claim(
                text=text,
                citations=[Citation(chunk_id=chunk_id, page=page, snippet=snippet)],
                derived_from=list(derived),
            )
        ],
    )


def test_verbatim_citation_passes(verifier, sources):
    report = verifier.verify(
        _answer(
            "Operating income was 2,879 million in FY2023.",
            "| Operating income | 2,879 | 2,113 |",
        ),
        sources,
    )
    assert report.passed, report.summary()
    assert report.citation_checks[0].status is VerificationStatus.VERIFIED


def test_fabricated_snippet_is_rejected(verifier, sources):
    report = verifier.verify(
        _answer(
            "Operating income was 2,879 million in FY2023.",
            "Operating income for the year totalled 2,879, an increase of 36%.",
        ),
        sources,
    )
    assert not report.passed
    assert report.citation_checks[0].status is VerificationStatus.SNIPPET_NOT_FOUND
    assert report.uncited_claims  # the claim has no surviving citation


def test_citation_to_a_chunk_that_was_never_retrieved_is_rejected(verifier, sources):
    report = verifier.verify(
        _answer(
            "Operating income was 2,879 million.",
            "| Operating income | 2,879 | 2,113 |",
            chunk_id="acme:p99:c1:deadbeef",
        ),
        sources,
    )
    assert report.citation_checks[0].status is VerificationStatus.UNKNOWN_CHUNK


def test_wrong_page_is_rejected(verifier, sources):
    report = verifier.verify(
        _answer("Operating income was 2,879 million.", "| Operating income | 2,879 | 2,113 |", page=7),
        sources,
    )
    assert report.citation_checks[0].status is VerificationStatus.PAGE_MISMATCH


def test_rewrapped_quote_still_verifies(verifier, sources):
    # Whitespace differences are a formatting artefact, not a fabrication.
    report = verifier.verify(
        _answer(
            "Operating income was 2,879 million.",
            "|  Operating   income |  2,879 |  2,113 |",
        ),
        sources,
    )
    assert report.passed, report.summary()


def test_invented_figure_is_caught_even_with_a_valid_citation(verifier, sources):
    # The citation is real; the number in the prose is not. This is the exact
    # failure a citation-only guardrail misses.
    report = verifier.verify(
        _answer(
            "Operating income was 3,450 million in FY2023.",
            "| Operating income | 2,879 | 2,113 |",
        ),
        sources,
    )
    assert report.citation_checks[0].ok
    assert not report.passed
    assert "3,450" in [n.value for n in report.ungrounded_numbers]


def test_calculated_figure_is_allowed_when_its_inputs_are_grounded(verifier, sources):
    report = verifier.verify(
        _answer(
            "Operating margin was 15.4% in FY2023 (2,879 / 18,742).",
            "| Operating income | 2,879 | 2,113 |",
            derived=("2,879", "18,742"),
        ),
        sources,
    )
    assert report.passed, report.summary()
    assert any("calculated from grounded inputs" in n.detail for n in report.numeric_checks)


def test_calculated_figure_with_ungrounded_inputs_is_rejected(verifier, sources):
    report = verifier.verify(
        _answer(
            "Operating margin was 22.5% in FY2023 (2,879 / 12,800).",
            "| Operating income | 2,879 | 2,113 |",
            derived=("2,879", "12,800"),
        ),
        sources,
    )
    assert not report.passed
    assert any("ungrounded" in n.detail for n in report.ungrounded_numbers)


def test_small_numbers_do_not_trip_the_guardrail(verifier, sources):
    report = verifier.verify(
        _answer(
            "The company reports three segments and 12 product lines.",
            "| Operating income | 2,879 | 2,113 |",
        ),
        sources,
    )
    assert report.passed, report.summary()


def test_refusal_needs_no_evidence(verifier, sources):
    report = verifier.verify(
        GroundedAnswer(answerable=False, refusal_reason="Segment data is not in these excerpts."),
        sources,
    )
    assert report.passed
    assert not report.citation_checks


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234", "1234"),
        ("(2,150)", "2150"),
        ("45.70%", "45.7"),
        ("$3.20", "3.2"),
        ("18,742", "18742"),
    ],
)
def test_number_normalization(raw, expected):
    assert normalize_number(raw) == expected
