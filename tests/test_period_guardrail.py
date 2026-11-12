"""Period-attribution and scale guardrails.

The failure these exist for is subtle and specific: the model quotes a figure
that is genuinely in the filing, cites a real passage verbatim, and attaches it
to the wrong fiscal year. Every text-level check passes, because in a three-year
statement every year's figure is "in the context". Only a cell-level check -
row label plus dated column - can tell a correct answer from a column slip.
"""

from __future__ import annotations

import pytest

from sec_copilot.config import GuardrailConfig
from sec_copilot.generation.periods import (
    detect_scale,
    parse_number,
    parse_table_facts,
    stated_scale,
    values_match,
)
from sec_copilot.generation.verifier import CitationVerifier
from sec_copilot.models import (
    BlockType,
    Chunk,
    Citation,
    Claim,
    GroundedAnswer,
    NumericStatus,
    ScoredChunk,
)

INCOME_STATEMENT = (
    "[Apple Inc. | 10-K | Consolidated Statements Of Operations]\n"
    "**CONSOLIDATED STATEMENTS OF OPERATIONS (in millions, except per share amounts)**\n\n"
    "|  | September 27, 2025 | September 28, 2024 | September 30, 2023 |\n"
    "| --- | --- | --- | --- |\n"
    "| Total net sales | 416,161 | 391,035 | 383,285 |\n"
    "| Operating income | 133,050 | 123,216 | 114,301 |\n"
    "| Net income | 112,010 | 93,736 | 96,995 |\n"
)
SNIPPET = "| Operating income | 133,050 | 123,216 | 114,301 |"


@pytest.fixture
def sources():
    chunk = Chunk("c1", "aapl", INCOME_STATEMENT, 61, BlockType.TABLE, "Statements Of Operations")
    return [ScoredChunk(chunk=chunk, score=1.0)]


@pytest.fixture
def verifier():
    return CitationVerifier(GuardrailConfig())


def answer(text: str, *, derived: tuple[str, ...] = (), snippet: str = SNIPPET) -> GroundedAnswer:
    return GroundedAnswer(
        answerable=True,
        answer=text,
        claims=[
            Claim(
                text=text,
                citations=[Citation(chunk_id="c1", page=61, snippet=snippet)],
                derived_from=list(derived),
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Period attribution
# --------------------------------------------------------------------------- #


def test_correct_figure_for_the_stated_year_passes(verifier, sources):
    report = verifier.verify(answer("Operating income was 133,050 million in fiscal 2025."), sources)
    assert report.passed, report.summary()


def test_a_real_figure_from_the_wrong_year_is_caught(verifier, sources):
    # 123,216 is genuinely in the filing - it is FY2024's operating income.
    # Every text-level check passes; only the column tells the truth.
    report = verifier.verify(answer("Operating income was 123,216 million in fiscal 2025."), sources)

    assert not report.passed
    assert report.citation_checks[0].ok, "the citation itself is legitimate"
    (mismatch,) = report.period_mismatches
    assert mismatch.status is NumericStatus.PERIOD_MISMATCH
    assert "2024" in mismatch.detail and "2025" in mismatch.detail
    assert "Operating income" in mismatch.detail


def test_a_prior_year_figure_correctly_labelled_is_not_flagged(verifier, sources):
    report = verifier.verify(answer("Operating income was 123,216 million in fiscal 2024."), sources)
    assert report.passed, report.summary()


def test_a_multi_year_comparison_passes_when_both_years_are_right(verifier, sources):
    report = verifier.verify(
        answer(
            "Operating income rose to 133,050 million in fiscal 2025 from 123,216 million in fiscal 2024."
        ),
        sources,
    )
    assert report.passed, report.summary()


def test_a_claim_with_no_year_is_not_penalised(verifier, sources):
    # Nothing to contradict: the guardrail must not invent an attribution.
    report = verifier.verify(answer("Operating income was 133,050 million."), sources)
    assert report.passed, report.summary()


def test_a_fabricated_figure_is_still_absent_not_a_period_mismatch(verifier, sources):
    report = verifier.verify(answer("Operating income was 999,999 million in fiscal 2025."), sources)
    assert not report.passed
    assert report.ungrounded_numbers[0].status is NumericStatus.ABSENT


def test_the_year_itself_is_not_treated_as_a_figure(verifier, sources):
    report = verifier.verify(answer("In fiscal 2025 operating income was 133,050 million."), sources)
    values = [c.value for c in report.numeric_checks]
    assert "2025" not in values


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #


def test_a_correct_figure_restated_in_billions_is_accepted(verifier, sources):
    # 133,050 in a table headed "(in millions)" IS $133.05 billion. Rejecting
    # this was over-blocking correct analyst work.
    report = verifier.verify(answer("Operating income was $133.05 billion in fiscal 2025."), sources)
    assert report.passed, report.summary()


def test_a_wrong_figure_in_billions_is_still_rejected(verifier, sources):
    report = verifier.verify(answer("Operating income was $145.00 billion in fiscal 2025."), sources)
    assert not report.passed


def test_scale_conversion_does_not_defeat_the_period_check(verifier, sources):
    report = verifier.verify(answer("Operating income was $123.216 billion in fiscal 2025."), sources)
    assert report.period_mismatches


def test_rounding_within_tolerance_is_accepted(verifier, sources):
    report = verifier.verify(answer("Net income was 112,010 million in fiscal 2025."), sources)
    assert report.passed, report.summary()


# --------------------------------------------------------------------------- #
# The parsing primitives
# --------------------------------------------------------------------------- #


def test_table_facts_bind_each_figure_to_its_row_and_year():
    facts = parse_table_facts(INCOME_STATEMENT)
    assert facts.scale == 1e6

    cell = next(
        c for c in facts.cells if c.row_label == "Operating income" and 2025 in c.years
    )
    assert cell.value == 133_050
    assert cell.absolute == pytest.approx(1.3305e11)
    assert facts.years() == {2023, 2024, 2025}


def test_a_chunk_with_no_table_yields_no_cells():
    assert parse_table_facts("Just prose about revenue of 1,234 million.").cells == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234", 1234.0), ("(2,150)", -2150.0), ("$3.20", 3.2), ("45.7%", 45.7), ("—", None)],
)
def test_number_parsing(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize(
    ("text", "factor"),
    [
        ("(in millions)", 1e6),
        ("In thousands, except per share", 1e3),
        ("amounts in billions", 1e9),
        ("no scale here", 1.0),
    ],
)
def test_scale_detection(text, factor):
    assert detect_scale(text) == factor


def test_scale_stated_alongside_a_figure():
    assert stated_scale(" billion in fiscal 2025") == 1e9
    assert stated_scale(" million") == 1e6
    assert stated_scale(" in fiscal 2025") == 1.0


def test_value_comparison_tolerates_rounding_but_not_error():
    assert values_match(133_050e6, 133.05e9)
    assert values_match(1000.0, 1004.0, tolerance=0.005)
    assert not values_match(1000.0, 1200.0)
