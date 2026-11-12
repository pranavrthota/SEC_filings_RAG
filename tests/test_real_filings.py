"""Integration tests against real 10-K filings from EDGAR.

Skipped unless the filings are present locally - run `python -m evals.fetch_filings`
to download them. These are the tests that would have caught every defect the
synthetic sample missed: ragged currency columns, split negatives, and headings
fragmented across inline spans.

The assertions are pinned to figures published in the filings themselves, and
several of them reconcile arithmetically, so a silent misalignment fails here
even if the output still looks like a well-formed table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.ingestion.html_loader import SecHtmlLoader
from sec_copilot.models import BlockType

FILINGS = Path(__file__).resolve().parents[1] / "filings"
AAPL = FILINGS / "aapl-10k-2025.htm"
MSFT = FILINGS / "msft-10k-2026.htm"

pytestmark = pytest.mark.skipif(
    not (AAPL.exists() and MSFT.exists()),
    reason="real filings not downloaded - run: python -m evals.fetch_filings",
)


def _to_num(value: str) -> float:
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        return -float(value[1:-1].replace(",", ""))
    return float(value.replace(",", ""))


def _find_table(doc, *required: str):
    for block in doc.blocks:
        if block.type is not BlockType.TABLE:
            continue
        markdown = block.table.to_markdown()
        if all(r in markdown for r in required):
            return block
    raise AssertionError(f"no table containing {required}")


@pytest.fixture(scope="module")
def apple():
    return SecHtmlLoader().load(AAPL, company="Apple Inc.", ticker="AAPL", form_type="10-K", fiscal_year=2025)


@pytest.fixture(scope="module")
def microsoft():
    return SecHtmlLoader().load(MSFT, company="Microsoft Corporation", ticker="MSFT", form_type="10-K", fiscal_year=2026)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_a_real_filing_parses_into_pages_sections_and_tables(apple):
    assert apple.n_pages > 100
    assert len([b for b in apple.blocks if b.type is BlockType.TABLE]) > 40
    sections = {b.section for b in apple.blocks if b.section}
    assert "Consolidated Statements Of Operations" in sections
    assert any(s.startswith("Item 1A.") for s in sections)
    assert any(s.startswith("Item 7.") for s in sections)


def test_section_labels_are_not_fragmented_by_inline_markup(microsoft):
    sections = {b.section for b in microsoft.blocks if b.section}
    # "INCOME STATEMENTS" is split across <span>s in Microsoft's filing and used
    # to arrive as "Ome Statements".
    assert "Income Statements" in sections
    assert not any(s.startswith("Ome ") for s in sections)


# --------------------------------------------------------------------------- #
# Figures - pinned to the published statements
# --------------------------------------------------------------------------- #


def test_apple_income_statement_figures_sit_under_the_right_year(apple):
    table = _find_table(apple, "Total net sales", "Operating income").table
    rows = {r[0]: r for r in table.rows}

    assert rows["Total net sales"][1:4] == ("416,161", "391,035", "383,285")
    assert rows["Operating income"][1:4] == ("133,050", "123,216", "114,301")
    assert rows["Net income"][1:4] == ("112,010", "93,736", "96,995")


def test_apple_income_statement_reconciles(apple):
    table = _find_table(apple, "Total net sales", "Operating income").table
    rows = {r[0]: r for r in table.rows}

    for col in (1, 2, 3):
        gross = _to_num(rows["Gross margin"][col])
        opex = _to_num(rows["Total operating expenses"][col])
        operating = _to_num(rows["Operating income"][col])
        assert gross - opex == pytest.approx(operating)


def test_apple_balance_sheet_reconciles(apple):
    table = _find_table(apple, "Total assets", "Total current liabilities").table
    rows = {r[0]: r for r in table.rows}

    for col in (1, 2):
        current = _to_num(rows["Total current assets"][col])
        non_current = _to_num(rows["Total non-current assets"][col])
        assert current + non_current == pytest.approx(_to_num(rows["Total assets"][col]))


def test_apple_ragged_currency_columns_are_realigned(apple):
    # The segment table decorates only its first and total rows with "$".
    table = _find_table(apple, "Greater China", "Total net sales").table
    rows = {r[0]: r for r in table.rows}
    assert rows["Americas"][1] == "178,353"
    assert rows["Europe"][1] == "111,032"
    assert rows["Greater China"][1] == "64,377"
    assert len({len(r) for r in table.rows}) == 1


def test_microsoft_negatives_split_across_cells_are_reassembled(microsoft):
    table = _find_table(microsoft, "Total revenue", "Operating income").table
    rows = {r[0]: r for r in table.rows}

    assert rows["Other income (expense), net"][1:4] == ("10,697", "(4,901)", "(1,646)")
    for col in (1, 2, 3):
        operating = _to_num(rows["Operating income"][col])
        other = _to_num(rows["Other income (expense), net"][col])
        pretax = _to_num(rows["Income before income taxes"][col])
        assert operating + other == pytest.approx(pretax)


def test_microsoft_income_statement_figures(microsoft):
    table = _find_table(microsoft, "Total revenue", "Operating income").table
    rows = {r[0]: r for r in table.rows}
    assert rows["Total revenue"][1:4] == ("331,839", "281,724", "245,122")
    assert rows["Net income"][1:4] == ("133,749", "101,832", "88,136")


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_a_real_filing_survives_chunking_with_its_statements_intact(apple):
    from sec_copilot.chunking.table_aware import TableAwareChunker

    chunks = TableAwareChunker().chunk(apple)
    assert len(chunks) > 100

    income = [c for c in chunks if "| Operating income | 133,050 |" in c.text]
    assert len(income) == 1, "the income statement was split or duplicated"
    assert income[0].is_table
    assert income[0].page > 1
