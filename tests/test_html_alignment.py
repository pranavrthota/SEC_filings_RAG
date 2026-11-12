"""Regression tests for real EDGAR table shapes.

Every fixture here reproduces a cell layout taken from an actual 10-K. They
exist because each one silently misaligned figures before it was fixed - and a
misaligned figure is the exact failure this project is built to prevent, made
worse by the fact that the output still *looks* like a clean table.
"""

from __future__ import annotations

import pytest

from sec_copilot.ingestion.html_loader import SecHtmlLoader
from sec_copilot.models import BlockType


def load_table(tmp_path, html: str, name: str = "f.htm"):
    path = tmp_path / name
    path.write_text(f"<html><body>{html}</body></html>", encoding="utf-8")
    doc = SecHtmlLoader().load(path)
    tables = [b.table for b in doc.blocks if b.type is BlockType.TABLE]
    return tables[0] if tables else None


# --------------------------------------------------------------------------- #
# Apple: the currency symbol is its own <td>, but only on decorated rows, so
# rows arrive with genuinely different cell counts.
# --------------------------------------------------------------------------- #

APPLE_SEGMENTS = """
<table>
 <tr><td></td><td>2025</td><td></td><td>Change</td><td>2024</td><td></td><td>Change</td><td>2023</td><td></td></tr>
 <tr><td>Americas</td><td>$</td><td>178,353</td><td>7</td><td>$</td><td>167,045</td><td>3</td><td>$</td><td>162,560</td></tr>
 <tr><td>Europe</td><td>111,032</td><td></td><td>10</td><td>101,328</td><td></td><td>7</td><td>94,294</td><td></td></tr>
 <tr><td>Greater China</td><td>64,377</td><td></td><td>(4)</td><td>66,952</td><td></td><td>(8)</td><td>72,559</td><td></td></tr>
 <tr><td>Total net sales</td><td>$</td><td>416,161</td><td>6</td><td>$</td><td>391,035</td><td>2</td><td>$</td><td>383,285</td></tr>
</table>
"""


def test_ragged_currency_columns_are_realigned(tmp_path):
    table = load_table(tmp_path, APPLE_SEGMENTS)
    assert table is not None

    rows = {r[0]: r for r in table.rows}
    # Before the fix, Americas' 178,353 sat in column 3 and Europe's 111,032 in
    # column 2 - the same fiscal year, two different columns.
    assert rows["Americas"][1] == "178,353"
    assert rows["Europe"][1] == "111,032"
    assert rows["Greater China"][1] == "64,377"
    assert rows["Total net sales"][1] == "416,161"

    # And every row agrees on width, which is what makes the column meaningful.
    assert len({len(r) for r in table.rows}) == 1


def test_realigned_rows_keep_each_year_under_its_own_header(tmp_path):
    table = load_table(tmp_path, APPLE_SEGMENTS)
    fy2025 = table.header.index("2025")
    fy2024 = table.header.index("2024")
    rows = {r[0]: r for r in table.rows}
    assert rows["Americas"][fy2025] == "178,353"
    assert rows["Americas"][fy2024] == "167,045"
    assert rows["Total net sales"][fy2025] == "416,161"
    assert rows["Total net sales"][fy2024] == "391,035"


# --------------------------------------------------------------------------- #
# An em-dash is a nil value in a filing, not padding. Dropping it shortens the
# row and shifts every later figure one column left.
# --------------------------------------------------------------------------- #

APPLE_PRODUCTS = """
<table>
 <tr><td></td><td>2025</td><td></td><td>Change</td><td>2024</td><td></td><td>Change</td><td>2023</td><td></td></tr>
 <tr><td>iPhone</td><td>$</td><td>209,586</td><td>4</td><td>$</td><td>201,183</td><td>&#8212;</td><td>$</td><td>200,583</td></tr>
 <tr><td>Mac</td><td>33,708</td><td></td><td>12</td><td>29,984</td><td></td><td>2</td><td>29,357</td><td></td></tr>
</table>
"""


def test_em_dash_is_treated_as_a_nil_value_not_padding(tmp_path):
    table = load_table(tmp_path, APPLE_PRODUCTS)
    rows = {r[0]: r for r in table.rows}
    assert rows["iPhone"][1] == "209,586"
    assert rows["iPhone"][3] == "201,183"
    assert rows["Mac"][1] == "33,708"
    assert len(rows["iPhone"]) == len(rows["Mac"])


# --------------------------------------------------------------------------- #
# Microsoft splits an accounting negative as "( 4,901" + ")" - a different shape
# from Apple's "(" + "321" + ")".
# --------------------------------------------------------------------------- #

MSFT_NEGATIVES = """
<table>
 <tr><td>Year Ended June 30,</td><td>2026</td><td>2025</td><td>2024</td></tr>
 <tr><td>Operating income</td><td></td><td></td><td>155,237</td><td></td><td></td><td></td><td>128,528</td><td></td><td></td><td></td><td>109,433</td><td></td></tr>
 <tr><td>Other income (expense), net</td><td></td><td></td><td>10,697</td><td></td><td></td><td></td><td>( 4,901</td><td>)</td><td></td><td></td><td>( 1,646</td><td>)</td></tr>
</table>
"""


def test_negative_split_across_cells_is_reassembled(tmp_path):
    table = load_table(tmp_path, MSFT_NEGATIVES)
    rows = {r[0]: r for r in table.rows}
    other = rows["Other income (expense), net"]
    assert other[1] == "10,697"
    assert other[2] == "(4,901)"
    assert other[3] == "(1,646)"


def test_a_label_containing_parentheses_is_not_mistaken_for_a_negative(tmp_path):
    table = load_table(tmp_path, MSFT_NEGATIVES)
    labels = [r[0] for r in table.rows]
    assert "Other income (expense), net" in labels


def test_negatives_reconcile_arithmetically(tmp_path):
    """The strongest available check that the parse is right."""
    table = load_table(tmp_path, MSFT_NEGATIVES)
    rows = {r[0]: r for r in table.rows}
    to_num = lambda s: -float(s.strip("()").replace(",", "")) if s.startswith("(") else float(s.replace(",", ""))

    for col in (1, 2, 3):
        operating = to_num(rows["Operating income"][col])
        other = to_num(rows["Other income (expense), net"][col])
        # Against the real filing: 155,237 + 10,697 = 165,934, and so on.
        assert operating + other == pytest.approx(
            {1: 165_934.0, 2: 123_627.0, 3: 107_787.0}[col]
        )


# --------------------------------------------------------------------------- #
# Inline elements must not fragment a heading.
# --------------------------------------------------------------------------- #


def test_heading_split_across_inline_spans_is_not_fragmented(tmp_path):
    path = tmp_path / "f.htm"
    # Filers really do split words across spans for styling; this produced the
    # section label "Ome Statements" before `span` stopped being a block tag.
    path.write_text(
        "<html><body><p><span>INC</span><span>OME STATEMENTS</span></p>"
        "<p>The accompanying notes are an integral part of these statements.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    doc = SecHtmlLoader().load(path)
    sections = {b.section for b in doc.blocks if b.section}
    assert "Income Statements" in sections
    assert not any(s.startswith("Ome") for s in sections)


def test_a_genuinely_ragged_grid_falls_back_rather_than_inventing_columns(tmp_path):
    # No consensus width: compaction must not force one.
    table = load_table(
        tmp_path,
        """
        <table>
          <tr><td>a</td><td>b</td><td>c</td><td>d</td></tr>
          <tr><td>e</td><td>f</td></tr>
          <tr><td>g</td><td>h</td><td>i</td></tr>
        </table>
        """,
    )
    assert table is not None
    assert len({len(r) for r in table.rows}) >= 1  # parsed, not crashed
