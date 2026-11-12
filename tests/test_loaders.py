"""Loader tests, focused on table fidelity and page attribution."""

from __future__ import annotations

import pytest

from sec_copilot.ingestion.base import LoaderRegistry
from sec_copilot.ingestion.html_loader import SecHtmlLoader
from sec_copilot.ingestion.text_loader import TextLoader
from sec_copilot.models import BlockType

EDGAR_HTML = """
<html><body>
<p>Item 7. Management's Discussion and Analysis of Financial Condition</p>
<p>Consolidated net sales increased in fiscal 2023 driven by price realization
across the Automation Systems segment and continued subscription growth.</p>
<table>
  <tr><td>Line item</td><td></td><td>FY2023</td><td></td><td>FY2022</td></tr>
  <tr><td>Net sales</td><td>$</td><td>18,742</td><td>$</td><td>17,109</td></tr>
  <tr><td>Operating income</td><td>$</td><td>2,879</td><td>$</td><td>2,113</td></tr>
</table>
<div style="page-break-after: always"></div>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>The accompanying notes are an integral part of these consolidated financial
statements and should be read together with Item 7 above.</p>
</body></html>
"""


@pytest.fixture
def html_doc(tmp_path):
    path = tmp_path / "acme-10k.htm"
    path.write_text(EDGAR_HTML, encoding="utf-8")
    return SecHtmlLoader().load(path, ticker="ACME", form_type="10-K")


def test_html_table_becomes_a_structured_table(html_doc):
    tables = [b for b in html_doc.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 1

    table = tables[0].table
    # The spacer and "$" columns EDGAR pads statements with are dropped, so the
    # figure columns line up with their fiscal-year headers.
    assert table.header == ("Line item", "FY2023", "FY2022")
    assert ("Net sales", "18,742", "17,109") in table.rows
    assert ("Operating income", "2,879", "2,113") in table.rows


def test_html_table_text_is_not_also_emitted_as_prose(html_doc):
    prose = " ".join(b.text for b in html_doc.blocks if b.type is not BlockType.TABLE)
    assert "18,742" not in prose


def test_html_page_breaks_advance_the_page_number(html_doc):
    by_section = {b.section: b.page for b in html_doc.blocks if b.section}
    assert by_section["Item 7. Management's Discussion and Analysis of Financial Condition"] == 1
    assert by_section["Item 8. Financial Statements and Supplementary Data"] == 2


def test_html_sections_are_detected_from_item_headings(html_doc):
    sections = {b.section for b in html_doc.blocks if b.section}
    assert any(s.startswith("Item 7.") for s in sections)
    assert any(s.startswith("Item 8.") for s in sections)


def test_text_loader_honours_explicit_page_numbers(tmp_path):
    path = tmp_path / "filing.md"
    path.write_text(
        "Cover page text that is long enough to be kept as a paragraph block.\n"
        "\n[page 7]\n\n"
        "Item 1A. Risk Factors\n\n"
        "Demand for the products is cyclical and correlated with capital budgets.\n",
        encoding="utf-8",
    )
    doc = TextLoader().load(path)
    risk = next(b for b in doc.blocks if "cyclical" in b.text)
    assert risk.page == 7
    assert risk.section == "Item 1A. Risk Factors"


def test_registry_dispatches_on_suffix_and_rejects_the_unknown(tmp_path):
    registry = LoaderRegistry.default()
    md = tmp_path / "a.md"
    md.write_text("hello", encoding="utf-8")
    assert isinstance(registry.loader_for(md), TextLoader)
    assert isinstance(registry.loader_for(tmp_path / "b.htm"), SecHtmlLoader)
    with pytest.raises(ValueError):
        registry.loader_for(tmp_path / "c.xlsx")


def test_a_custom_loader_can_override_a_built_in(tmp_path):
    class MyLoader(TextLoader):
        pass

    registry = LoaderRegistry.default().register(MyLoader())
    md = tmp_path / "a.md"
    md.write_text("hello", encoding="utf-8")
    assert type(registry.loader_for(md)) is MyLoader
