"""Harder PDF cases.

`test_pdf_loader.py` covers a ruled table, which pdfplumber's line strategy
finds easily. Real filings also print statements with no borders at all, laid
out by whitespace alone, and split them across pages. Those exercise the
loader's fallback detector, and they are where a naive extractor emits a flat
run of digits with no line item attached.
"""

from __future__ import annotations

import pytest

pdfplumber = pytest.importorskip("pdfplumber")
reportlab = pytest.importorskip("reportlab")

from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from sec_copilot.chunking.table_aware import TableAwareChunker  # noqa: E402
from sec_copilot.ingestion.pdf_loader import PdfPlumberLoader  # noqa: E402
from sec_copilot.models import BlockType  # noqa: E402

# A statement with no rules at all - columns held only by monospaced alignment.
UNRULED = [
    "Line item                          FY2025      FY2024      FY2023",
    "Net sales                         416,161     391,035     383,285",
    "Cost of sales                     220,960     210,352     214,137",
    "Gross margin                      195,201     180,683     169,148",
    "Operating income                  133,050     123,216     114,301",
    "Net income                        112,010      93,736      96,995",
]


@pytest.fixture(scope="module")
def unruled_pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "unruled.pdf"
    styles = getSampleStyleSheet()
    mono = styles["Code"]

    story = [
        Paragraph("Item 7. Management's Discussion and Analysis", styles["Heading2"]),
        Paragraph(
            "Consolidated results for the fiscal year are summarised below. Net "
            "sales grew across every reportable segment during the period.",
            styles["BodyText"],
        ),
        Spacer(1, 14),
    ]
    story += [Paragraph(line.replace(" ", "&nbsp;"), mono) for line in UNRULED]
    story += [
        PageBreak(),
        Paragraph("Item 8. Financial Statements and Supplementary Data", styles["Heading2"]),
        Paragraph(
            "The accompanying notes are an integral part of these consolidated "
            "financial statements and should be read together with Item 7.",
            styles["BodyText"],
        ),
    ]
    SimpleDocTemplate(str(path), pagesize=letter).build(story)
    return path


@pytest.fixture(scope="module")
def document(unruled_pdf):
    return PdfPlumberLoader().load(unruled_pdf, ticker="TEST", form_type="10-K")


def test_an_unruled_statement_is_not_silently_dropped(document):
    """Whether it lands as a table or as text, the figures must survive."""
    everything = " ".join(
        b.text if b.type is not BlockType.TABLE else b.table.to_markdown()
        for b in document.blocks
    )
    for figure in ("416,161", "133,050", "112,010"):
        assert figure in everything, f"{figure} was lost entirely"


def test_figures_stay_with_their_line_item(document):
    """The property that matters, whichever extraction path was taken.

    A flat "Operating income 133,050 123,216 114,301" is acceptable; what is not
    acceptable is the label and its figures ending up in different blocks.
    """
    blocks = [
        b.text if b.type is not BlockType.TABLE else b.table.to_markdown()
        for b in document.blocks
    ]
    owner = next(b for b in blocks if "Operating income" in b)
    assert "133,050" in owner
    # And ordered as printed, so the year columns remain readable.
    assert owner.index("133,050") < owner.index("123,216") < owner.index("114,301")


def test_pages_are_attributed_across_a_page_break(document):
    sections = {b.page: b.section for b in document.blocks if b.section}
    assert any(s and s.startswith("Item 7.") for p, s in sections.items() if p == 1)
    assert any(s and s.startswith("Item 8.") for p, s in sections.items() if p == 2)


def test_chunking_keeps_the_statement_in_one_piece(document):
    chunks = TableAwareChunker().chunk(document)
    owner = next(c for c in chunks if "Operating income" in c.text)
    assert "133,050" in owner.text


def test_a_pdf_with_no_tables_at_all_still_parses(tmp_path):
    path = tmp_path / "prose.pdf"
    styles = getSampleStyleSheet()
    SimpleDocTemplate(str(path), pagesize=letter).build(
        [
            Paragraph("Item 1A. Risk Factors", styles["Heading2"]),
            Paragraph(
                "Demand for the products is cyclical and closely correlated with "
                "capital expenditure budgets in the semiconductor industry.",
                styles["BodyText"],
            ),
        ]
    )
    doc = PdfPlumberLoader().load(path)
    assert doc.blocks
    assert not [b for b in doc.blocks if b.type is BlockType.TABLE]
    assert any("cyclical" in b.text for b in doc.blocks)


def test_an_empty_pdf_does_not_crash(tmp_path):
    path = tmp_path / "empty.pdf"
    SimpleDocTemplate(str(path), pagesize=letter).build([Spacer(1, 1)])
    doc = PdfPlumberLoader().load(path)
    assert doc.blocks == []
    assert TableAwareChunker().chunk(doc) == []
