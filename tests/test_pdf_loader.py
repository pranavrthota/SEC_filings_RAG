"""PDF loader tests against a real, generated PDF.

Skipped unless pdfplumber and reportlab are both installed - the point is to
exercise actual PDF geometry, not a mock of it.
"""

from __future__ import annotations

import pytest

pdfplumber = pytest.importorskip("pdfplumber")
reportlab = pytest.importorskip("reportlab")

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from sec_copilot.chunking.table_aware import TableAwareChunker  # noqa: E402
from sec_copilot.ingestion.pdf_loader import PdfPlumberLoader  # noqa: E402
from sec_copilot.models import BlockType  # noqa: E402

ROWS = [
    ["Line item", "FY2023", "FY2022"],
    ["Net sales", "18,742", "17,109"],
    ["Operating income", "2,879", "2,113"],
    ["Net income", "2,037", "1,409"],
]


@pytest.fixture(scope="module")
def pdf_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "acme-10k.pdf"
    styles = getSampleStyleSheet()
    table = Table(ROWS)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    SimpleDocTemplate(str(path), pagesize=letter).build(
        [
            Paragraph(
                "Item 7. Management's Discussion and Analysis of Financial Condition",
                styles["Heading2"],
            ),
            Paragraph(
                "Consolidated net sales for fiscal 2023 increased over fiscal 2022, "
                "driven by price realization in the Automation Systems segment and "
                "continued subscription growth in Digital Solutions.",
                styles["BodyText"],
            ),
            Spacer(1, 18),
            table,
            PageBreak(),
            Paragraph("Item 8. Financial Statements and Supplementary Data", styles["Heading2"]),
            Paragraph(
                "The accompanying notes are an integral part of these consolidated "
                "financial statements and should be read together with Item 7.",
                styles["BodyText"],
            ),
        ]
    )
    return path


@pytest.fixture(scope="module")
def document(pdf_path):
    return PdfPlumberLoader().load(pdf_path, ticker="ACME", form_type="10-K")


def test_ruled_table_is_extracted_with_its_columns(document):
    tables = [b for b in document.blocks if b.type is BlockType.TABLE]
    assert len(tables) == 1

    table = tables[0].table
    assert table.header == ("Line item", "FY2023", "FY2022")
    assert ("Operating income", "2,879", "2,113") in table.rows


def test_table_text_is_masked_out_of_the_prose_stream(document):
    # This is the behaviour that stops "2,879 2,113" appearing as flat prose,
    # detached from the line item that names it.
    prose = " ".join(b.text for b in document.blocks if b.type is not BlockType.TABLE)
    assert "18,742" not in prose
    assert "Automation Systems" in prose


def test_pages_and_sections_are_attributed(document):
    sections = {b.page: b.section for b in document.blocks if b.section}
    assert any(s and s.startswith("Item 7.") for p, s in sections.items() if p == 1)
    assert any(s and s.startswith("Item 8.") for p, s in sections.items() if p == 2)


def test_chunking_a_real_pdf_keeps_the_statement_intact(document):
    chunks = TableAwareChunker().chunk(document)
    table_chunk = next(c for c in chunks if c.is_table)
    assert "| Operating income | 2,879 | 2,113 |" in table_chunk.text
    assert "Automation Systems" not in table_chunk.text
