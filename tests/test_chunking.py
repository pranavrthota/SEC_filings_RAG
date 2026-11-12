"""Tests for the property that motivates the whole project: tables stay intact."""

from __future__ import annotations

from sec_copilot.chunking.table_aware import TableAwareChunker
from sec_copilot.config import ChunkingConfig
from sec_copilot.models import Block, BlockType, Document, Table


def _income_statement() -> Table:
    return Table(
        header=("Line item", "FY2023", "FY2022", "FY2021"),
        rows=(
            ("Net sales", "18,742", "17,109", "15,388"),
            ("Operating income", "2,879", "2,113", "2,146"),
            ("Net income", "2,037", "1,409", "1,441"),
        ),
        caption="CONSOLIDATED STATEMENTS OF OPERATIONS (in millions)",
    )


def _doc(*blocks: Block) -> Document:
    return Document(
        doc_id="test", source_path="test.md", blocks=list(blocks), form_type="10-K"
    )


def test_table_is_its_own_chunk_and_never_merged_with_prose():
    doc = _doc(
        Block(BlockType.TEXT, page=4, text="Operating expenses increased. " * 5, order=0),
        Block(BlockType.TABLE, page=4, table=_income_statement(), order=1),
        Block(BlockType.TEXT, page=4, text="Liquidity remained strong. " * 5, order=2),
    )
    chunks = TableAwareChunker().chunk(doc)

    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1
    assert "Operating expenses" not in tables[0].text
    assert "Liquidity" not in tables[0].text


def test_every_figure_stays_on_the_row_that_names_it():
    doc = _doc(Block(BlockType.TABLE, page=4, table=_income_statement(), order=0))
    (chunk,) = TableAwareChunker().chunk(doc)

    # This is the failure mode a naive splitter produces: a number severed from
    # its line item and its fiscal-year column.
    for line_item, fy23, fy22, fy21 in _income_statement().rows:
        row = next(l for l in chunk.text.splitlines() if l.startswith(f"| {line_item} "))
        assert row.index(fy23) < row.index(fy22) < row.index(fy21)


def test_oversized_table_splits_on_rows_and_repeats_the_header():
    big = Table(
        header=("Segment", "FY2023", "FY2022"),
        rows=tuple((f"Segment {i}", f"{i},000", f"{i},100") for i in range(1, 40)),
    )
    doc = _doc(Block(BlockType.TABLE, page=9, table=big, order=0))
    chunks = TableAwareChunker(ChunkingConfig(table_max_chars=600)).chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert "| Segment | FY2023 | FY2022 |" in chunk.text
        # No chunk may end mid-row.
        for line in chunk.text.splitlines():
            if line.startswith("|"):
                assert line.rstrip().endswith("|")

    # Every source row survives somewhere, exactly once.
    for name, _, _ in big.rows:
        assert sum(f"| {name} |" in c.text for c in chunks) == 1


def test_prose_chunks_carry_section_and_page_and_do_not_straddle_items():
    doc = _doc(
        Block(BlockType.TEXT, page=2, text="Business text. " * 40, section="Item 1. Business", order=0),
        Block(BlockType.TEXT, page=3, text="Risk text. " * 40, section="Item 1A. Risk Factors", order=1),
    )
    chunks = TableAwareChunker().chunk(doc)

    for chunk in chunks:
        assert chunk.section in {"Item 1. Business", "Item 1A. Risk Factors"}
        assert not ("Business text" in chunk.text and "Risk text" in chunk.text)
    assert {c.page for c in chunks} == {2, 3}


def test_long_paragraph_is_split_not_truncated():
    text = " ".join(f"Sentence number {i} about revenue recognition." for i in range(200))
    doc = _doc(Block(BlockType.TEXT, page=1, text=text, order=0))
    chunks = TableAwareChunker(ChunkingConfig(max_chars=500, overlap_chars=0)).chunk(doc)

    assert len(chunks) > 1
    joined = " ".join(c.text for c in chunks)
    for i in (0, 99, 199):
        assert f"Sentence number {i} " in joined


def test_chunk_text_carries_provenance_prefix_for_retrieval():
    doc = Document(
        doc_id="acme",
        source_path="acme.md",
        company="Acme Industries, Inc.",
        form_type="10-K",
        blocks=[Block(BlockType.TABLE, page=4, table=_income_statement(), section="Item 7. MD&A", order=0)],
    )
    (chunk,) = TableAwareChunker().chunk(doc)

    # A bare table is unfindable by "MD&A revenue" without this prefix.
    assert chunk.text.startswith("[Acme Industries, Inc. | 10-K | Item 7. MD&A]")
    assert chunk.metadata["company"] == "Acme Industries, Inc."
