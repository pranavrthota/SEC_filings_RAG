"""Table-aware chunking.

Two rules distinguish this from a generic recursive character splitter, and
both exist because of how financial statements fail under naive chunking:

1. **A table is never merged with prose and never split mid-row.** A fixed-size
   splitter cuts a balance sheet at character 1000, which lands mid-row and
   severs figures from the line item that names them. Here a table becomes its
   own chunk; only if it exceeds `table_max_chars` is it divided, and then on
   row boundaries with the **header row repeated on every part**, so every
   figure still sits under its fiscal-year column.

2. **Every chunk carries its section heading and page.** The heading is
   prefixed into the embedded text (so "revenue" in Item 7 is retrievable by
   "MD&A revenue discussion") and the page is preserved for citation.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..config import ChunkingConfig
from ..models import Block, BlockType, Chunk, Document, Table
from .base import Chunker

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


class TableAwareChunker(Chunker):
    """The default chunker for filings."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> list[Chunk]:
        base_meta = document.filter_metadata()
        chunks: list[Chunk] = []
        order = 0
        prose_buffer: list[Block] = []

        def flush_prose() -> None:
            nonlocal order
            if not prose_buffer:
                return
            for text, page, section in self._pack_prose(prose_buffer):
                chunks.append(
                    self._make_chunk(
                        document, text, page, BlockType.TEXT, section, order, base_meta
                    )
                )
                order += 1
            prose_buffer.clear()

        for block in document.blocks:
            if block.type is BlockType.TABLE:
                # Flushing first is what keeps the table out of a prose chunk.
                flush_prose()
                assert block.table is not None
                for text in self._render_table(block.table, block):
                    chunks.append(
                        self._make_chunk(
                            document,
                            text,
                            block.page,
                            BlockType.TABLE,
                            block.section,
                            order,
                            base_meta,
                        )
                    )
                    order += 1
            else:
                prose_buffer.append(block)

        flush_prose()
        return chunks

    # -- prose ------------------------------------------------------------- #

    def _pack_prose(self, blocks: Sequence[Block]) -> list[tuple[str, int, str | None]]:
        """Greedily pack consecutive prose blocks up to `max_chars`.

        Packing stops at a section change so a chunk never straddles two Items.
        """
        out: list[tuple[str, int, str | None]] = []
        buf: list[str] = []
        size = 0
        page = blocks[0].page
        section = blocks[0].section

        def emit() -> None:
            nonlocal buf, size
            if not buf:
                return
            text = "\n\n".join(buf).strip()
            if len(text) >= self.config.min_chars:
                out.append((text, page, section))
            buf, size = [], 0

        for block in blocks:
            if block.section != section:
                emit()
                page, section = block.page, block.section

            for piece in self._split_long_text(block.text):
                if size and size + len(piece) > self.config.max_chars:
                    tail = self._overlap_tail(buf)
                    emit()
                    page, section = block.page, block.section
                    if tail:
                        buf.append(tail)
                        size += len(tail)
                if not buf:
                    page = block.page
                buf.append(piece)
                size += len(piece)

        emit()
        return out

    def _split_long_text(self, text: str) -> list[str]:
        """Split a single oversized paragraph on sentence boundaries."""
        text = text.strip()
        if len(text) <= self.config.max_chars:
            return [text] if text else []

        pieces: list[str] = []
        current: list[str] = []
        size = 0
        for sentence in _SENTENCE_END.split(text):
            if size and size + len(sentence) > self.config.max_chars:
                pieces.append(" ".join(current))
                current, size = [], 0
            # A single sentence longer than the limit is hard-split rather than
            # dropped; rare in filings, but truncation is never acceptable.
            while len(sentence) > self.config.max_chars:
                pieces.append(sentence[: self.config.max_chars])
                sentence = sentence[self.config.max_chars :]
            current.append(sentence)
            size += len(sentence) + 1
        if current:
            pieces.append(" ".join(current))
        return [p.strip() for p in pieces if p.strip()]

    def _overlap_tail(self, buf: Sequence[str]) -> str:
        """The trailing slice of the emitted chunk, repeated as overlap."""
        if not self.config.overlap_chars or not buf:
            return ""
        joined = "\n\n".join(buf)
        tail = joined[-self.config.overlap_chars :]
        # Start the overlap at a sentence boundary when one is nearby.
        parts = _SENTENCE_END.split(tail, maxsplit=1)
        return (parts[1] if len(parts) == 2 else tail).strip()

    # -- tables ------------------------------------------------------------ #

    def _render_table(self, table: Table, block: Block) -> list[str]:
        """Serialise a table into one or more chunks, never cutting a row."""
        full = table.to_markdown()
        if len(full) <= self.config.table_max_chars or not table.rows:
            return [full]

        # Budget per part, less the header we will repeat on each one.
        header_cost = len(table.to_markdown(rows=())) + 40
        budget = max(self.config.table_max_chars - header_cost, 400)

        parts: list[str] = []
        current: list[tuple[str, ...]] = []
        size = 0
        for row in table.rows:
            row_len = sum(len(str(c)) for c in row) + 3 * table.n_cols + 1
            if current and size + row_len > budget:
                parts.append(current)
                current, size = [], 0
            current.append(row)
            size += row_len
        if current:
            parts.append(current)

        total = len(parts)
        rendered: list[str] = []
        for i, rows in enumerate(parts, start=1):
            note = f"_(table continued: rows {i} of {total} groups; header repeated)_\n\n"
            rendered.append((note if total > 1 else "") + table.to_markdown(rows=rows))
        return rendered

    # -- shared ------------------------------------------------------------ #

    def _make_chunk(
        self,
        document: Document,
        text: str,
        page: int,
        content_type: BlockType,
        section: str | None,
        order: int,
        base_meta: dict,
    ) -> Chunk:
        # The heading prefix is part of the embedded text: it is what makes a
        # bare table retrievable by the name of the statement it belongs to.
        prefix_bits = [b for b in (document.company, document.form_type, section) if b]
        prefix = " | ".join(str(b) for b in prefix_bits)
        embedded = f"[{prefix}]\n{text}" if prefix else text

        return Chunk(
            chunk_id=Chunk.make_id(document.doc_id, page, order, text),
            doc_id=document.doc_id,
            text=embedded,
            page=page,
            content_type=content_type,
            section=section,
            order=order,
            metadata=dict(base_meta),
        )
