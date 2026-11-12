"""Loader for plain-text and markdown filings.

Useful for EDGAR's older .txt submissions, for markdown exported by another
parser, and for tests - it needs no third-party dependency at all. Pipe tables
and form-feed page breaks are both understood, so the demo corpus exercises the
same table-aware path as a real PDF.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import Block, BlockType, Document, Table
from .base import DocumentLoader

_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_DIVIDER = re.compile(r"^\s*\|(?:\s*:?-{2,}:?\s*\|)+\s*$")
# Page breaks as they appear in EDGAR .txt submissions and in markdown exports.
# An explicit page number is honoured when present, so a document that numbers
# its own pages cites the same numbers a reader would see.
_PAGE_MARKER = re.compile(
    r"^\s*(?:\f\s*)?(?:<PAGE>|\[\s*page\s*(?P<n1>\d+)?\s*\]|-{2,}\s*page\s+(?P<n2>\d+)\s*-{2,})\s*$",
    re.IGNORECASE,
)


class TextLoader(DocumentLoader):
    """Parses .txt/.md into prose blocks, markdown tables, and page numbers."""

    suffixes = (".txt", ".md", ".markdown", ".text")

    def load(self, path: str | Path, **metadata: Any) -> Document:
        path = Path(path)
        doc = Document(
            doc_id=self.make_doc_id(path, metadata.pop("doc_id", None)),
            source_path=str(path),
            title=metadata.pop("title", None) or path.stem,
        )
        for key, value in metadata.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
            else:
                doc.metadata[key] = value

        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        page = 1
        order = 0
        section: str | None = None
        paragraph: list[str] = []
        caption: str | None = None
        i = 0

        def flush() -> None:
            nonlocal order, section, caption
            text = " ".join(paragraph).strip()
            paragraph.clear()
            if not text:
                return
            heading = self.detect_heading(text)
            if heading:
                section = heading
                caption = text
                doc.blocks.append(
                    Block(BlockType.HEADING, page, text=text, section=section, order=order)
                )
            else:
                caption = text if len(text) <= 120 else None
                doc.blocks.append(
                    Block(BlockType.TEXT, page, text=text, section=section, order=order)
                )
            order += 1

        while i < len(lines):
            line = lines[i]

            marker = _PAGE_MARKER.match(line)
            if marker or "\f" in line:
                flush()
                stated = (marker.group("n1") or marker.group("n2")) if marker else None
                page = int(stated) if stated else page + 1
                i += 1
                continue

            if _PIPE_ROW.match(line):
                flush()
                table_lines: list[str] = []
                while i < len(lines) and _PIPE_ROW.match(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                table = self._parse_pipe_table(table_lines, caption)
                if table is not None:
                    doc.blocks.append(
                        Block(
                            BlockType.TABLE,
                            page,
                            table=table,
                            section=section,
                            order=order,
                        )
                    )
                    order += 1
                continue

            if not line.strip():
                flush()
                i += 1
                continue

            paragraph.append(line.strip())
            i += 1

        flush()
        return doc

    @staticmethod
    def _parse_pipe_table(lines: list[str], caption: str | None) -> Table | None:
        rows: list[tuple[str, ...]] = []
        for line in lines:
            if _PIPE_DIVIDER.match(line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            rows.append(tuple(cells))
        if len(rows) < 2:
            return None
        header, *body = rows
        return Table(header=header, rows=tuple(body), caption=caption)
