"""Layout-aware PDF loader built on pdfplumber.

The important behaviour is negative: text that pdfplumber identifies as living
inside a table's bounding box is *removed* from the page's prose stream. A
naive extractor emits "Total revenue 394,328 365,817 2022 2021" as flat text
and the retriever then serves a chunk in which no number is attached to a year.
Here the table leaves the prose stream entirely and re-enters as a `Table`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from ..models import Block, BlockType, Document, Table
from .base import DocumentLoader

_BLANK_RUN = re.compile(r"\n{2,}")


class PdfPlumberLoader(DocumentLoader):
    """Extracts prose and tables separately, preserving page and reading order."""

    suffixes = (".pdf",)

    def __init__(
        self,
        *,
        table_settings: dict[str, Any] | None = None,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
        y_tolerance: float = 3.0,
    ) -> None:
        # "lines" strategy first (ruled financial statements), falling back to
        # whitespace alignment for tables drawn without borders.
        self.table_settings = table_settings or {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
        }
        self.fallback_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "text_tolerance": 2,
        }
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols
        self.y_tolerance = y_tolerance

    def load(self, path: str | Path, **metadata: Any) -> Document:
        try:
            import pdfplumber
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "PdfPlumberLoader requires pdfplumber. Install with: pip install pdfplumber"
            ) from exc

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

        order = 0
        section: str | None = None
        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                tables = self._find_tables(page)

                prose = self._text_outside_tables(page, [t.bbox for t in tables])
                for para in self._paragraphs(prose):
                    heading = self._leading_heading(para)
                    if heading:
                        section = heading
                        doc.blocks.append(
                            Block(
                                type=BlockType.HEADING,
                                page=page_index,
                                text=para,
                                section=section,
                                order=order,
                            )
                        )
                    else:
                        doc.blocks.append(
                            Block(
                                type=BlockType.TEXT,
                                page=page_index,
                                text=para,
                                section=section,
                                order=order,
                            )
                        )
                    order += 1

                for found in tables:
                    table = self._to_table(found.rows)
                    if table is None:
                        continue
                    doc.blocks.append(
                        Block(
                            type=BlockType.TABLE,
                            page=page_index,
                            table=table,
                            section=section,
                            order=order,
                        )
                    )
                    order += 1

        return doc

    # -- internals --------------------------------------------------------- #

    class _Found:
        __slots__ = ("bbox", "rows")

        def __init__(self, bbox: tuple[float, float, float, float], rows: list[list[Any]]) -> None:
            self.bbox = bbox
            self.rows = rows

    def _find_tables(self, page: Any) -> list["PdfPlumberLoader._Found"]:
        """Locate tables, trying ruled detection then whitespace alignment."""
        for settings, numeric_only in (
            (self.table_settings, False),
            (self.fallback_settings, True),
        ):
            found: list[PdfPlumberLoader._Found] = []
            try:
                for tbl in page.find_tables(table_settings=settings):
                    rows = tbl.extract()
                    if self._plausible(rows, numeric_only=numeric_only):
                        found.append(self._Found(tbl.bbox, rows))
            except Exception:  # pdfplumber raises on some malformed pages
                continue
            if found:
                return found
        return []

    def _plausible(
        self, rows: Sequence[Sequence[Any]] | None, *, numeric_only: bool = False
    ) -> bool:
        if not rows or len(rows) < self.min_table_rows:
            return False
        width = max(len(r) for r in rows)
        if width < self.min_table_cols:
            return False
        filled = sum(1 for r in rows for c in r if c and str(c).strip())
        # Reject near-empty grids, which whitespace detection loves to invent.
        if filled < max(4, 0.25 * len(rows) * width):
            return False

        if not numeric_only:
            return True

        # The whitespace strategy has no rules to anchor on, so it will happily
        # carve a paragraph of prose into columns - which is far worse than
        # missing the table, because the heading and sentences then arrive as
        # table cells. A financial statement is overwhelmingly numeric, so
        # require that most rows actually carry figures.
        numeric_rows = sum(1 for row in rows if self._numeric_cells(row) >= 2)
        return numeric_rows >= max(2, 0.5 * len(rows))

    @staticmethod
    def _numeric_cells(row: Sequence[Any]) -> int:
        count = 0
        for cell in row:
            text = str(cell or "").strip().strip("()$%").replace(",", "")
            if not text:
                continue
            try:
                float(text)
            except ValueError:
                continue
            count += 1
        return count

    def _text_outside_tables(
        self, page: Any, bboxes: Sequence[tuple[float, float, float, float]]
    ) -> str:
        """Page text with every table region masked out."""
        if not bboxes:
            return page.extract_text(y_tolerance=self.y_tolerance) or ""
        region = page
        for bbox in bboxes:
            try:
                region = region.outside_bbox(bbox)
            except Exception:
                continue
        return region.extract_text(y_tolerance=self.y_tolerance) or ""

    @staticmethod
    def _to_table(rows: Sequence[Sequence[Any]]) -> Table | None:
        cleaned = [
            tuple("" if c is None else str(c).replace("\n", " ").strip() for c in row)
            for row in rows
        ]
        cleaned = [r for r in cleaned if any(c for c in r)]
        if len(cleaned) < 2:
            return None
        header, *body = cleaned
        # A header of blanks is worse than synthesised column names.
        if not any(header):
            width = max(len(r) for r in cleaned)
            header = tuple(f"col_{i + 1}" for i in range(width))
            body = cleaned
        return Table(header=header, rows=tuple(body))

    def _paragraphs(self, prose: str) -> list[str]:
        """Split a page's text into blocks.

        Splitting on blank lines alone is not enough: `extract_text` separates
        every visual line with a single newline, and a densely typeset filing
        page contains no blank lines at all - so the whole page would arrive as
        one block with its heading buried inside, and every section boundary
        would be lost. A recognised heading therefore also starts a new block.
        """
        out: list[str] = []
        current: list[str] = []

        def flush() -> None:
            text = "\n".join(current).strip()
            current.clear()
            if text:
                out.append(text)

        for raw_para in _BLANK_RUN.split(prose):
            for line in raw_para.splitlines():
                if not line.strip():
                    continue
                if self.detect_heading(line):
                    flush()
                    out.append(line.strip())
                    continue
                current.append(line.strip())
            flush()

        flush()
        return out

    def _leading_heading(self, para: str) -> str | None:
        first_line = para.split("\n", 1)[0]
        return self.detect_heading(first_line)
