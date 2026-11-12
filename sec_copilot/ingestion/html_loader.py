"""Loader for EDGAR HTML filings.

EDGAR's primary documents are HTML, and their financial statements are real
`<table>` elements - richer structure than a PDF ever gives us. This loader
walks the DOM in document order so tables keep their position relative to the
prose that introduces them, and reconstructs page numbers from the explicit
page breaks EDGAR filings carry.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ..models import Block, BlockType, Document, Table
from .base import DocumentLoader

_WS = re.compile(r"[\s ]+")
# Joining inline tags without a separator can weld a date together
# ("September 27,2025"). A comma followed by four digits is never a thousands
# separator - those groups are always three digits - so this is unambiguous.
_DATE_COMMA = re.compile(r",(?=(?:19|20)\d\d(?!\d))")
_PAGE_BREAK_STYLE = re.compile(r"page-break-(before|after)\s*:\s*always", re.IGNORECASE)
# `span` and `font` are deliberately absent: they are inline elements, and
# treating them as blocks fragments a heading that a filer split mid-word for
# styling ("INC" + "OME STATEMENTS") into two useless pieces.
_BLOCK_TAGS = {"p", "div", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class SecHtmlLoader(DocumentLoader):
    """Parses .htm/.html filings into prose blocks and structured tables."""

    suffixes = (".htm", ".html", ".xhtml")

    def __init__(self, *, min_table_cols: int = 2, min_paragraph_chars: int = 40) -> None:
        self.min_table_cols = min_table_cols
        self.min_paragraph_chars = min_paragraph_chars

    def load(self, path: str | Path, **metadata: Any) -> Document:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "SecHtmlLoader requires beautifulsoup4. Install with: pip install beautifulsoup4 lxml"
            ) from exc

        path = Path(path)
        raw = path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings():
            # EDGAR primary documents are inline XBRL and open with an XML
            # declaration. Parsing them as HTML is deliberate and correct.
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(raw, "lxml" if _has_lxml() else "html.parser")

        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        doc = Document(
            doc_id=self.make_doc_id(path, metadata.pop("doc_id", None)),
            source_path=str(path),
            title=metadata.pop("title", None) or (soup.title.get_text(strip=True) if soup.title else path.stem),
        )
        for key, value in metadata.items():
            if hasattr(doc, key):
                setattr(doc, key, value)
            else:
                doc.metadata[key] = value

        page = 1
        order = 0
        section: str | None = None
        body = soup.body or soup

        for node in self._walk(body):
            if node == "__PAGE_BREAK__":
                page += 1
                continue

            tag = node
            if tag.name == "table":
                table = self._to_table(tag)
                if table is not None:
                    doc.blocks.append(
                        Block(
                            type=BlockType.TABLE,
                            page=page,
                            table=table,
                            section=section,
                            order=order,
                        )
                    )
                    order += 1
                continue

            # Join inline children with no separator and let the source's own
            # whitespace decide word boundaries. Filers split words across
            # <span>s for styling ("INC" + "OME STATEMENTS"), and inserting a
            # separator turns that into "Inc Ome Statements".
            text = _DATE_COMMA.sub(", ", _WS.sub(" ", tag.get_text("", strip=False))).strip()
            if not text:
                continue

            heading = self.detect_heading(text) if (
                tag.name in _HEADING_TAGS or len(text) <= 100
            ) else None
            if heading:
                section = heading
                doc.blocks.append(
                    Block(
                        type=BlockType.HEADING,
                        page=page,
                        text=text,
                        section=section,
                        order=order,
                    )
                )
                order += 1
            elif len(text) >= self.min_paragraph_chars:
                doc.blocks.append(
                    Block(
                        type=BlockType.TEXT,
                        page=page,
                        text=text,
                        section=section,
                        order=order,
                    )
                )
                order += 1

        return doc

    # -- internals --------------------------------------------------------- #

    def _walk(self, root: Any) -> Iterator[Any]:
        """Yield tables and leaf-ish text blocks in document order.

        Tables are yielded whole and not descended into, so a cell's text never
        also surfaces as a stray paragraph.
        """
        for child in getattr(root, "children", []):
            name = getattr(child, "name", None)
            if name is None:
                continue

            if self._is_page_break(child):
                yield "__PAGE_BREAK__"

            if name == "table":
                yield child
                continue

            if name == "hr":
                yield "__PAGE_BREAK__"
                continue

            if name in _BLOCK_TAGS and not self._has_block_descendant(child):
                yield child
                continue

            yield from self._walk(child)

    @staticmethod
    def _is_page_break(tag: Any) -> bool:
        style = tag.get("style") if hasattr(tag, "get") else None
        return bool(style and _PAGE_BREAK_STYLE.search(style))

    @staticmethod
    def _has_block_descendant(tag: Any) -> bool:
        return any(
            getattr(d, "name", None) in (_BLOCK_TAGS | {"table"}) for d in tag.descendants
        )

    def _to_table(self, tag: Any) -> Table | None:
        rows: list[tuple[str, ...]] = []
        for tr in tag.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            values: list[str] = []
            for cell in cells:
                text = _DATE_COMMA.sub(", ", _WS.sub(" ", cell.get_text("", strip=False))).strip()
                # EDGAR pads statements with empty spacer cells and lone "$"
                # columns; dropping them keeps columns aligned to fiscal years.
                span = _int_attr(cell.get("colspan"), 1)
                values.append(text)
                values.extend([""] * (span - 1))
            rows.append(tuple(values))

        rows = [r for r in rows if any(c.strip() and c.strip() not in {"$", "%"} for c in r)]
        if len(rows) < 2:
            return None

        rows = _align_rows(rows)
        if not rows or len(rows[0]) < self.min_table_cols:
            return None

        header, body = _split_header(rows)
        if not body:
            return None
        return Table(header=tuple(header), rows=tuple(body))


def _int_attr(value: Any, default: int) -> int:
    try:
        return max(1, int(str(value)))
    except (TypeError, ValueError):
        return default


# An em-dash in a filing's data cell means nil, not padding - dropping it
# shortens the row and shifts every later figure one column left.
_DECORATION = {"", "$", "%"}

# A footnote reference that filers place in its own cell: "(a)", "(g)". Left in
# place it occupies a data column and shifts every figure after it - JPMorgan's
# income statement rendered "Total net revenue | 177,556 | (g) | 158,104",
# putting a footnote where the prior year's figure belongs. Only letters are
# matched: "(1)" is ambiguous with a real negative, so it is kept.
_FOOTNOTE_MARKER = re.compile(r"^\(\s*[a-z]{1,2}\s*\)$", re.IGNORECASE)


def _compact(row: tuple[str, ...]) -> tuple[str, ...]:
    """Drop a row's presentational cells, merging split negatives.

    EDGAR statements put the currency symbol in its own `<td>`, but only on the
    rows a filer chose to decorate - typically the first and the total. The
    result is that rows carry *different cell layouts*, and a plain
    column-by-column alignment silently shifts figures out from under their
    fiscal-year header. Compacting each row to its meaningful cells before
    aligning is what keeps `178,353` and `111,032` in the same column.
    """
    cells = [c.strip() for c in row]
    merged: list[str] = []
    i = 0
    while i < len(cells):
        cell = cells[i]
        # An accounting negative split across cells. Filers disagree on where
        # the boundaries fall - Apple emits "(", "321", ")" while Microsoft
        # emits "( 4,901" then ")" - so gather forward to the closing paren
        # instead of assuming a fixed shape. A cell that already closes its own
        # paren (a label like "Other income (expense), net", or a footnote
        # marker "(1)") is left alone.
        if cell.startswith("(") and ")" not in cell:
            parts = [cell.lstrip("(").strip()]
            j = i + 1
            # Bounded lookahead: a runaway merge would swallow real columns.
            while j < len(cells) and j <= i + 3 and ")" not in cells[j]:
                if cells[j]:
                    parts.append(cells[j])
                j += 1
            if j < len(cells) and j <= i + 3:
                tail = cells[j].replace(")", "").strip()
                if tail:
                    parts.append(tail)
                inner = " ".join(p for p in parts if p).strip()
                merged.append(f"({inner})" if inner else "")
                i = j + 1
                continue
        merged.append(cell)
        i += 1
    # Tidy negatives that arrived padded inside a single cell: "( 321 )".
    merged = [
        f"({c[1:-1].strip()})" if c.startswith("(") and c.endswith(")") else c for c in merged
    ]
    return tuple(
        c for c in merged if c not in _DECORATION and not _FOOTNOTE_MARKER.match(c)
    )


def _align_rows(rows: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Normalise ragged EDGAR rows to a single consistent width.

    Compaction is applied only if it actually produces agreement: if the
    compacted rows do not settle on one dominant width, the table is not a
    decorated statement and we fall back to dropping wholly empty columns,
    which is the safe behaviour for a genuinely ragged grid.
    """
    compacted = [_compact(r) for r in rows]
    widths = Counter(len(r) for r in compacted if len(r) > 1)
    if widths:
        target, count = widths.most_common(1)[0]
        # Two thirds of the substantive rows must agree before we trust it.
        substantive = sum(1 for r in compacted if len(r) > 1)
        if target >= 2 and count >= max(2, int(0.66 * substantive)):
            out: list[tuple[str, ...]] = []
            for row in compacted:
                if not row:
                    continue
                if len(row) == target:
                    out.append(row)
                elif len(row) == target - 1:
                    # A header that names only the value columns, or a section
                    # label row: give it the missing label column.
                    out.append(("",) + row)
                elif len(row) < target:
                    out.append(row + ("",) * (target - len(row)))
                else:
                    # Wider than the consensus: keep the label and the last
                    # `target - 1` figures, which is where the periods sit.
                    out.append((row[0],) + row[-(target - 1) :])
            return out

    return _drop_empty_columns(rows)


def _drop_empty_columns(rows: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Remove columns that are blank (or pure currency symbols) throughout."""
    width = max(len(r) for r in rows)
    padded = [r + ("",) * (width - len(r)) for r in rows]
    keep = [
        i
        for i in range(width)
        if any(padded[j][i].strip() not in {"", "$", "%", "(", ")"} for j in range(len(padded)))
    ]
    if not keep:
        return []
    return [tuple(row[i] for i in keep) for row in padded]


def _split_header(rows: list[tuple[str, ...]]) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Pick the header row, folding a two-row period header into one.

    Statements often stack "Years ended" above the actual dates. The row that
    names the periods is the one worth keeping, so prefer the first row whose
    value columns are all non-empty.
    """
    width = len(rows[0])
    for i, row in enumerate(rows[:2]):
        values = row[1:]
        if values and all(v.strip() for v in values):
            body = [r for j, r in enumerate(rows) if j != i and any(c.strip() for c in r)]
            if body:
                return row, body

    header, *body = rows
    if not any(h.strip() for h in header):
        header = tuple(f"col_{i + 1}" for i in range(width))
        body = rows
    return header, body


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401

        return True
    except ImportError:  # pragma: no cover - env-dependent
        return False
