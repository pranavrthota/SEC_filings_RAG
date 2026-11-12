"""Period and scale reasoning for financial figures.

Two facts about filings drive everything here:

* **A statement is a grid, and the column is half the meaning.** "Operating
  income 133,050" is only true of one fiscal year; the same row also carries
  123,216 and 114,301. A guardrail that asks "does this figure appear in the
  context?" cannot tell those apart, because all three are in the context. So
  figures are located by *cell* - row label and column header together.

* **Statements are scaled, and the scale lives in the caption.** "(in millions)"
  above a table means 133,050 and "$133.05 billion" are the same quantity. A
  verifier that compares digits rejects a correct restatement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

# "September 27, 2025" | "FY2025" | "fiscal 2025" | "2025" | "June 30, 2026"
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_SCALE = re.compile(
    r"\(?\s*(?:amounts?\s+)?in\s+(thousands|millions|billions)\b", re.IGNORECASE
)
_SCALE_WORD = re.compile(r"\b(thousand|million|billion|trillion)s?\b", re.IGNORECASE)
_NUMBER = re.compile(r"\(?\$?\s?\d[\d,]*(?:\.\d+)?\s?\)?%?")

SCALE_FACTORS = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
}


@dataclass(frozen=True)
class Cell:
    """One figure, with everything needed to know what it means."""

    value: float
    raw: str
    row_label: str
    column_label: str
    years: frozenset[int] = frozenset()
    scale: float = 1.0

    @property
    def absolute(self) -> float:
        """The figure restated in units, i.e. with the table's scale applied."""
        return self.value * self.scale


@dataclass
class TableFacts:
    """Every figure in one rendered markdown table, indexed by cell."""

    cells: list[Cell] = field(default_factory=list)
    scale: float = 1.0
    caption: str = ""

    def years(self) -> set[int]:
        out: set[int] = set()
        for cell in self.cells:
            out |= cell.years
        return out


def parse_number(token: str) -> float | None:
    """Parse a filing-formatted figure. Parentheses mean negative."""
    text = token.strip().rstrip("%").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("$", "").replace(",", "").strip()
    if not text or not any(ch.isdigit() for ch in text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def detect_scale(text: str) -> float:
    """Return the multiplier implied by an '(in millions)' style caption."""
    match = _SCALE.search(text)
    if match:
        return SCALE_FACTORS[match.group(1).lower()]
    return 1.0


def years_in(text: str) -> frozenset[int]:
    return frozenset(int(m.group(0)) for m in _YEAR.finditer(text or ""))


def _split_markdown_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_divider(cells: Iterable[str]) -> bool:
    return all(set(c) <= set("-: ") and c for c in cells)


def parse_table_facts(text: str) -> TableFacts:
    """Extract cell-level facts from a chunk containing a markdown table.

    Non-table chunks simply yield no cells, and the verifier falls back to
    whole-text matching for those.
    """
    facts = TableFacts(scale=detect_scale(text))

    lines = text.splitlines()
    caption_parts = [l for l in lines if not l.startswith("|") and l.strip()]
    facts.caption = " ".join(caption_parts[:3])
    if facts.scale == 1.0:
        facts.scale = detect_scale(facts.caption)

    rows = [_split_markdown_row(l) for l in lines if l.strip().startswith("|")]
    rows = [r for r in rows if not _is_divider(r)]
    if len(rows) < 2:
        return facts

    header = rows[0]
    # A statement often stacks "Years ended" above the dates; prefer whichever
    # of the first two rows actually names periods.
    if len(rows) > 2 and not any(years_in(c) for c in header[1:]):
        for candidate in rows[1:3]:
            if any(years_in(c) for c in candidate[1:]):
                header = candidate
                break

    for row in rows:
        if row is header:
            continue
        label = row[0] if row else ""
        for i, raw in enumerate(row[1:], start=1):
            value = parse_number(raw)
            if value is None:
                continue
            column = header[i] if i < len(header) else ""
            facts.cells.append(
                Cell(
                    value=value,
                    raw=raw,
                    row_label=label,
                    column_label=column,
                    years=years_in(column),
                    scale=facts.scale,
                )
            )
    return facts


def iter_numbers(text: str) -> Iterator[tuple[str, float]]:
    """Yield (token, value) for every figure in free text."""
    for match in _NUMBER.finditer(text or ""):
        value = parse_number(match.group(0))
        if value is not None:
            yield match.group(0).strip(), value


def stated_scale(text: str) -> float:
    """The scale a claim states about itself ('$133.05 billion' -> 1e9)."""
    match = _SCALE_WORD.search(text or "")
    return SCALE_FACTORS[match.group(1).lower()] if match else 1.0


def values_match(a: float, b: float, *, tolerance: float = 0.005) -> bool:
    """Compare two quantities with a relative tolerance.

    Rounding is legitimate in an analyst's answer - 133,050 million reported as
    $133.05 billion, or a margin quoted to one decimal - so exact equality would
    reject correct work. Half a percent is tight enough to catch a transposed
    or fabricated figure.
    """
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0:
        return False
    return abs(a - b) / scale <= tolerance
