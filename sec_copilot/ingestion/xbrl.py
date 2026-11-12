"""Inline XBRL fact extraction.

EDGAR primary documents are inline XBRL: every reported figure is already
machine-tagged with its accounting concept, its exact period, its unit and its
decimal scale. Parsing the *rendered* table and hoping the columns line up is
strictly worse than reading the tagging the filer was legally required to
provide.

    <ix:nonFraction name="us-gaap:OperatingIncomeLoss"
                    contextRef="c-1" unitRef="usd" scale="6">133,050</ix:nonFraction>
    <xbrli:context id="c-1">
      <xbrli:period><xbrli:startDate>2024-09-29</xbrli:startDate>
                    <xbrli:endDate>2025-09-27</xbrli:endDate></xbrli:period>
    </xbrli:context>

That is unambiguous in exactly the two ways the HTML is not: the figure's period
and its scale. The fact index built here is used as authoritative ground truth
by the numeric guardrail, which is how a figure quoted for the wrong fiscal year
gets caught even when the rendered table it was read from is ambiguous.

Facts are *not* a replacement for the text pipeline: XBRL tags figures, not
prose, so risk factors and MD&A narrative still come from the HTML loader.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_NON_FRACTION = re.compile(r"^ix:nonfraction$", re.IGNORECASE)
_CONTEXT = re.compile(r"^(?:xbrli:)?context$", re.IGNORECASE)
# Namespace prefixes contain hyphens and digits ("us-gaap", "dei", "aapl"),
# so the character class must be wider than [a-zA-Z].
_LOCAL = re.compile(r"^\{[^}]*\}|^[\w.-]+:")


def _localname(tag: str) -> str:
    return _LOCAL.sub("", tag or "")


@dataclass(frozen=True)
class XbrlFact:
    """One tagged figure, with everything needed to interpret it."""

    concept: str  # e.g. "us-gaap:OperatingIncomeLoss"
    value: float  # already scaled and signed
    raw: str
    unit: str | None = None
    start: str | None = None  # ISO date, duration facts only
    end: str | None = None  # ISO date; the instant for point-in-time facts
    dimensions: tuple[tuple[str, str], ...] = ()

    @property
    def label(self) -> str:
        """A human-readable concept name: OperatingIncomeLoss -> Operating Income Loss."""
        local = _localname(self.concept)
        return re.sub(r"(?<!^)(?=[A-Z])", " ", local).strip()

    @property
    def fiscal_year(self) -> int | None:
        """The year a filer would call this figure's period.

        A fiscal year ending 2025-09-27 is FY2025 even though it began in 2024,
        so the end date governs.
        """
        if not self.end:
            return None
        try:
            return int(self.end[:4])
        except ValueError:
            return None

    @property
    def is_consolidated(self) -> bool:
        """True for company-wide figures; dimensioned facts are segment-level."""
        return not self.dimensions

    @property
    def is_instant(self) -> bool:
        """Balance-sheet style: a point in time rather than a period."""
        return self.start is None

    def describe(self) -> str:
        period = self.end if self.is_instant else f"{self.start}..{self.end}"
        segment = (
            " [" + ", ".join(f"{_localname(a)}={_localname(b)}" for a, b in self.dimensions) + "]"
            if self.dimensions
            else ""
        )
        return f"{self.label}{segment} ({period}) = {self.value:,.10g} {self.unit or ''}".strip()


@dataclass
class XbrlFactIndex:
    """All facts from one filing, queryable by value and by period."""

    facts: list[XbrlFact] = field(default_factory=list)
    source: str = ""

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def fiscal_years(self) -> set[int]:
        return {f.fiscal_year for f in self.facts if f.fiscal_year is not None}

    def by_concept(self, needle: str, *, consolidated_only: bool = True) -> list[XbrlFact]:
        needle = needle.lower()
        return [
            f
            for f in self.facts
            if needle in f.concept.lower() and (f.is_consolidated or not consolidated_only)
        ]

    def matching(self, value: float, *, tolerance: float = 0.005) -> list[XbrlFact]:
        """Facts whose value equals `value`, at any scale the filer might use.

        A figure is compared both as tagged (133050000000) and as printed in the
        statement (133,050 in a table headed "in millions"), because an answer
        may legitimately quote either.

        Scale equivalences are only considered for values of 1,000 or more. A
        percentage or a per-share figure is never a scaled-down millions figure,
        and allowing the division made 46.9 match an unrelated fact worth
        46,900,000 - which then blocked a correct answer.
        """
        if value == 0:
            return [f for f in self.facts if f.value == 0]

        scales = (1e3, 1e6, 1e9) if abs(value) >= 1000 else ()
        hits: list[XbrlFact] = []
        for fact in self.facts:
            for candidate in (fact.value, *(fact.value / s for s in scales)):
                if candidate == 0:
                    continue
                if abs(candidate - value) / max(abs(candidate), abs(value)) <= tolerance:
                    hits.append(fact)
                    break
        return hits

    def years_for(self, value: float, *, tolerance: float = 0.005) -> set[int]:
        """Fiscal years in which `value` is a reported figure."""
        return {
            f.fiscal_year
            for f in self.matching(value, tolerance=tolerance)
            if f.fiscal_year is not None
        }


def extract_facts(path: str | Path, *, consolidated_only: bool = False) -> XbrlFactIndex:
    """Parse the inline XBRL facts out of an EDGAR primary document."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "XBRL extraction requires beautifulsoup4 and lxml. "
            "Install with: pip install beautifulsoup4 lxml"
        ) from exc

    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        soup = BeautifulSoup(raw, "lxml")

    contexts = _parse_contexts(soup)
    index = XbrlFactIndex(source=str(path))

    for node in soup.find_all(_NON_FRACTION):
        value = _parse_value(node)
        if value is None:
            continue
        context = contexts.get((node.get("contextref") or "").strip(), {})
        fact = XbrlFact(
            concept=(node.get("name") or "").strip(),
            value=value,
            raw=node.get_text(strip=True),
            unit=(node.get("unitref") or None),
            start=context.get("start"),
            end=context.get("end"),
            dimensions=context.get("dimensions", ()),
        )
        if consolidated_only and not fact.is_consolidated:
            continue
        index.facts.append(fact)

    return index


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _parse_value(node: Any) -> float | None:
    """Apply the tag's scale and sign to its printed text."""
    text = node.get_text(strip=True).replace(",", "").replace("—", "").strip()
    if not text or not any(ch.isdigit() for ch in text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None

    try:
        scale = int(node.get("scale") or 0)
    except ValueError:
        scale = 0
    value *= 10**scale

    # `sign="-"` is how iXBRL marks a negative that is *rendered* in parentheses.
    if (node.get("sign") or "").strip() == "-":
        value = -value
    return value


def _parse_contexts(soup: Any) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for node in soup.find_all(_CONTEXT):
        cid = (node.get("id") or "").strip()
        if not cid:
            continue
        entry: dict[str, Any] = {}

        for child in node.find_all(True):
            local = _localname(child.name).lower()
            if local == "startdate":
                entry["start"] = child.get_text(strip=True)
            elif local == "enddate":
                entry["end"] = child.get_text(strip=True)
            elif local == "instant":
                entry["end"] = child.get_text(strip=True)

        entry["dimensions"] = tuple(
            (
                (member.get("dimension") or "").strip(),
                member.get_text(strip=True),
            )
            for member in node.find_all(True)
            if _localname(member.name).lower() in {"explicitmember", "typedmember"}
        )
        contexts[cid] = entry
    return contexts


def facts_to_markdown(facts: Iterable[XbrlFact], limit: int = 40) -> str:
    """Render facts as a table, for inspection or as retrievable context."""
    rows = list(facts)[:limit]
    if not rows:
        return ""
    lines = [
        "| Concept | Period | Value | Unit |",
        "| --- | --- | --- | --- |",
    ]
    for fact in rows:
        period = fact.end if fact.is_instant else f"{fact.start} to {fact.end}"
        lines.append(f"| {fact.label} | {period} | {fact.value:,.10g} | {fact.unit or ''} |")
    return "\n".join(lines)
