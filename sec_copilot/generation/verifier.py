"""Post-generation guardrails.

Prompting a model to cite its sources is necessary but not sufficient: a
plausible-looking citation is exactly the kind of thing a language model is good
at fabricating. So nothing the model says about its own grounding is trusted
here. Every answer is re-checked against the retrieved text mechanically:

* **Citation verification** - the quoted snippet must appear, verbatim after
  whitespace normalisation, inside the chunk it claims to come from, and that
  chunk must have actually been retrieved. Near-misses are caught with a
  similarity ratio so a re-wrapped quote is not falsely rejected.

* **Numeric grounding, at cell level.** Every figure in the answer must be
  findable in the retrieved context *and* belong to the period the claim is
  about. Checking only that a figure appears somewhere in the text is not
  enough: in a three-year income statement, 133,050, 123,216 and 114,301 are
  all "in the context", so a text-level check cannot tell a correct answer from
  one that read the wrong column - which is the most likely genuine error in
  financial retrieval. Figures are therefore located by cell (row label plus
  dated column header) via `periods.parse_table_facts`.

* **Scale awareness.** A statement headed "(in millions)" means 133,050 and
  "$133.05 billion" are the same quantity. Comparisons are made on absolute
  values with a relative tolerance, so a correct restatement is not rejected as
  a fabrication.

* **Claim coverage** - a claim with no verified citation left standing is
  reported as uncited.

The report is advisory data; `GuardrailConfig` decides which failures block.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Sequence

from ..config import GuardrailConfig
from ..models import (
    Chunk,
    CitationCheck,
    GroundedAnswer,
    NumericCheck,
    NumericStatus,
    ScoredChunk,
    VerificationReport,
    VerificationStatus,
)
from .periods import (
    Cell,
    iter_numbers,
    parse_number,
    parse_table_facts,
    stated_scale,
    values_match,
    years_in,
)

_WS = re.compile(r"\s+")
# Figures as filings write them: 1,234.5  (2,150)  45.7%  $3.2
_NUMBER = re.compile(r"\(?\$?\s?\d[\d,]*(?:\.\d+)?\s?\)?%?")
_STRIP_CHARS = str.maketrans("", "", "$,()% ")


def normalize_text(text: str) -> str:
    """Collapse whitespace and case for verbatim comparison."""
    return _WS.sub(" ", text or "").strip().lower()


def normalize_number(token: str) -> str:
    """Reduce a figure to a stable key for de-duplication.

    Parentheses (the accounting negative) and currency decorations are dropped,
    and trailing zeros after a decimal point trimmed, so "1,234.50" and "1234.5"
    key the same.
    """
    core = token.translate(_STRIP_CHARS)
    if "." in core:
        core = core.rstrip("0").rstrip(".")
    return core


class CitationVerifier:
    """Mechanically re-checks a generated answer against its sources."""

    def __init__(
        self,
        config: GuardrailConfig | None = None,
        *,
        fact_index: Any | None = None,
    ) -> None:
        self.config = config or GuardrailConfig()
        # Optional inline-XBRL facts for the indexed filings. When present
        # they are authoritative about a figure's period, because the filer
        # tagged it rather than us inferring it from a rendered column.
        self.fact_index = fact_index

    def verify(
        self,
        answer: GroundedAnswer,
        retrieved: Sequence[ScoredChunk] | Sequence[Chunk],
    ) -> VerificationReport:
        chunks = [sc.chunk if isinstance(sc, ScoredChunk) else sc for sc in retrieved]
        by_id = {c.chunk_id: c for c in chunks}
        normalized_chunks = {cid: normalize_text(c.text) for cid, c in by_id.items()}

        report = VerificationReport()
        if not answer.answerable:
            # A refusal has nothing to prove; it is the safe outcome.
            return report

        for claim in answer.claims:
            claim_ok = False
            for citation in claim.citations:
                check = self._check_citation(citation, by_id, normalized_chunks)
                report.citation_checks.append(check)
                claim_ok = claim_ok or check.ok
            if not claim_ok:
                report.uncited_claims.append(claim.text)

        context = self._build_context(chunks)
        report.numeric_checks = self._check_numbers(answer, context)
        return report

    # -- citations --------------------------------------------------------- #

    def _check_citation(
        self,
        citation,
        by_id: dict[str, Chunk],
        normalized_chunks: dict[str, str],
    ) -> CitationCheck:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            return CitationCheck(
                citation,
                VerificationStatus.UNKNOWN_CHUNK,
                f"chunk_id {citation.chunk_id!r} was not among the retrieved passages",
            )

        if citation.page != chunk.page:
            return CitationCheck(
                citation,
                VerificationStatus.PAGE_MISMATCH,
                f"cited page {citation.page} but chunk is on page {chunk.page}",
            )

        haystack = normalized_chunks[citation.chunk_id]
        needle = normalize_text(citation.snippet)
        if needle and needle in haystack:
            return CitationCheck(citation, VerificationStatus.VERIFIED)

        ratio = self._best_window_ratio(needle, haystack)
        if ratio >= self.config.snippet_match_threshold:
            return CitationCheck(
                citation,
                VerificationStatus.VERIFIED,
                f"matched at similarity {ratio:.3f} (not byte-identical)",
            )

        return CitationCheck(
            citation,
            VerificationStatus.SNIPPET_NOT_FOUND,
            f"best similarity {ratio:.3f} < threshold {self.config.snippet_match_threshold}",
        )

    @staticmethod
    def _best_window_ratio(needle: str, haystack: str) -> float:
        """Similarity of the best-matching window of `haystack` to `needle`.

        Comparing against the whole chunk would drown a short quote in a long
        passage, so the comparison is anchored on the longest shared block.
        """
        if not needle or not haystack:
            return 0.0
        matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
        block = matcher.find_longest_match(0, len(needle), 0, len(haystack))
        if block.size == 0:
            return 0.0
        start = max(0, block.b - block.a)
        window = haystack[start : start + len(needle)]
        return SequenceMatcher(None, needle, window, autojunk=False).ratio()

    # -- numbers ----------------------------------------------------------- #

    def _build_context(self, chunks: Sequence[Chunk]) -> "_ContextModel":
        """Index the retrieved passages by cell, not merely by text."""
        model = _ContextModel(tolerance=self.config.numeric_tolerance)
        for chunk in chunks:
            facts = parse_table_facts(chunk.text)
            model.cells.extend(facts.cells)
            for _token, value in iter_numbers(chunk.text):
                model.loose.append((value, facts.scale))
        return model

    def _check_numbers(
        self, answer: GroundedAnswer, context: "_ContextModel"
    ) -> list[NumericCheck]:
        checks: list[NumericCheck] = []
        seen: set[str] = set()

        for claim in answer.claims:
            declared = self._figures_in(" ".join(claim.derived_from))
            inputs_grounded = bool(declared) and all(
                context.find(value, scale) for value, scale in declared
            )
            years = years_in(claim.text) or years_in(answer.answer)
            checks.extend(
                self._check_text(claim.text, context, years, inputs_grounded, declared, seen)
            )

        # Figures that appear only in the prose answer are checked too - that is
        # where an unsupported number usually hides.
        checks.extend(
            self._check_text(
                answer.answer, context, years_in(answer.answer), False, [], seen
            )
        )
        return checks

    @staticmethod
    def _figures_in(text: str) -> list[tuple[float, float]]:
        """(value, scale stated immediately after it) for each figure."""
        out: list[tuple[float, float]] = []
        for match in _NUMBER.finditer(text or ""):
            value = parse_number(match.group(0))
            if value is not None:
                out.append((value, stated_scale(text[match.end() : match.end() + 24])))
        return out

    def _check_text(
        self,
        text: str,
        context: "_ContextModel",
        years: frozenset[int],
        inputs_grounded: bool,
        declared: Sequence[tuple[float, float]],
        seen: set[str],
    ) -> list[NumericCheck]:
        out: list[NumericCheck] = []
        for match in _NUMBER.finditer(text or ""):
            token = match.group(0).strip()
            value = parse_number(token)
            if value is None or not self._significant(token):
                continue
            key = normalize_number(token)
            if key in seen:
                continue
            seen.add(key)

            # A four-digit number that is one of the periods under discussion is
            # a date, not a figure.
            if value == int(value) and int(value) in years:
                continue

            scale = stated_scale(text[match.end() : match.end() + 24])
            matches = context.find(value, scale)

            if matches:
                out.append(self._check_period(token, matches, years))
            elif inputs_grounded:
                inputs = ", ".join(f"{v:,.10g}" for v, _ in declared)
                out.append(
                    NumericCheck(
                        token,
                        True,
                        f"calculated from grounded inputs: {inputs}",
                        NumericStatus.DERIVED,
                    )
                )
            else:
                detail = "not present in any retrieved passage"
                if declared and not inputs_grounded:
                    detail += " and declared inputs are themselves ungrounded"
                out.append(NumericCheck(token, False, detail, NumericStatus.ABSENT))
        return out

    def _check_period(
        self, token: str, matches: Sequence[Cell], years: frozenset[int]
    ) -> NumericCheck:
        """Is this real figure attached to the period the claim is about?

        Only cells that actually carry a period can answer that; a figure quoted
        from prose, or from a table whose columns are not dated, is accepted
        because there is nothing to contradict it.
        """
        if not years:
            return NumericCheck(token, True, "", NumericStatus.GROUNDED)

        dated = [c for c in matches if c.years]
        if not dated:
            # The rendered passage cannot say which period this belongs to, but
            # the filer's own XBRL tagging can.
            tagged = self._xbrl_years(token)
            if tagged and not (tagged & years):
                return NumericCheck(
                    token,
                    False,
                    f"XBRL tags this figure to {', '.join(str(y) for y in sorted(tagged))}, "
                    f"but the claim is about {', '.join(str(y) for y in sorted(years))}",
                    NumericStatus.PERIOD_MISMATCH,
                )
            return NumericCheck(
                token, True, "period not stated in source", NumericStatus.GROUNDED
            )

        if any(c.years & years for c in dated):
            return NumericCheck(token, True, "", NumericStatus.GROUNDED)

        actual = sorted({y for c in dated for y in c.years})
        label = next((c.row_label for c in dated if c.row_label), "")
        detail = (
            f"{label + ': ' if label else ''}belongs to "
            f"{', '.join(str(y) for y in actual)}, but the claim is about "
            f"{', '.join(str(y) for y in sorted(years))}"
        )
        return NumericCheck(token, False, detail, NumericStatus.PERIOD_MISMATCH)

    def _xbrl_years(self, token: str) -> set[int]:
        """Fiscal years the filer tagged this value to, if XBRL is available.

        Deliberately conservative, because this is the only path that can
        contradict a period without a dated column to appeal to. A value is
        only trusted to identify a period when it is distinctive enough that a
        coincidental collision is implausible - a five-digit statement figure
        like 133,050, not a percentage like 46.9, which collided with an
        unrelated fact and blocked a correct answer.
        """
        if self.fact_index is None:
            return set()
        if sum(ch.isdigit() for ch in token) < 5:
            return set()
        value = parse_number(token)
        if value is None:
            return set()
        try:
            years = {y for y in self.fact_index.years_for(value) if y}
        except Exception:  # a malformed index must never break verification
            return set()
        # Ambiguous tagging proves nothing; only a unanimous answer can flag.
        return years if len(years) == 1 else set()

    def _significant(self, token: str) -> bool:
        """Ignore figures too small to be a meaningful hallucination signal.

        Single- and double-digit numbers appear constantly in ordinary prose
        ("three segments", "10 years"), and flagging them would make the
        guardrail fire on almost every answer.
        """
        return sum(ch.isdigit() for ch in token) >= self.config.numeric_min_digits


class _ContextModel:
    """Every figure available to the answer, with its period and scale."""

    def __init__(self, tolerance: float = 0.005) -> None:
        self.cells: list[Cell] = []
        self.loose: list[tuple[float, float]] = []
        self.tolerance = tolerance

    def find(self, value: float, stated: float = 1.0) -> list[Cell]:
        """Cells whose figure equals `value`, allowing for a restated scale."""
        wanted = {value}
        if stated != 1.0:
            wanted.add(value * stated)

        hits = [
            cell
            for cell in self.cells
            if any(
                values_match(w, cell.value, tolerance=self.tolerance)
                or values_match(w, cell.absolute, tolerance=self.tolerance)
                for w in wanted
            )
        ]
        if hits:
            return hits

        for raw, scale in self.loose:
            if any(
                values_match(w, raw, tolerance=self.tolerance)
                or values_match(w, raw * scale, tolerance=self.tolerance)
                for w in wanted
            ):
                # A prose figure carries no period, so it grounds the value but
                # can neither confirm nor deny an attribution.
                return [Cell(value=raw, raw=str(raw), row_label="", column_label="")]
        return []
