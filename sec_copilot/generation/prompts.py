"""Prompt construction for grounded financial Q&A.

Two design choices are worth naming:

* The system prompt is a **frozen string constant**. It is the cache prefix for
  every request, so nothing volatile (timestamps, question text, retrieved
  context) may appear in it - see `render_context` for where those go.
* The context header for each passage carries the exact `chunk_id` and page the
  model must cite. Citation becomes a copying task rather than a recall task,
  which is the difference between a citation that verifies and one that reads
  plausibly and does not.
"""

from __future__ import annotations

from typing import Sequence

from ..models import ScoredChunk

SYSTEM_PROMPT = """\
You are a securities-filing analyst assistant. You answer questions strictly \
from the excerpts of SEC filings supplied to you in the user message, and you \
make every answer auditable.

Hard rules. These are not preferences:

1. Use ONLY the supplied excerpts. You have no other knowledge of these \
companies. If the excerpts do not contain the answer, set answerable to false \
and explain what is missing in refusal_reason. A refusal is a correct and \
valued outcome; a guess is a defect.

2. Every factual statement in your answer must appear as an entry in `claims`, \
and every claim must carry at least one citation. Do not put a fact in `answer` \
that is absent from `claims`.

3. Each citation must contain:
   - `chunk_id`: copied character-for-character from the [SOURCE ...] header of \
the excerpt you used. Never invent, abbreviate or reformat a chunk_id.
   - `page`: the page number shown in that same header.
   - `snippet`: a VERBATIM quote from that excerpt, copied exactly - same \
digits, same punctuation, same order. Do not paraphrase, round, reflow or \
translate it. For a figure from a table, quote the whole table row so the line \
item and the column value stay together.
   Every citation is checked mechanically against the source text. A snippet \
that is not found verbatim invalidates the whole answer.

4. Figures. Quote numbers exactly as printed, including the scale ("in \
millions") and the sign convention (parentheses mean negative). Never round, \
never convert units, never carry a figure from one fiscal year's column into \
another. If a figure you state is one you CALCULATED rather than read - a \
growth rate, a margin, a total - list every input figure in `derived_from` \
exactly as it appears in the source, and show the calculation in the claim text.

5. Ambiguity is information. If two excerpts disagree, if a figure is restated, \
or if the excerpt is a subtotal rather than the total asked for, say so in the \
answer rather than picking one silently.

6. State the period and scale for every figure you report ("FY2023, in \
millions"). An unqualified number is not an answer.

Write for a professional analyst: direct, specific, no preamble, no hedging \
language that adds nothing. Lead with the figure or the finding."""


def render_context(
    results: Sequence[ScoredChunk],
    *,
    max_chars: int | None = None,
) -> str:
    """Render retrieved passages with citable headers.

    Passages are emitted best-first and truncated at the passage boundary, never
    mid-passage: half a table row is worse than no table row.
    """
    parts: list[str] = []
    used = 0

    for i, sc in enumerate(results, start=1):
        chunk = sc.chunk
        header_bits = [
            f"SOURCE {i}",
            f"chunk_id={chunk.chunk_id}",
            f"page={chunk.page}",
            f"type={chunk.content_type.value}",
        ]
        if chunk.section:
            header_bits.append(f"section={chunk.section}")
        for key in ("company", "ticker", "form_type", "fiscal_year"):
            value = chunk.metadata.get(key)
            if value:
                header_bits.append(f"{key}={value}")

        block = f"[{' | '.join(header_bits)}]\n{chunk.text}"
        if max_chars is not None and used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block) + 2

    if not parts:
        return "(no excerpts retrieved)"
    return "\n\n".join(parts)


def build_user_message(question: str, context: str) -> str:
    """Assemble the user turn.

    Context precedes the question: it is the bulk of the tokens and the part
    that repeats across a follow-up on the same filing, so it sits earlier in
    the prefix where caching can reach it.
    """
    return (
        "Filing excerpts:\n\n"
        "<excerpts>\n"
        f"{context}\n"
        "</excerpts>\n\n"
        f"Analyst question: {question}\n\n"
        "Answer using only the excerpts above. Cite every claim with a verbatim "
        "snippet and the chunk_id and page from its [SOURCE ...] header."
    )


RETRY_INSTRUCTION = """\
Your previous answer failed mechanical verification. The problems found:

{problems}

Produce the answer again, fixing exactly those problems:
- Re-copy each snippet character-for-character from the excerpt named by its \
chunk_id. Do not adjust the source to match your prose; adjust your prose.
- Remove or correct any figure that does not appear in the excerpts, or declare \
its inputs in `derived_from`.
- If the excerpts genuinely do not support the answer, set answerable to false \
and say what is missing. That is the correct response here."""
