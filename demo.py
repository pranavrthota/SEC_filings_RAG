"""Walk through the pipeline's inputs and outputs, with no API key required.

    python demo.py

Shows four things in order:
  1. what ingestion produces from a filing (chunks, pages, tables)
  2. what retrieval returns for a question
  3. what a well-cited answer looks like
  4. what happens when the model invents a figure - the guardrail blocking path
"""

from __future__ import annotations

import json
from pathlib import Path

from sec_copilot import SecFilingRagPipeline, Settings
from sec_copilot.generation.claude_client import AnswerGenerator, EchoAnswerGenerator
from sec_copilot.models import Citation, Claim, GroundedAnswer

SAMPLE = Path(__file__).parent / "sample_data" / "acme-industries-10k-2023.md"
QUESTION = "What was operating income in fiscal 2023?"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


class FabricatingGenerator(AnswerGenerator):
    """Stands in for a model that invents a figure and a supporting quote."""

    def generate(self, question, results, *, extra_instruction=None):
        top = results[0].chunk
        return (
            GroundedAnswer(
                answerable=True,
                answer="Operating income was 9,999 million in fiscal 2023.",
                claims=[
                    Claim(
                        text="Operating income was 9,999 million in fiscal 2023.",
                        citations=[
                            Citation(
                                chunk_id=top.chunk_id,
                                page=top.page,
                                snippet="Operating income for fiscal 2023 was 9,999 million.",
                            )
                        ],
                    )
                ],
            ),
            {"generator": "fabricating-demo"},
        )


def build(generator: AnswerGenerator) -> SecFilingRagPipeline:
    # The hashing backend keeps this demo offline; production defaults to a
    # sentence-transformer.
    pipeline = SecFilingRagPipeline(
        Settings(index_dir=Path("index-demo"), embedding_backend="hashing"),
        generator=generator,
    )
    pipeline.ingest(
        SAMPLE,
        company="Acme Industries, Inc.",
        ticker="ACME",
        form_type="10-K",
        fiscal_year=2023,
    )
    return pipeline


def main() -> None:
    pipeline = build(EchoAnswerGenerator())

    # ------------------------------------------------------------------ #
    rule("1. INGESTION - what the loader and chunker produced")
    print(f"input file: {SAMPLE}\n")
    for chunk in pipeline.store.all_chunks():
        kind = "TABLE" if chunk.is_table else "text "
        first = next(
            (l for l in chunk.text.splitlines() if l.strip() and not l.startswith("[")), ""
        )
        print(f"  p.{chunk.page}  {kind}  {first[:64]}")
    print(f"\n  {pipeline.store.size} chunks, "
          f"{sum(c.is_table for c in pipeline.store.all_chunks())} of them tables")

    # ------------------------------------------------------------------ #
    rule("2. RETRIEVAL - top passages for the question (no model call)")
    print(f"question: {QUESTION}\n")
    for i, sc in enumerate(pipeline.retrieve(QUESTION, k=3), start=1):
        print(f"  [{i}] score={sc.score:.5f}  page {sc.chunk.page}  {sc.chunk.content_type.value}")
        print(f"      {sc.chunk_id}")
    top = pipeline.retrieve(QUESTION, k=1)[0].chunk
    print(f"\n  full text of the top passage (note the table survived chunking intact):\n")
    print("      " + top.text.replace("\n", "\n      "))

    # ------------------------------------------------------------------ #
    rule("3. ANSWER - a well-cited answer passes verification")
    result = pipeline.answer(QUESTION)
    print(json.dumps(result.to_dict(), indent=2)[:1200] + "\n  ...")

    # ------------------------------------------------------------------ #
    rule("4. GUARDRAIL - a fabricated figure is caught and withheld")
    blocked = build(FabricatingGenerator()).answer(QUESTION)
    payload = blocked.to_dict()
    print(f"  model said:       {blocked.answer.answer}")
    print(f"  returned to user: {payload['answer']!r}")
    print(f"  answerable:       {payload['answerable']}")
    print(f"  reason:           {payload['refusal_reason']}")
    print(f"  ungrounded:       {payload['verification']['ungrounded_numbers']}")
    print(f"  citation checks:  {payload['verification']['citation_checks']}")
    print("\n  The filing says 2,879. The model said 9,999 with a real chunk_id attached.")
    print("  Nothing but the numeric-grounding check would have caught that.")


if __name__ == "__main__":
    main()
