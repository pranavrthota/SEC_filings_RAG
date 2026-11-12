"""Command line interface.

    python -m sec_copilot.cli ingest sample_data/ --company "Acme Industries, Inc." --form-type 10-K
    python -m sec_copilot.cli ask "What was operating income in fiscal 2023?"
    python -m sec_copilot.cli search "goodwill impairment" -k 5
    python -m sec_copilot.cli info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import GenerationConfig, GuardrailConfig, Settings
from .generation.claude_client import EchoAnswerGenerator, GenerationError
from .models import AnswerResult
from .pipeline import SecFilingRagPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sec-copilot",
        description="Table-aware RAG over SEC filings, with verified citations.",
    )
    parser.add_argument("--index", default=None, help="Index directory (default: ./index)")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "sentence-transformers", "hashing"],
        default=None,
        help="Dense embedding backend (default: auto)",
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "gemini", "echo"],
        default=None,
        help="Which model writes the answer (default: SEC_COPILOT_PROVIDER, else claude). "
             "The guardrails are identical for all of them.",
    )
    parser.add_argument("--model", default=None, help="Model id for the chosen provider")
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Parse, chunk, embed and index filings")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories to ingest")
    p_ingest.add_argument("--company")
    p_ingest.add_argument("--ticker")
    p_ingest.add_argument("--form-type", dest="form_type")
    p_ingest.add_argument("--fiscal-year", dest="fiscal_year", type=int)
    p_ingest.add_argument("--pattern", default="*", help="Glob used when a path is a directory")

    p_search = sub.add_parser("search", help="Retrieve passages without calling the model")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=8)
    p_search.add_argument("--ticker")
    p_search.add_argument("--fiscal-year", dest="fiscal_year", type=int)
    p_search.add_argument("--tables-only", action="store_true")

    p_ask = sub.add_parser("ask", help="Answer a question with verified citations")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=None)
    p_ask.add_argument("--ticker")
    p_ask.add_argument("--fiscal-year", dest="fiscal_year", type=int)
    p_ask.add_argument("--json", action="store_true", help="Emit the full result as JSON")
    p_ask.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Report verification failures without withholding the answer",
    )
    p_ask.add_argument(
        "--offline",
        action="store_true",
        help="Use the deterministic echo generator instead of calling Claude",
    )

    p_facts = sub.add_parser(
        "facts", help="Query the filing's own inline-XBRL tagged figures (exact, no retrieval)"
    )
    p_facts.add_argument("path", help="EDGAR HTML filing to read facts from")
    p_facts.add_argument("--concept", help="Substring of the us-gaap concept, e.g. OperatingIncome")
    p_facts.add_argument("--value", type=float, help="Find which periods report this figure")
    p_facts.add_argument("--segments", action="store_true", help="Include segment-level facts")
    p_facts.add_argument("--limit", type=int, default=25)

    sub.add_parser("info", help="Show the pipeline wiring and index size")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    overrides: dict[str, Any] = {}
    if args.index:
        overrides["index_dir"] = Path(args.index)
    if args.embedding_backend:
        overrides["embedding_backend"] = args.embedding_backend
    if args.provider or args.model:
        from .config import DEFAULT_GEMINI_MODEL, DEFAULT_MODEL

        provider = args.provider or settings.generation.provider
        # A model id belongs to its provider, so switching provider without
        # naming a model must not carry the previous provider's id across.
        if args.model:
            model = args.model
        elif args.provider and args.provider != settings.generation.provider:
            model = DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_MODEL
        else:
            model = settings.generation.model
        overrides["generation"] = GenerationConfig(
            provider=provider,
            model=model,
            max_tokens=settings.generation.max_tokens,
            effort=settings.generation.effort,
        )
    if getattr(args, "no_guardrails", False):
        overrides["guardrails"] = GuardrailConfig(
            block_on_failed_citation=False, block_on_ungrounded_number=False
        )
    if overrides:
        from dataclasses import replace

        settings = replace(settings, **overrides)
    return settings


def _open(settings: Settings, args: argparse.Namespace, *, must_exist: bool) -> SecFilingRagPipeline:
    generator = EchoAnswerGenerator() if getattr(args, "offline", False) else None
    index_dir = settings.index_dir
    if (index_dir / "manifest.json").exists():
        pipeline = SecFilingRagPipeline.load(index_dir, settings)
        if generator is not None:
            pipeline.generator = generator
        return pipeline
    if must_exist:
        raise SystemExit(
            f"No index at {index_dir}. Run `ingest` first, or pass --index to point elsewhere."
        )
    return SecFilingRagPipeline(settings, generator=generator)


def _where(args: argparse.Namespace) -> dict[str, Any] | None:
    where: dict[str, Any] = {}
    if getattr(args, "ticker", None):
        where["ticker"] = args.ticker
    if getattr(args, "fiscal_year", None):
        where["fiscal_year"] = args.fiscal_year
    if getattr(args, "tables_only", False):
        where["content_type"] = "table"
    return where or None


def cmd_ingest(pipeline: SecFilingRagPipeline, args: argparse.Namespace) -> int:
    metadata = {
        key: value
        for key, value in (
            ("company", args.company),
            ("ticker", args.ticker),
            ("form_type", args.form_type),
            ("fiscal_year", args.fiscal_year),
        )
        if value is not None
    }

    all_stats: list[dict[str, Any]] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            all_stats.extend(pipeline.ingest_directory(path, pattern=args.pattern, **metadata))
        elif path.is_file():
            all_stats.append(pipeline.ingest(path, **metadata))
        else:
            print(f"skipped (not found): {path}", file=sys.stderr)

    pipeline.persist()

    ok = [s for s in all_stats if "error" not in s]
    failed = [s for s in all_stats if "error" in s]
    print(f"Indexed {len(ok)} document(s) -> {pipeline.store.size} chunks in {pipeline.settings.index_dir}")
    for stat in ok:
        print(
            f"  {stat['doc_id']}: {stat['pages']} pages, {stat['chunks']} chunks "
            f"({stat['table_chunks']} tables) in {stat['seconds']}s"
        )
    for stat in failed:
        print(f"  FAILED {stat['source_path']}: {stat['error']}", file=sys.stderr)
    return 1 if failed and not ok else 0


def cmd_search(pipeline: SecFilingRagPipeline, args: argparse.Namespace) -> int:
    results = pipeline.retrieve(args.query, k=args.k, where=_where(args))
    if not results:
        print("No matching passages.")
        return 1
    for i, sc in enumerate(results, start=1):
        chunk = sc.chunk
        print(f"\n[{i}] score={sc.score:.5f}  {chunk.citation_label()}  ({chunk.content_type.value})")
        print(f"    chunk_id: {chunk.chunk_id}")
        body = chunk.text if chunk.is_table else chunk.text[:400] + ("..." if len(chunk.text) > 400 else "")
        print("    " + body.replace("\n", "\n    "))
    return 0


def cmd_ask(pipeline: SecFilingRagPipeline, args: argparse.Namespace) -> int:
    try:
        result = pipeline.answer(args.question, k=args.k, where=_where(args))
    except GenerationError as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0 if not result.blocked else 3

    _print_answer(result)
    return 0 if not result.blocked else 3


def _print_answer(result: AnswerResult) -> None:
    print(f"\nQ: {result.question}\n")

    if result.blocked:
        print("ANSWER WITHHELD")
        print(result.blocked_reason)
    elif not result.answer.answerable:
        print("NOT ANSWERABLE FROM THE INDEXED FILINGS")
        print(result.answer.refusal_reason)
    else:
        print(result.answer.answer)

    if result.answer.claims:
        print("\nClaims and evidence:")
        for i, claim in enumerate(result.answer.claims, start=1):
            print(f"\n  {i}. {claim.text}")
            if claim.derived_from:
                print(f"     calculated from: {', '.join(claim.derived_from)}")
            for citation in claim.citations:
                label = result._label_for(citation.chunk_id)
                print(f"     - {label} (page {citation.page})")
                print(f'       "{citation.snippet.strip()}"')

    print(f"\nVerification: {'PASS' if result.verification.passed else 'FAIL'} - "
          f"{result.verification.summary()}")

    if result.retrieved:
        pages = sorted({sc.chunk.page for sc in result.retrieved})
        print(f"Retrieved {len(result.retrieved)} passage(s) from page(s) {pages}")


def cmd_facts(pipeline: SecFilingRagPipeline | None, args: argparse.Namespace) -> int:
    """Read tagged figures straight from the filing - no retrieval, no model."""
    from .ingestion.xbrl import extract_facts

    path = Path(args.path)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    index = extract_facts(path)
    years = sorted(y for y in index.fiscal_years if y)
    print(f"{len(index)} tagged facts; fiscal years {years}\n")

    if args.value is not None:
        hits = index.matching(args.value)
        if not hits:
            print(f"No tagged fact matches {args.value:,.10g}.")
            return 1
        print(f"Facts matching {args.value:,.10g}:")
        for fact in hits[: args.limit]:
            print(f"  {fact.describe()}")
        return 0

    facts = (
        index.by_concept(args.concept, consolidated_only=not args.segments)
        if args.concept
        else (index.facts if args.segments else [f for f in index.facts if f.is_consolidated])
    )
    if not facts:
        print("No matching facts.")
        return 1
    for fact in sorted(facts, key=lambda f: (f.concept, f.end or ""))[: args.limit]:
        print(f"  {fact.describe()}")
    if len(facts) > args.limit:
        print(f"  ... {len(facts) - args.limit} more (raise --limit)")
    return 0


def cmd_info(pipeline: SecFilingRagPipeline, args: argparse.Namespace) -> int:
    print(json.dumps(pipeline.describe(), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # `facts` reads a file directly - no index, no embedding model, no cost.
    if args.command == "facts":
        return cmd_facts(None, args)

    settings = _settings(args)
    pipeline = _open(settings, args, must_exist=args.command in {"ask", "search"})

    handlers = {
        "ingest": cmd_ingest,
        "search": cmd_search,
        "ask": cmd_ask,
        "facts": cmd_facts,
        "info": cmd_info,
    }
    return handlers[args.command](pipeline, args)


if __name__ == "__main__":
    raise SystemExit(main())
