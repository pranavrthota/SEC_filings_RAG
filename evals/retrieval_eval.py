"""Retrieval evaluation on real filings.

    python -m evals.retrieval_eval                # default config
    python -m evals.retrieval_eval --rerank       # add the cross-encoder
    python -m evals.retrieval_eval --backend hashing

**The headline metric is `answerable@k`: does the retrieved context actually
contain the answer figure?** That is the only retrieval property the generator
depends on - if the figure is in context the model can produce a grounded,
verifiable answer, and if it is not, no amount of prompting will save it.

MRR and nDCG are reported alongside it as *ranking* diagnostics - how far down
the list the first usable chunk sits. They deliberately do not designate a
single "gold chunk": a figure legitimately appears in the statement, the MD&A
discussion and the segment note, and retrieving any of them lets the model
answer, so scoring against one arbitrary owner understates quality badly.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from sec_copilot import ChunkingConfig, ScoredChunk, SecFilingRagPipeline, Settings

ROOT = Path(__file__).resolve().parents[1]
FILINGS = ROOT / "filings"

CORPUS = {
    "aapl-10k-2025.htm": dict(company="Apple Inc.", ticker="AAPL", form_type="10-K", fiscal_year=2025),
    "aapl-10k-2024.htm": dict(company="Apple Inc.", ticker="AAPL", form_type="10-K", fiscal_year=2024),
    "msft-10k-2026.htm": dict(company="Microsoft Corporation", ticker="MSFT", form_type="10-K", fiscal_year=2026),
    "jpm-10k-2026.htm": dict(company="JPMorgan Chase & Co.", ticker="JPM", form_type="10-K", fiscal_year=2025),
}

# (question, the figure that must reach the model's context, metadata filter)
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("What were Apple's total net sales in fiscal 2025?",           "416,161", {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What was Apple's operating income in fiscal 2025?",           "133,050", {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What was Apple's gross margin in fiscal 2025?",               "195,201", {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What was Apple's diluted earnings per share in fiscal 2025?", "7.46",    {"ticker": "AAPL", "fiscal_year": 2025}),
    ("How much did Apple spend on research and development?",       "34,550",  {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What were Apple's total assets at the end of fiscal 2025?",   "359,241", {"ticker": "AAPL", "fiscal_year": 2025}),
    ("How much cash and cash equivalents did Apple hold?",          "35,934",  {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What were Apple's net sales in Greater China?",               "64,377",  {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What were Apple's iPhone net sales?",                         "209,586", {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What was Apple's non-current term debt?",                     "78,328",  {"ticker": "AAPL", "fiscal_year": 2025}),
    ("What was Microsoft's total revenue?",                         "331,839", {"ticker": "MSFT"}),
    ("What was Microsoft's operating income?",                      "155,237", {"ticker": "MSFT"}),
    ("What was Microsoft's net income?",                            "133,749", {"ticker": "MSFT"}),
    ("What was Microsoft's gross margin?",                          "225,465", {"ticker": "MSFT"}),
    ("What was Microsoft's diluted EPS?",                           "17.95",   {"ticker": "MSFT"}),
    ("What was Microsoft's provision for income taxes?",            "32,185",  {"ticker": "MSFT"}),
    # JPMorgan - a bank, whose statements share almost no vocabulary with the
    # two technology filings above.
    ("What was JPMorgan's total net revenue in 2025?",                "182,447", {"ticker": "JPM"}),
    ("What was JPMorgan's net income in 2025?",                       "57,048",  {"ticker": "JPM"}),
    ("What was the provision for credit losses in 2025?",             "14,212",  {"ticker": "JPM"}),
    ("What was JPMorgan's diluted earnings per share in 2025?",       "20.02",   {"ticker": "JPM"}),
    ("What was JPMorgan's book value per share at year end 2025?",    "126.99",  {"ticker": "JPM"}),
    ("What was JPMorgan's total noninterest expense in 2025?",        "95,640",  {"ticker": "JPM"}),
    ("What was JPMorgan's income before income tax expense in 2025?", "72,595",  {"ticker": "JPM"}),
    ("What was JPMorgan's market capitalization at year end 2025?",   "868,793", {"ticker": "JPM"}),
]

# Comparative and cross-filing questions. These need figures from two periods -
# and sometimes two filings - to reach the context *together*, which is a
# strictly harder retrieval problem than a single lookup.
COMPARATIVE: list[tuple[str, tuple[str, ...], dict[str, Any]]] = [
    ("How did Apple's operating income change from fiscal 2024 to fiscal 2025?",
     ("133,050", "123,216"), {"ticker": "AAPL", "fiscal_year": 2025}),
    ("Compare Apple's total net sales in fiscal 2025 with fiscal 2024.",
     ("416,161", "391,035"), {"ticker": "AAPL", "fiscal_year": 2025}),
    ("How did Apple's total assets move year over year?",
     ("359,241", "364,980"), {"ticker": "AAPL", "fiscal_year": 2025}),
    ("How did Microsoft's revenue and operating income both change year over year?",
     ("331,839", "281,724", "155,237", "128,528"), {"ticker": "MSFT"}),
    ("What were Apple's iPhone sales in fiscal 2025 versus fiscal 2024?",
     ("209,586", "201,183"), {"ticker": "AAPL", "fiscal_year": 2025}),
    ("Which company had higher operating income, Apple in fiscal 2025 or Microsoft?",
     ("133,050", "155,237"), {}),
    ("How did JPMorgan's provision for credit losses change from 2024 to 2025?",
     ("14,212", "10,678"), {"ticker": "JPM"}),
    ("How did JPMorgan's net income in 2025 compare with 2024?",
     ("57,048", "58,471"), {"ticker": "JPM"}),
]

KS = (1, 3, 5, 8, 12)

Retrieve = Callable[[str, int, dict[str, Any] | None], Sequence[ScoredChunk]]


def build(
    backend: str, chunking: ChunkingConfig, rerank: bool, model: str | None = None
) -> SecFilingRagPipeline:
    missing = [name for name in CORPUS if not (FILINGS / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing filings: {missing}\nRun: python -m evals.fetch_filings"
        )

    pipeline = SecFilingRagPipeline(
        Settings(
            index_dir=ROOT / "index-eval",
            embedding_backend=backend,
            chunking=chunking,
            **({"embedding_model": model} if model else {}),
        )
    )
    started = time.perf_counter()
    for name, metadata in CORPUS.items():
        pipeline.ingest(FILINGS / name, **metadata)
    print(
        f"indexed {pipeline.store.size} chunks from {len(CORPUS)} filings "
        f"in {time.perf_counter() - started:.1f}s  [{pipeline.embedding_model.name}]"
    )

    if rerank:
        from sec_copilot.retrieval.reranker import CrossEncoderReranker, RerankingRetriever

        pipeline.retriever = RerankingRetriever(
            pipeline.retriever, CrossEncoderReranker(), candidate_k=60
        )
        print("reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 over 60 candidates")
    return pipeline


def evaluate(pipeline: SecFilingRagPipeline, *, use_filter: bool) -> dict[str, float]:
    """answerable@k, plus how far down the list the first usable chunk sits."""
    answerable = {k: 0 for k in KS}
    # Context precision: what share of the passages we hand the model actually
    # carry the answer figure. Not a correctness measure - it is a measure of
    # how much of the context window (and cost) is doing useful work.
    precision = {k: 0.0 for k in KS}
    mrr = ndcg = 0.0
    found = 0
    started = time.perf_counter()

    for question, figure, where in CASES:
        results = pipeline.retrieve(question, k=max(KS), where=where if use_filter else None)

        for k in KS:
            window = results[:k]
            hits = sum(1 for sc in window if figure in sc.chunk.text)
            if hits:
                answerable[k] += 1
            precision[k] += hits / k if window else 0.0

        # Rank of the FIRST chunk carrying the figure. Designating a single
        # "gold" chunk would be arbitrary - a figure appears in the statement,
        # the MD&A discussion and the segment note, and any of them lets the
        # model answer - so rank the first usable one instead.
        hit = next((i for i, sc in enumerate(results, 1) if figure in sc.chunk.text), None)
        if hit:
            found += 1
            mrr += 1 / hit
            ndcg += 1 / math.log2(hit + 1)

    n = len(CASES)
    return {
        **{f"answerable@{k}": answerable[k] / n for k in KS},
        **{f"precision@{k}": precision[k] / n for k in KS},
        **{
            f"f1@{k}": (
                0.0
                if (precision[k] / n) + (answerable[k] / n) == 0
                else 2 * (precision[k] / n) * (answerable[k] / n)
                / ((precision[k] / n) + (answerable[k] / n))
            )
            for k in KS
        },
        # Averaged over all questions: a question whose figure never surfaces
        # contributes zero, which is the honest treatment.
        "mrr": mrr / n,
        "ndcg": ndcg / n,
        "found": found,
        "total": n,
        "ms_per_query": (time.perf_counter() - started) / n * 1000,
    }


def evaluate_comparative(pipeline: SecFilingRagPipeline, k: int = 12) -> dict[str, float]:
    """Fraction of multi-figure questions where EVERY needed figure reached context.

    Partial credit is also reported: getting three of four figures still leaves
    the model unable to answer, but it says how close retrieval came.
    """
    complete = 0
    partial = 0.0
    for question, figures, where in COMPARATIVE:
        results = pipeline.retrieve(question, k=k, where=where or None)
        text = " ".join(sc.chunk.text for sc in results)
        found = sum(1 for f in figures if f in text)
        partial += found / len(figures)
        complete += found == len(figures)
    n = len(COMPARATIVE)
    return {"complete": complete / n, "partial": partial / n, "n": n}


def report(label: str, metrics: dict[str, float]) -> None:
    print(f"\n{label}")
    print(f"  {'k':>4} {'answerable@k':>13} {'precision@k':>12} {'F1@k':>7}")
    for k in KS:
        print(
            f"  {k:>4} {metrics[f'answerable@{k}']:>13.3f} "
            f"{metrics[f'precision@{k}']:>12.3f} {metrics[f'f1@{k}']:>7.3f}"
        )
    print(
        f"  MRR={metrics['mrr']:.3f}  nDCG={metrics['ndcg']:.3f}  "
        f"(figure reached context for {metrics['found']:.0f}/{metrics['total']:.0f} questions)"
    )
    print(f"  {metrics['ms_per_query']:.0f} ms/query")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="sentence-transformers",
                        choices=["auto", "sentence-transformers", "hashing"])
    parser.add_argument("--model", default=None,
                        help="sentence-transformers model id to embed with")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--table-max-chars", type=int, default=None)
    args = parser.parse_args(argv)

    chunking = ChunkingConfig()
    if args.max_chars or args.table_max_chars:
        chunking = ChunkingConfig(
            max_chars=args.max_chars or chunking.max_chars,
            table_max_chars=args.table_max_chars or chunking.table_max_chars,
        )
        print(f"chunking override: max_chars={chunking.max_chars} table_max_chars={chunking.table_max_chars}")

    pipeline = build(args.backend, chunking, args.rerank, args.model)
    report("searching every filing (no metadata filter)", evaluate(pipeline, use_filter=False))
    report("filtered by ticker/year (how an analyst queries)", evaluate(pipeline, use_filter=True))

    comparative = evaluate_comparative(pipeline)
    print("\ncomparative / cross-filing questions (all figures must arrive together)")
    print(
        f"  complete@12={comparative['complete']:.2f}  "
        f"partial={comparative['partial']:.2f}  over {comparative['n']:.0f} questions"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
