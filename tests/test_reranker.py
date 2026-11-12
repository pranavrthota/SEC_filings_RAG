"""Reranker tests.

The cross-encoder itself is not exercised here - downloading a model in CI
would make the suite slow and network-dependent. What is tested is the wiring:
that the wrapper asks for a deep enough shortlist, that ordering is genuinely
handed to the reranker, and that index invalidation still reaches the wrapped
retriever. A stub reranker makes those assertions exact.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from sec_copilot import SecFilingRagPipeline
from sec_copilot.generation.claude_client import EchoAnswerGenerator
from sec_copilot.models import ScoredChunk
from sec_copilot.retrieval.reranker import Reranker, RerankingRetriever


class ReverseReranker(Reranker):
    """Reverses the candidate order - unmistakable evidence it ran."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        self.seen.append(len(candidates))
        return list(reversed(list(candidates)))[:k]


@pytest.fixture
def wrapped(pipeline):
    reranker = ReverseReranker()
    base = pipeline.retriever
    pipeline.retriever = RerankingRetriever(base, reranker, candidate_k=10)
    return pipeline, base, reranker


def test_reranker_decides_the_final_order(wrapped):
    pipeline, base, reranker = wrapped
    query = "operating income fiscal 2023"

    before = [sc.chunk_id for sc in base.retrieve(query, k=10)]
    after = [sc.chunk_id for sc in pipeline.retrieve(query, k=10)]

    assert after == list(reversed(before))
    assert reranker.seen, "the reranker was never consulted"


def test_a_deeper_shortlist_is_requested_than_is_returned(wrapped):
    pipeline, _, reranker = wrapped
    results = pipeline.retrieve("net sales", k=3)

    assert len(results) == 3
    # The whole point: rank 4-10 get a chance to be promoted into the top 3.
    assert reranker.seen[-1] > 3


def test_k_defaults_to_the_base_retrievers_top_k(wrapped):
    pipeline, base, _ = wrapped
    assert len(pipeline.retrieve("net sales")) == base.config.top_k


def test_metadata_filters_survive_the_wrapper(wrapped):
    pipeline, _, _ = wrapped
    results = pipeline.retrieve("net sales", k=5, where={"content_type": "table"})
    assert results
    assert all(sc.chunk.is_table for sc in results)

    assert pipeline.retrieve("net sales", where={"ticker": "NOPE"}) == []


def test_ingesting_through_a_wrapped_retriever_still_invalidates_the_sparse_index(
    settings, sample_path
):
    # A regression guard: the pipeline used to invalidate only if the retriever
    # was *literally* a HybridRetriever, which a wrapper silently defeated.
    pipeline = SecFilingRagPipeline(settings, generator=EchoAnswerGenerator())
    base = pipeline.retriever
    pipeline.retriever = RerankingRetriever(base, ReverseReranker(), candidate_k=10)

    pipeline.ingest(sample_path, ticker="ACME")
    assert base.bm25.size == pipeline.store.size


def test_empty_index_returns_nothing_without_consulting_the_reranker(settings):
    reranker = ReverseReranker()
    pipeline = SecFilingRagPipeline(settings, generator=EchoAnswerGenerator())
    pipeline.retriever = RerankingRetriever(pipeline.retriever, reranker, candidate_k=10)

    assert pipeline.retrieve("anything") == []
    assert not reranker.seen
