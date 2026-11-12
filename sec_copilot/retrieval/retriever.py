"""Retriever abstraction and the default hybrid implementation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..config import RetrievalConfig
from ..embeddings.base import EmbeddingModel
from ..models import Chunk, ScoredChunk
from ..vectorstore.base import VectorStore
from .bm25 import BM25Index

# Signals that a question is asking for a figure, which lives in a table.
_NUMERIC_QUERY = re.compile(
    r"\b(how much|how many|total|revenue|revenues|net income|loss|margin|eps|"
    r"cash|debt|assets|liabilities|equity|expense|expenses|cost|capex|"
    r"dividend|shares|ratio|percent|percentage|yoy|growth|balance|"
    r"amount|value|figure|number|\d[\d,.]*)\b",
    re.IGNORECASE,
)


class Retriever(ABC):
    """Selects the chunks an answer may be built from."""

    @abstractmethod
    def retrieve(
        self, query: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        """Return the top chunks for `query`, best first."""


class HybridRetriever(Retriever):
    """Dense + BM25 retrieval fused by reciprocal rank.

    Reciprocal rank fusion is used rather than score averaging because the two
    retrievers' scores are not on comparable scales (cosine in [-1, 1] vs.
    unbounded BM25), and normalising them per query is unstable when one side
    returns few candidates.
    """

    def __init__(
        self,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        config: RetrievalConfig | None = None,
        *,
        bm25: BM25Index | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()
        self._bm25 = bm25
        self._bm25_stale = bm25 is None

    # -- sparse index lifecycle -------------------------------------------- #

    def invalidate(self) -> None:
        """Mark the BM25 index stale; call after ingesting new documents."""
        self._bm25_stale = True

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index().build(self.store.all_chunks())
            self._bm25_stale = False
            return self._bm25

        if self._bm25_stale or self._bm25.size != self.store.size:
            # Ingest usually appends, so add only what is new; `add` skips ids
            # it already holds, and a shrunken store means a full rebuild.
            if self._bm25.size <= self.store.size:
                self._bm25.add(self.store.all_chunks())
            else:
                self._bm25 = BM25Index().build(self.store.all_chunks())
            self._bm25_stale = False
        return self._bm25

    # -- retrieval --------------------------------------------------------- #

    def retrieve(
        self, query: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        k = k or self.config.top_k
        if self.store.size == 0:
            return []

        candidate_k = max(self.config.candidate_k, k)

        query_vector = self.embedding_model.encode_query(query)
        dense = self.store.search(query_vector, candidate_k, where=where)

        allowed_ids = (
            {c.chunk_id for c in self.store.all_chunks(where)} if where else None
        )
        sparse = self.bm25.search(query, candidate_k, allowed_ids=allowed_ids)

        fused = self._fuse(dense, sparse)
        if self.config.table_boost != 1.0 and self._looks_numeric(query):
            for sc in fused:
                if sc.chunk.is_table:
                    sc.score *= self.config.table_boost
            fused.sort(key=lambda sc: -sc.score)

        return fused[:k]

    def _fuse(
        self, dense: Sequence[ScoredChunk], sparse: Sequence[ScoredChunk]
    ) -> list[ScoredChunk]:
        rrf_k = self.config.rrf_k
        merged: dict[str, ScoredChunk] = {}
        scores: dict[str, float] = {}

        for results, weight, attr in (
            (dense, self.config.dense_weight, "dense_rank"),
            (sparse, self.config.sparse_weight, "sparse_rank"),
        ):
            for rank, sc in enumerate(results, start=1):
                cid = sc.chunk_id
                entry = merged.get(cid)
                if entry is None:
                    entry = ScoredChunk(chunk=sc.chunk, score=0.0)
                    merged[cid] = entry
                setattr(entry, attr, rank)
                scores[cid] = scores.get(cid, 0.0) + weight / (rrf_k + rank)

        for cid, score in scores.items():
            merged[cid].score = score

        return sorted(merged.values(), key=lambda sc: -sc.score)

    @staticmethod
    def _looks_numeric(query: str) -> bool:
        return bool(_NUMERIC_QUERY.search(query))


class DenseRetriever(Retriever):
    """Dense-only retrieval, kept as a baseline for evaluating the hybrid."""

    def __init__(
        self,
        store: VectorStore,
        embedding_model: EmbeddingModel,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedding_model = embedding_model
        self.config = config or RetrievalConfig()

    def retrieve(
        self, query: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        k = k or self.config.top_k
        return self.store.search(self.embedding_model.encode_query(query), k, where=where)
