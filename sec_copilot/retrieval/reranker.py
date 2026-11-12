"""Cross-encoder reranking.

A bi-encoder embeds the query and the passage independently, so it can only
compare two vectors that never met. A cross-encoder reads the pair together and
scores the match directly - far more accurate, far too slow to run over a whole
corpus. The standard arrangement, and the one used here, is to let the cheap
hybrid retriever propose a shortlist and let the expensive model reorder it.

This is where retrieval *precision* is bought. Recall is already handled by
fusing dense and sparse candidates; what the reranker fixes is the ordering
inside that shortlist - the "segment table above the income statement" problem.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..models import ScoredChunk
from .retriever import Retriever

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Reorders a candidate list for a query."""

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Return the best `k` candidates, best first."""


class CrossEncoderReranker(Reranker):
    """Scores (query, passage) pairs with a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_text_chars: int = 2000,
    ) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        # Cross-encoders truncate at a few hundred tokens anyway; trimming here
        # keeps a long balance sheet from crowding out the query.
        self.max_text_chars = max_text_chars
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return list(candidates)

        pairs = [(query, sc.chunk.text[: self.max_text_chars]) for sc in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        reranked = [
            ScoredChunk(
                chunk=sc.chunk,
                score=float(score),
                dense_rank=sc.dense_rank,
                sparse_rank=sc.sparse_rank,
            )
            for sc, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda sc: -sc.score)
        return reranked[:k]


class RerankingRetriever(Retriever):
    """Wraps any retriever with a reranking pass.

    Composition rather than inheritance: the base retriever is asked for a
    deliberately generous shortlist (`candidate_k`), and the reranker decides
    what actually reaches the model. Swapping the base or the reranker changes
    nothing else in the pipeline.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int = 30,
    ) -> None:
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(
        self, query: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if k is None:
            config = getattr(self.base, "config", None)
            k = getattr(config, "top_k", 8)
        candidates = self.base.retrieve(query, k=max(self.candidate_k, k), where=where)
        if not candidates:
            return []
        return self.reranker.rerank(query, candidates, k)

    def invalidate(self) -> None:
        """Forward index invalidation to the wrapped retriever."""
        invalidate = getattr(self.base, "invalidate", None)
        if callable(invalidate):
            invalidate()
