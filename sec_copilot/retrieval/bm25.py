"""A small BM25 implementation (no third-party dependency).

Sparse retrieval is not optional for filings. Analysts search for exact
strings - "Item 1A", "goodwill impairment", "394,328" - and a dense encoder
that has never seen a given figure will happily return a semantically similar
paragraph containing the wrong number. BM25 finds the literal token; the dense
half handles paraphrase. Fusing them is what makes the pair reliable.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from ..embeddings.hashing import tokenize
from ..models import Chunk, ScoredChunk


class BM25Index:
    """Okapi BM25 over chunk text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_len = 0.0
        self._ids: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Sequence[Chunk]) -> "BM25Index":
        """Index a corpus from scratch."""
        self._chunks = []
        self._term_freqs = []
        self._lengths = []
        self._doc_freq = Counter()
        self._ids = set()
        return self.add(chunks)

    def add(self, chunks: Sequence[Chunk]) -> "BM25Index":
        """Add documents without rebuilding the whole index.

        Ingesting a second filing should cost time proportional to that filing,
        not to everything indexed so far. Document frequencies and the average
        length are running totals, so an incremental add is exact rather than an
        approximation.
        """
        for chunk in chunks:
            if chunk.chunk_id in self._ids:
                continue
            self._ids.add(chunk.chunk_id)
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self._chunks.append(chunk)
            self._term_freqs.append(tf)
            self._lengths.append(len(tokens))
            self._doc_freq.update(tf.keys())

        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        return self

    def search(self, query: str, k: int, allowed_ids: set[str] | None = None) -> list[ScoredChunk]:
        if not self._chunks or k <= 0:
            return []

        q_terms = [t for t in tokenize(query) if t in self._doc_freq]
        if not q_terms:
            return []

        n = len(self._chunks)
        idf = {
            # BM25+ style floor: keeps very common terms from scoring negative.
            t: max(math.log(1 + (n - df + 0.5) / (df + 0.5)), 1e-6)
            for t, df in ((t, self._doc_freq[t]) for t in set(q_terms))
        }

        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._term_freqs):
            if allowed_ids is not None and self._chunks[i].chunk_id not in allowed_ids:
                continue
            length = self._lengths[i] or 1
            norm = self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1.0))
            score = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                score += idf[term] * (freq * (self.k1 + 1)) / (freq + norm)
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            ScoredChunk(chunk=self._chunks[i], score=score, sparse_rank=rank)
            for rank, (score, i) in enumerate(scored[:k], start=1)
        ]
