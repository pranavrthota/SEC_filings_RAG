"""Dependency-free hashing embedding backend.

This exists so the whole service - ingest, index, retrieve, guardrails - runs
with nothing but numpy installed, which matters for CI and for reviewers who
want to read the pipeline without downloading a torch wheel. It is a hashed
bag-of-ngrams projection with sublinear term weighting: genuinely useful for
lexical recall on filing text, and strictly weaker than a trained encoder on
paraphrase. The hybrid retriever's BM25 half carries the load either way.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterator, Sequence

import numpy as np

from .base import EmbeddingModel

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, keeping numbers with their separators intact.

    Keeping "394,328" and "1.5" as single tokens is deliberate: in a filing the
    figure *is* the search term, and splitting on the comma destroys it.
    """
    return _TOKEN.findall(text.lower())


class HashingEmbeddingModel(EmbeddingModel):
    """Hashed n-gram projection with sublinear term frequency weighting."""

    def __init__(self, dimension: int = 512, ngram_range: tuple[int, int] = (1, 2)) -> None:
        if dimension < 16:
            raise ValueError("dimension must be >= 16")
        self._dimension = dimension
        self.ngram_range = ngram_range

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        lo, hi = self.ngram_range
        return f"hashing:d{self._dimension}:n{lo}-{hi}"

    def _grams(self, tokens: Sequence[str]) -> Iterator[str]:
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            if n == 1:
                yield from tokens
            else:
                for i in range(len(tokens) - n + 1):
                    yield " ".join(tokens[i : i + n])

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        counts = Counter(self._grams(tokenize(text)))
        for gram, count in counts.items():
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            # Signed hashing keeps unrelated collisions from always adding up.
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        return vec

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return self.normalize(np.vstack([self._vector(t) for t in texts]))
