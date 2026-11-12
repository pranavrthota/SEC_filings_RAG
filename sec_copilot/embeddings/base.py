"""Embedding model abstraction.

Anthropic's API has no embeddings endpoint, so the dense retriever is
deliberately provider-independent: the abstraction below is what lets the
service run on a local sentence-transformer, a hosted embedding provider, or
the dependency-free hashing fallback without any other component noticing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class EmbeddingModel(ABC):
    """Maps text to unit-norm dense vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width. Must be stable for the life of an index."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier persisted with the index, to catch model mismatches."""

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus passages. Returns (len(texts), dimension) float32."""

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed queries. Overridden by asymmetric models with a query prefix."""
        return self.encode_documents(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise rows so a dot product is cosine similarity."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def build_embedding_model(
    backend: str = "auto",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    dimension: int = 512,
) -> EmbeddingModel:
    """Factory. `auto` prefers sentence-transformers, falling back to hashing."""
    from .hashing import HashingEmbeddingModel
    from .sentence_transformer import SentenceTransformerEmbeddingModel

    backend = (backend or "auto").lower()
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbeddingModel(model_name)
    if backend in {"hashing", "hash", "lexical"}:
        return HashingEmbeddingModel(dimension=dimension)
    if backend == "auto":
        try:
            return SentenceTransformerEmbeddingModel(model_name)
        except ImportError:
            return HashingEmbeddingModel(dimension=dimension)
    raise ValueError(f"Unknown embedding backend: {backend!r}")
