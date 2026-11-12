"""Vector store abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models import Chunk, ScoredChunk


class VectorStore(ABC):
    """Persistent store of chunks plus their dense vectors.

    Implementations must keep the chunk payload alongside the vector: retrieval
    has to return page numbers and section labels, not just row ids, or the
    citation guardrail downstream has nothing to verify against.
    """

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed chunks."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        """Index chunks with their (n, dim) unit-norm vectors."""

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Top-k by cosine similarity, optionally filtered on chunk metadata."""

    @abstractmethod
    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        """Every chunk, for building the sparse index and for verification."""

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """Look up one chunk by id."""

    @abstractmethod
    def persist(self, path: str | Path) -> None:
        """Write the index to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Read an index back from disk."""

    @abstractmethod
    def clear(self) -> None:
        """Drop all indexed data."""

    # -- shared ------------------------------------------------------------ #

    @staticmethod
    def matches(chunk: Chunk, where: dict[str, Any] | None) -> bool:
        """Metadata predicate: equality, or membership when given a list/set."""
        if not where:
            return True
        for key, expected in where.items():
            actual = getattr(chunk, key, None)
            if actual is None:
                actual = chunk.metadata.get(key)
            if hasattr(actual, "value"):  # enum-valued fields like content_type
                actual = actual.value
            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True


def build_vector_store(kind: str, dimension: int, **kwargs: Any) -> VectorStore:
    """Factory for the configured store backend."""
    from .in_memory import InMemoryVectorStore

    kind = (kind or "in-memory").lower()
    if kind in {"in-memory", "memory", "numpy"}:
        return InMemoryVectorStore(dimension=dimension, **kwargs)
    if kind == "chroma":
        from .chroma import ChromaVectorStore

        return ChromaVectorStore(dimension=dimension, **kwargs)
    raise ValueError(f"Unknown vector store: {kind!r}")
