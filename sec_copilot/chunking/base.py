"""Chunker abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..models import Chunk, Document


class Chunker(ABC):
    """Turns a parsed `Document` into embeddable, citable `Chunk`s."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document."""

    def chunk_many(self, documents: Iterable[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in documents:
            out.extend(self.chunk(doc))
        return out
