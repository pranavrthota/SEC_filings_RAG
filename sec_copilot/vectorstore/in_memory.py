"""In-memory vector store with on-disk persistence.

Exact brute-force cosine search over a numpy matrix. For a corpus of filings -
tens of thousands of chunks - this is fast enough to be the right default, and
exactness means retrieval quality is never confounded by ANN recall loss while
the guardrails are being evaluated. Swap in `ChromaVectorStore` (or FAISS
behind the same ABC) when the corpus outgrows it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models import BlockType, Chunk, ScoredChunk
from .base import VectorStore

_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.jsonl"
_MANIFEST_FILE = "manifest.json"


class InMemoryVectorStore(VectorStore):
    def __init__(self, dimension: int, *, embedding_name: str | None = None) -> None:
        self.dimension = dimension
        self.embedding_name = embedding_name
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, dimension), dtype=np.float32)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vector width {vectors.shape[1]} != store dimension {self.dimension}"
            )

        keep_rows: list[int] = []
        for i, chunk in enumerate(chunks):
            if chunk.chunk_id in self._index:
                # Re-ingesting the same filing replaces rather than duplicates.
                self._matrix[self._index[chunk.chunk_id]] = vectors[i]
                self._chunks[self._index[chunk.chunk_id]] = chunk
                continue
            self._index[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            keep_rows.append(i)

        if keep_rows:
            self._matrix = np.vstack([self._matrix, vectors[keep_rows]])

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if not self._chunks or k <= 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm:
            query = query / norm

        scores = self._matrix @ query

        allowed = (
            np.array([self.matches(c, where) for c in self._chunks], dtype=bool)
            if where
            else None
        )
        if allowed is not None:
            if not allowed.any():
                return []
            scores = np.where(allowed, scores, -np.inf)

        k = min(k, int(allowed.sum()) if allowed is not None else len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), dense_rank=rank)
            for rank, i in enumerate(top, start=1)
            if np.isfinite(scores[i])
        ]

    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        if not where:
            return list(self._chunks)
        return [c for c in self._chunks if self.matches(c, where)]

    def get(self, chunk_id: str) -> Chunk | None:
        i = self._index.get(chunk_id)
        return self._chunks[i] if i is not None else None

    def clear(self) -> None:
        self._chunks.clear()
        self._index.clear()
        self._matrix = np.zeros((0, self.dimension), dtype=np.float32)

    # -- persistence ------------------------------------------------------- #

    def persist(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / _VECTORS_FILE, self._matrix)
        with (path / _CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False) + "\n")
        (path / _MANIFEST_FILE).write_text(
            json.dumps(
                {
                    "store": "in-memory",
                    "dimension": self.dimension,
                    "embedding_name": self.embedding_name,
                    "size": self.size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "InMemoryVectorStore":
        path = Path(path)
        manifest_path = path / _MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        store = cls(
            dimension=int(manifest["dimension"]),
            embedding_name=manifest.get("embedding_name"),
        )
        matrix = np.load(path / _VECTORS_FILE)
        chunks = [
            _chunk_from_dict(json.loads(line))
            for line in (path / _CHUNKS_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(chunks) != len(matrix):
            raise ValueError(
                f"corrupt index: {len(chunks)} chunks vs {len(matrix)} vectors in {path}"
            )
        store._chunks = chunks
        store._index = {c.chunk_id: i for i, c in enumerate(chunks)}
        store._matrix = np.asarray(matrix, dtype=np.float32)
        return store


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "page": chunk.page,
        "content_type": chunk.content_type.value,
        "section": chunk.section,
        "order": chunk.order,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=data["chunk_id"],
        doc_id=data["doc_id"],
        text=data["text"],
        page=int(data["page"]),
        content_type=BlockType(data.get("content_type", "text")),
        section=data.get("section"),
        order=int(data.get("order", 0)),
        metadata=data.get("metadata") or {},
    )
