"""ChromaDB-backed vector store.

Included to demonstrate that the storage layer is genuinely swappable: nothing
above this file changes when the index moves out of process. Chroma is optional
- import errors are raised at construction, not at module import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..models import BlockType, Chunk, ScoredChunk
from .base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        dimension: int,
        *,
        collection_name: str = "sec_filings",
        persist_directory: str | Path | None = None,
        embedding_name: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaVectorStore requires chromadb. Install with: pip install chromadb"
            ) from exc

        self.dimension = dimension
        self.embedding_name = embedding_name
        self.collection_name = collection_name
        self.persist_directory = str(persist_directory) if persist_directory else None

        self._client = (
            chromadb.PersistentClient(path=self.persist_directory)
            if self.persist_directory
            else chromadb.EphemeralClient()
        )
        # Cosine space, so scores line up with the in-memory store's.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def size(self) -> int:
        return int(self._collection.count())

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            documents=[c.text for c in chunks],
            metadatas=[_to_metadata(c) for c in chunks],
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if k <= 0 or self.size == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).reshape(-1).tolist()],
            n_results=min(k, self.size),
            where=_to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )
        out: list[ScoredChunk] = []
        ids = result.get("ids", [[]])[0]
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(
                ids,
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ),
            start=1,
        ):
            out.append(
                ScoredChunk(
                    chunk=_from_metadata(cid, doc, meta),
                    # Chroma reports cosine *distance*; convert to similarity.
                    score=1.0 - float(dist),
                    dense_rank=rank,
                )
            )
        return out

    def all_chunks(self, where: dict[str, Any] | None = None) -> list[Chunk]:
        result = self._collection.get(
            where=_to_chroma_where(where), include=["documents", "metadatas"]
        )
        return [
            _from_metadata(cid, doc, meta)
            for cid, doc, meta in zip(
                result.get("ids", []), result.get("documents", []), result.get("metadatas", [])
            )
        ]

    def get(self, chunk_id: str) -> Chunk | None:
        result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = result.get("ids", [])
        if not ids:
            return None
        return _from_metadata(ids[0], result["documents"][0], result["metadatas"][0])

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def persist(self, path: str | Path) -> None:
        # A PersistentClient writes through on every mutation; an ephemeral one
        # cannot be persisted after the fact, so say so rather than fail quietly.
        if not self.persist_directory:
            raise RuntimeError(
                "This ChromaVectorStore is ephemeral. Construct it with "
                "persist_directory=... to persist."
            )

    @classmethod
    def load(cls, path: str | Path) -> "ChromaVectorStore":
        import json

        path = Path(path)
        manifest_path = path / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        return cls(
            dimension=int(manifest.get("dimension", 0)),
            collection_name=manifest.get("collection_name", "sec_filings"),
            persist_directory=path,
            embedding_name=manifest.get("embedding_name"),
        )


def _to_metadata(chunk: Chunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "doc_id": chunk.doc_id,
        "page": chunk.page,
        "content_type": chunk.content_type.value,
        "section": chunk.section or "",
        "order": chunk.order,
    }
    # Chroma only stores scalars, so non-scalar extras are dropped rather than
    # silently stringified into something no filter could match.
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)) and key not in meta:
            meta[key] = value
    return meta


def _from_metadata(chunk_id: str, document: str, meta: dict[str, Any]) -> Chunk:
    meta = dict(meta or {})
    return Chunk(
        chunk_id=chunk_id,
        doc_id=str(meta.get("doc_id", "")),
        text=document or "",
        page=int(meta.get("page", 0)),
        content_type=BlockType(meta.get("content_type", "text")),
        section=meta.get("section") or None,
        order=int(meta.get("order", 0)),
        metadata=meta,
    )


def _to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate the ABC's filter dialect into Chroma's operator syntax."""
    if not where:
        return None
    clauses: list[dict[str, Any]] = []
    for key, value in where.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: {"$eq": value}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}
