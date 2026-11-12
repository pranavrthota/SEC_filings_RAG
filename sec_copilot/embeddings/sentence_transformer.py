"""Local sentence-transformer embedding backend."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import EmbeddingModel


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Wraps a sentence-transformers model.

    The model is loaded lazily on first use so constructing a pipeline (and
    importing this module) stays cheap in tests and in the CLI's help path.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbeddingModel requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            # `get_sentence_embedding_dimension` was renamed; prefer the new
            # name and fall back so older sentence-transformers still work.
            get_dim = getattr(
                self._model,
                "get_embedding_dimension",
                getattr(self._model, "get_sentence_embedding_dimension", None),
            )
            self._dimension = int(get_dim())
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model
        assert self._dimension is not None
        return self._dimension

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self.model_name}"

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vectors = self.model.encode(
            payload,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return self.normalize(np.asarray(vectors, dtype=np.float32))

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.document_prefix)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)
