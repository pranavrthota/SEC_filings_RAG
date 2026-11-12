"""The service facade.

`SecFilingRagPipeline` is the single object an application (or an Aladdin-style
tool integration) holds. It owns one instance of each abstraction - loader,
chunker, embedder, store, retriever, generator, verifier - and none of them
knows about any other's implementation. Swapping the vector store or the
embedding model is a constructor argument; the guardrail path is unchanged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunking.base import Chunker
from .chunking.table_aware import TableAwareChunker
from .config import Settings
from .embeddings.base import EmbeddingModel, build_embedding_model
from .generation import build_generator
from .generation.claude_client import AnswerGenerator
from .generation.prompts import RETRY_INSTRUCTION
from .generation.verifier import CitationVerifier
from .ingestion.base import DocumentLoader, LoaderRegistry
from .ingestion.xbrl import XbrlFactIndex, extract_facts
from .models import (
    AnswerResult,
    Chunk,
    Document,
    GroundedAnswer,
    NumericStatus,
    ScoredChunk,
    VerificationReport,
)
from .retrieval.retriever import HybridRetriever, Retriever
from .vectorstore.base import VectorStore, build_vector_store

logger = logging.getLogger(__name__)


class SecFilingRagPipeline:
    """Ingest filings, answer questions about them, prove every answer."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        loaders: LoaderRegistry | None = None,
        chunker: Chunker | None = None,
        embedding_model: EmbeddingModel | None = None,
        store: VectorStore | None = None,
        retriever: Retriever | None = None,
        generator: AnswerGenerator | None = None,
        verifier: CitationVerifier | None = None,
    ) -> None:
        self.settings = settings or Settings()

        self.loaders = loaders or LoaderRegistry.default()
        self.chunker = chunker or TableAwareChunker(self.settings.chunking)
        self.embedding_model = embedding_model or build_embedding_model(
            self.settings.embedding_backend,
            self.settings.embedding_model,
            self.settings.embedding_dim,
        )
        self.store = store or build_vector_store(
            self.settings.vector_store,
            dimension=self.embedding_model.dimension,
            embedding_name=self.embedding_model.name,
        )
        self.retriever = retriever or HybridRetriever(
            self.store, self.embedding_model, self.settings.retrieval
        )
        self.generator = generator or build_generator(self.settings.generation)

        # Inline-XBRL facts accumulated across ingested filings. Shared with the
        # verifier so a figure's period can be checked against the filer's own
        # tagging rather than against a column header we inferred.
        self.facts = XbrlFactIndex()
        self.verifier = verifier or CitationVerifier(
            self.settings.guardrails, fact_index=self.facts
        )

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def register_loader(self, loader: DocumentLoader) -> "SecFilingRagPipeline":
        """Add a loader (e.g. an `unstructured`-backed one) at runtime."""
        self.loaders.register(loader)
        return self

    def ingest(self, path: str | Path, **metadata: Any) -> dict[str, Any]:
        """Parse, chunk, embed and index a single filing."""
        started = time.perf_counter()
        document = self.loaders.load(path, **metadata)
        chunks = self.chunker.chunk(document)
        self.index_chunks(chunks)
        n_facts = self._index_facts(path)

        stats = {
            "doc_id": document.doc_id,
            "source_path": document.source_path,
            "pages": document.n_pages,
            "blocks": len(document.blocks),
            "chunks": len(chunks),
            "table_chunks": sum(1 for c in chunks if c.is_table),
            "xbrl_facts": n_facts,
            "seconds": round(time.perf_counter() - started, 3),
        }
        logger.info("ingested %s: %s", document.doc_id, stats)
        return stats

    def ingest_directory(
        self, directory: str | Path, *, pattern: str = "*", **metadata: Any
    ) -> list[dict[str, Any]]:
        """Ingest every file in `directory` a registered loader can handle."""
        directory = Path(directory)
        stats: list[dict[str, Any]] = []
        for path in sorted(directory.rglob(pattern)):
            if not path.is_file():
                continue
            try:
                self.loaders.loader_for(path)
            except ValueError:
                continue
            try:
                stats.append(self.ingest(path, **metadata))
            except Exception as exc:  # one bad filing must not stop the batch
                logger.error("failed to ingest %s: %s", path, exc)
                stats.append({"source_path": str(path), "error": str(exc)})
        return stats

    def _index_facts(self, path: str | Path) -> int:
        """Pull inline-XBRL facts out of an EDGAR document, if it has any."""
        if not self.settings.extract_xbrl:
            return 0
        if Path(path).suffix.lower() not in {".htm", ".html", ".xhtml"}:
            return 0
        try:
            index = extract_facts(path)
        except Exception as exc:  # never let fact extraction break ingestion
            logger.warning("XBRL extraction failed for %s: %s", path, exc)
            return 0
        self.facts.facts.extend(index.facts)
        return len(index)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Embed and store pre-built chunks."""
        if not chunks:
            return
        vectors = self.embedding_model.encode_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        # Duck-typed so a retriever wrapped in a reranker still gets the call.
        invalidate = getattr(self.retriever, "invalidate", None)
        if callable(invalidate):
            invalidate()

    def ingest_documents(self, documents: Iterable[Document]) -> None:
        """Index already-parsed documents (bypassing the loader layer)."""
        self.index_chunks(self.chunker.chunk_many(documents))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def persist(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.settings.index_dir)
        self.store.persist(target)
        logger.info("persisted %d chunks to %s", self.store.size, target)
        return target

    @classmethod
    def load(
        cls, path: str | Path | None = None, settings: Settings | None = None, **kwargs: Any
    ) -> "SecFilingRagPipeline":
        """Rebuild a pipeline around an index already on disk."""
        settings = settings or Settings()
        target = Path(path or settings.index_dir)

        embedding_model = kwargs.pop("embedding_model", None) or build_embedding_model(
            settings.embedding_backend, settings.embedding_model, settings.embedding_dim
        )
        if settings.vector_store in {"in-memory", "memory", "numpy"}:
            from .vectorstore.in_memory import InMemoryVectorStore

            store: VectorStore = InMemoryVectorStore.load(target)
        else:
            from .vectorstore.chroma import ChromaVectorStore

            store = ChromaVectorStore.load(target)

        expected = getattr(store, "embedding_name", None)
        if expected and expected != embedding_model.name:
            raise ValueError(
                f"Index at {target} was built with embedding model {expected!r}, but "
                f"{embedding_model.name!r} is configured. Re-ingest or reconfigure - "
                "querying across models returns meaningless neighbours."
            )

        return cls(settings, embedding_model=embedding_model, store=store, **kwargs)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def retrieve(
        self, question: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        return self.retriever.retrieve(question, k=k, where=where)

    def answer(
        self,
        question: str,
        *,
        k: int | None = None,
        where: dict[str, Any] | None = None,
        retry_on_failure: bool = True,
    ) -> AnswerResult:
        """Answer a question, verify it, and block it if it cannot be proven.

        The retry is not a formality: a failed verification is fed back with the
        specific problems named, which converts most near-miss snippets (a
        re-wrapped quote, a figure copied from the wrong column) into a correct
        citation rather than a refusal.
        """
        results = self.retrieve(question, k=k, where=where)

        if not results:
            empty = GroundedAnswer(
                answerable=False,
                refusal_reason=(
                    "No passages in the indexed filings matched this question. "
                    "The filing may not be indexed, or the question may fall outside it."
                ),
            )
            return AnswerResult(
                question=question, answer=empty, verification=VerificationReport()
            )

        answer, usage = self.generator.generate(question, results)
        report = self.verifier.verify(answer, results)

        if retry_on_failure and not report.passed and answer.answerable:
            problems = self._describe(report)
            logger.info("verification failed, retrying once: %s", report.summary())
            retry_answer, retry_usage = self.generator.generate(
                question, results, extra_instruction=RETRY_INSTRUCTION.format(problems=problems)
            )
            retry_report = self.verifier.verify(retry_answer, results)
            usage = {"first_attempt": usage, "retry": retry_usage}
            if retry_report.passed or not answer.answerable:
                answer, report = retry_answer, retry_report
            else:
                # Keep whichever attempt is closer to provable.
                if len(retry_report.failed_citations) <= len(report.failed_citations):
                    answer, report = retry_answer, retry_report

        blocked, reason = self._should_block(report, answer)
        return AnswerResult(
            question=question,
            answer=answer,
            verification=report,
            retrieved=results,
            blocked=blocked,
            blocked_reason=reason,
            usage=usage,
        )

    # ------------------------------------------------------------------ #

    def _should_block(
        self, report: VerificationReport, answer: GroundedAnswer
    ) -> tuple[bool, str]:
        if not answer.answerable:
            return False, ""
        cfg = self.settings.guardrails
        reasons: list[str] = []
        if cfg.block_on_failed_citation and (report.failed_citations or report.uncited_claims):
            reasons.append("one or more claims could not be traced to the cited source text")
        absent = [n for n in report.ungrounded_numbers if n.status is NumericStatus.ABSENT]
        if cfg.block_on_ungrounded_number and absent:
            values = ", ".join(n.value for n in absent[:5])
            reasons.append(f"figures not found in the filing excerpts: {values}")
        if cfg.block_on_period_mismatch and report.period_mismatches:
            values = "; ".join(f"{n.value} ({n.detail})" for n in report.period_mismatches[:3])
            reasons.append(f"figures attributed to the wrong period: {values}")
        if not reasons:
            return False, ""
        return True, (
            "Answer withheld by the citation guardrail - "
            + "; ".join(reasons)
            + ". Inspect `verification` for the specific failures."
        )

    @staticmethod
    def _describe(report: VerificationReport) -> str:
        lines: list[str] = []
        for check in report.failed_citations:
            lines.append(
                f"- Citation to {check.citation.chunk_id} (page {check.citation.page}) "
                f"failed [{check.status.value}]: {check.detail}. "
                f"Snippet was: {check.citation.snippet[:120]!r}"
            )
        for num in report.ungrounded_numbers:
            if num.status is NumericStatus.PERIOD_MISMATCH:
                lines.append(
                    f"- Figure {num.value!r} is real but attributed to the wrong period - "
                    f"{num.detail}. Re-read the column headers and use the figure from the "
                    f"period the question asks about."
                )
            else:
                lines.append(f"- Figure {num.value!r} is not in any excerpt: {num.detail}")
        for claim in report.uncited_claims:
            lines.append(f"- Claim has no verifiable citation: {claim[:160]!r}")
        return "\n".join(lines) or "- unspecified verification failure"

    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return self.store.size

    def describe(self) -> dict[str, Any]:
        """The wiring, as data - useful in a health endpoint."""
        return {
            "chunks_indexed": self.store.size,
            "loaders": [type(l).__name__ for l in self.loaders.loaders],
            "chunker": type(self.chunker).__name__,
            "embedding_model": self.embedding_model.name,
            "embedding_dim": self.embedding_model.dimension,
            "vector_store": type(self.store).__name__,
            "retriever": type(self.retriever).__name__,
            "generator": type(self.generator).__name__,
            "xbrl_facts": len(self.facts),
            "provider": self.settings.generation.provider,
            "model": self.settings.generation.model,
            "guardrails": {
                "block_on_failed_citation": self.settings.guardrails.block_on_failed_citation,
                "block_on_ungrounded_number": self.settings.guardrails.block_on_ungrounded_number,
                "snippet_match_threshold": self.settings.guardrails.snippet_match_threshold,
            },
        }
