"""End-to-end tests: ingestion, retrieval, persistence, and the blocking path."""

from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot import SecFilingRagPipeline, Settings
from sec_copilot.generation.claude_client import AnswerGenerator, EchoAnswerGenerator
from sec_copilot.ingestion.base import LoaderRegistry
from sec_copilot.models import BlockType, Citation, Claim, GroundedAnswer


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def test_ingest_extracts_tables_pages_and_sections(pipeline):
    chunks = pipeline.store.all_chunks()
    assert chunks

    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 3, "income statement, segment results and balance sheet"

    # The statement of operations is on page 4 of the sample and must be cited
    # as such.
    income = next(c for c in tables if "Net sales" in c.text)
    assert income.page == 4
    assert income.section.startswith("Item 7.")
    assert "| Operating income | 2,879 | 2,113 | 2,146 |" in income.text


def test_metadata_propagates_to_every_chunk(pipeline):
    for chunk in pipeline.store.all_chunks():
        assert chunk.metadata["ticker"] == "ACME"
        assert chunk.metadata["fiscal_year"] == 2023
        assert chunk.metadata["form_type"] == "10-K"


def test_reingesting_the_same_filing_replaces_rather_than_duplicates(pipeline, sample_path):
    before = pipeline.store.size
    pipeline.ingest(sample_path, company="Acme Industries, Inc.", ticker="ACME",
                    form_type="10-K", fiscal_year=2023)
    assert pipeline.store.size == before


def test_unsupported_file_type_is_reported_clearly(pipeline, tmp_path):
    bogus = tmp_path / "filing.docx"
    bogus.write_bytes(b"not really a docx")
    with pytest.raises(ValueError, match="No registered loader"):
        pipeline.ingest(bogus)


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def test_numeric_question_surfaces_the_statement_table(pipeline):
    results = pipeline.retrieve("What was operating income in fiscal 2023?", k=3)
    assert results
    assert any(sc.chunk.is_table and "Operating income" in sc.chunk.text for sc in results)


def test_exact_figure_lookup_works_via_the_sparse_half(pipeline):
    # A dense encoder alone will not reliably find a literal figure; BM25 will.
    results = pipeline.retrieve("18,742", k=5)
    assert any("18,742" in sc.chunk.text for sc in results)


def test_prose_question_surfaces_prose(pipeline):
    results = pipeline.retrieve("What supply chain risks does the company disclose?", k=5)
    assert any("semiconductor components" in sc.chunk.text for sc in results)


def test_metadata_filter_restricts_the_candidate_pool(pipeline):
    tables = pipeline.retrieve("net sales", k=10, where={"content_type": "table"})
    assert tables
    assert all(sc.chunk.content_type is BlockType.TABLE for sc in tables)

    assert pipeline.retrieve("net sales", k=10, where={"ticker": "NOPE"}) == []


def test_retrieval_on_an_empty_index_returns_nothing(settings):
    empty = SecFilingRagPipeline(settings, generator=EchoAnswerGenerator())
    assert empty.retrieve("anything") == []


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_index_round_trips_through_disk(pipeline, settings):
    pipeline.persist()
    reloaded = SecFilingRagPipeline.load(settings.index_dir, settings)

    assert reloaded.store.size == pipeline.store.size
    original = pipeline.retrieve("operating income fiscal 2023", k=3)
    restored = reloaded.retrieve("operating income fiscal 2023", k=3)
    assert [s.chunk_id for s in restored] == [s.chunk_id for s in original]


def test_loading_an_index_built_with_another_embedding_model_is_refused(pipeline, settings):
    from dataclasses import replace

    pipeline.persist()
    # Silently querying across embedding spaces returns plausible-looking
    # neighbours that are meaningless, so this must fail loudly.
    mismatched = replace(settings, embedding_backend="hashing", embedding_dim=128)
    with pytest.raises(ValueError, match="was built with embedding model"):
        SecFilingRagPipeline.load(settings.index_dir, mismatched)


# --------------------------------------------------------------------------- #
# Answering and blocking
# --------------------------------------------------------------------------- #


class _FabricatingGenerator(AnswerGenerator):
    """Stands in for a model that invents a figure and a quote."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question, results, *, extra_instruction=None):
        self.calls += 1
        top = results[0].chunk
        return (
            GroundedAnswer(
                answerable=True,
                answer="Operating income was 9,999 million in fiscal 2023.",
                claims=[
                    Claim(
                        text="Operating income was 9,999 million in fiscal 2023.",
                        citations=[
                            Citation(
                                chunk_id=top.chunk_id,
                                page=top.page,
                                snippet="Operating income for fiscal 2023 was 9,999 million.",
                            )
                        ],
                    )
                ],
            ),
            {"generator": "fabricating"},
        )


def test_a_fabricated_answer_is_withheld_and_explained(settings, sample_path):
    generator = _FabricatingGenerator()
    p = SecFilingRagPipeline(settings, generator=generator)
    p.ingest(sample_path, ticker="ACME", form_type="10-K", fiscal_year=2023)

    result = p.answer("What was operating income in fiscal 2023?")

    assert result.blocked
    assert not result.to_dict()["answerable"]
    assert result.to_dict()["answer"] == ""
    assert "guardrail" in result.blocked_reason
    assert generator.calls == 2, "a failed verification triggers exactly one retry"


def test_guardrails_can_be_relaxed_to_report_only(settings, sample_path):
    from dataclasses import replace

    from sec_copilot.config import GuardrailConfig

    relaxed = replace(
        settings,
        guardrails=GuardrailConfig(
            block_on_failed_citation=False, block_on_ungrounded_number=False
        ),
    )
    p = SecFilingRagPipeline(relaxed, generator=_FabricatingGenerator())
    p.ingest(sample_path, ticker="ACME")

    result = p.answer("What was operating income in fiscal 2023?")
    assert not result.blocked
    assert not result.verification.passed, "the failure is still reported, just not fatal"


def test_a_well_cited_answer_passes_through(pipeline):
    result = pipeline.answer("What was operating income in fiscal 2023?")
    assert not result.blocked
    assert result.verification.passed
    assert result.to_dict()["claims"][0]["citations"][0]["page"] > 0


def test_question_with_no_retrievable_context_refuses(settings):
    p = SecFilingRagPipeline(settings, generator=EchoAnswerGenerator())
    result = p.answer("What was operating income?")
    assert not result.answer.answerable
    assert "No passages" in result.answer.refusal_reason


def test_describe_reports_the_wiring(pipeline):
    described = pipeline.describe()
    assert described["chunker"] == "TableAwareChunker"
    assert described["retriever"] == "HybridRetriever"
    assert described["vector_store"] == "InMemoryVectorStore"
    assert described["chunks_indexed"] == pipeline.store.size


def test_components_are_swappable_without_touching_the_pipeline(settings, sample_path):
    from sec_copilot.embeddings.hashing import HashingEmbeddingModel
    from sec_copilot.retrieval.retriever import DenseRetriever
    from sec_copilot.vectorstore.in_memory import InMemoryVectorStore

    embedder = HashingEmbeddingModel(dimension=128)
    store = InMemoryVectorStore(dimension=embedder.dimension, embedding_name=embedder.name)
    p = SecFilingRagPipeline(
        settings,
        loaders=LoaderRegistry.default(),
        embedding_model=embedder,
        store=store,
        retriever=DenseRetriever(store, embedder, settings.retrieval),
        generator=EchoAnswerGenerator(),
    )
    p.ingest(sample_path, ticker="ACME")

    assert p.describe()["retriever"] == "DenseRetriever"
    assert p.retrieve("operating income", k=3)
