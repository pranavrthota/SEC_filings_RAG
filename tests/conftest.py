from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot import SecFilingRagPipeline, Settings
from sec_copilot.generation.claude_client import EchoAnswerGenerator

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data" / "acme-industries-10k-2023.md"


@pytest.fixture(scope="session")
def sample_path() -> Path:
    assert SAMPLE.exists(), f"missing sample filing at {SAMPLE}"
    return SAMPLE


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # The hashing backend keeps the suite offline and deterministic; the
    # sentence-transformer path is exercised separately where available.
    return Settings(index_dir=tmp_path / "index", embedding_backend="hashing")


@pytest.fixture
def pipeline(settings: Settings, sample_path: Path) -> SecFilingRagPipeline:
    p = SecFilingRagPipeline(settings, generator=EchoAnswerGenerator())
    p.ingest(
        sample_path,
        company="Acme Industries, Inc.",
        ticker="ACME",
        form_type="10-K",
        fiscal_year=2023,
    )
    return p
