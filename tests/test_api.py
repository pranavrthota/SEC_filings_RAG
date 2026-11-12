"""HTTP surface tests, driven through FastAPI's dependency override."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sec_copilot import SecFilingRagPipeline
from sec_copilot.api import app, get_pipeline
from sec_copilot.generation.claude_client import EchoAnswerGenerator


@pytest.fixture
def client(pipeline):
    # No lifespan context: the app's startup would load the real on-disk index,
    # and these tests supply their own pipeline through the dependency.
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_the_wiring(client, pipeline):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == pipeline.store.size
    assert body["guardrails"]["block_on_failed_citation"] is True


def test_search_returns_scored_passages_without_calling_the_model(client):
    response = client.post("/search", json={"query": "operating income fiscal 2023", "k": 3})
    assert response.status_code == 200
    results = response.json()["results"]
    assert 0 < len(results) <= 3
    assert all({"chunk_id", "page", "score", "text"} <= set(r) for r in results)


def test_search_can_be_restricted_to_tables(client):
    results = client.post(
        "/search", json={"query": "net sales", "tables_only": True}
    ).json()["results"]
    assert results
    assert all(r["content_type"] == "table" for r in results)


def test_ask_returns_a_cited_answer(client):
    body = client.post("/ask", json={"question": "What was operating income in fiscal 2023?"}).json()
    assert body["answerable"] is True
    assert body["verification"]["passed"] is True
    assert body["claims"][0]["citations"][0]["chunk_id"]
    assert body["sources"]


def test_ask_against_an_empty_index_is_a_409(settings):
    empty = SecFilingRagPipeline(settings, generator=EchoAnswerGenerator())
    app.dependency_overrides[get_pipeline] = lambda: empty
    response = TestClient(app).post("/ask", json={"question": "anything at all"})
    app.dependency_overrides.clear()
    assert response.status_code == 409


def test_ingest_indexes_a_file(client, pipeline, sample_path, tmp_path):
    copy = tmp_path / "second-filing.md"
    copy.write_text(sample_path.read_text(encoding="utf-8"), encoding="utf-8")

    before = pipeline.store.size
    response = client.post(
        "/ingest",
        json={"path": str(copy), "ticker": "ACME2", "form_type": "10-K", "persist": False},
    )
    assert response.status_code == 201
    assert response.json()["chunks_indexed"] > before


def test_ingest_missing_path_is_a_404(client):
    assert client.post("/ingest", json={"path": "no/such/file.pdf"}).status_code == 404


def test_ingest_unsupported_type_is_a_415(client, tmp_path):
    bogus = tmp_path / "model.xlsx"
    bogus.write_bytes(b"nope")
    assert client.post("/ingest", json={"path": str(bogus)}).status_code == 415


def test_a_blank_question_is_rejected_by_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
