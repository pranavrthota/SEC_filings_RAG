"""HTTP service.

    uvicorn sec_copilot.api:app --reload

One pipeline is constructed at startup and shared across requests. That is
correct rather than merely convenient: the embedding model and the index are
both expensive to build and safe to read concurrently, and the BM25 index is
extended incrementally when ingestion adds to it.

Retrieval, embedding and generation are all synchronous and CPU- or
network-bound, so the routes are `def` rather than `async def`: Starlette then
runs them in a worker thread and the event loop stays free. Declaring them
`async` would block the loop for the whole of a model call and serialise every
concurrent request behind it.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import Settings
from .generation.claude_client import GenerationError
from .pipeline import SecFilingRagPipeline
from .service import ApiKeyAuth, RateLimiter, UsageTracker, client_key

logger = logging.getLogger(__name__)

_state: dict[str, Any] = {}


def get_pipeline() -> SecFilingRagPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Pipeline not initialised"
        )
    return pipeline


def get_auth() -> ApiKeyAuth:
    return _state.setdefault("auth", ApiKeyAuth.from_env())


def get_limiter() -> RateLimiter:
    return _state.setdefault("limiter", RateLimiter.from_env())


def get_usage() -> UsageTracker:
    return _state.setdefault("usage", UsageTracker())


def guard(
    request: Request,
    auth: ApiKeyAuth = Depends(get_auth),
    limiter: RateLimiter = Depends(get_limiter),
) -> None:
    """Authenticate, then rate limit. Applied to every route that does work."""
    auth.check(request)
    limiter.check(client_key(request))


def build_pipeline(settings: Settings | None = None) -> SecFilingRagPipeline:
    """Load the persisted index if there is one, else start empty."""
    settings = settings or Settings.from_env()
    if (settings.index_dir / "manifest.json").exists():
        logger.info("loading index from %s", settings.index_dir)
        return SecFilingRagPipeline.load(settings.index_dir, settings)
    logger.warning("no index at %s - starting empty; POST /ingest to populate", settings.index_dir)
    return SecFilingRagPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = build_pipeline()
    _state.setdefault("auth", ApiKeyAuth.from_env())
    _state.setdefault("limiter", RateLimiter.from_env())
    _state.setdefault("usage", UsageTracker())
    yield
    _state.clear()


app = FastAPI(
    title="SEC Filing Co-pilot",
    description=(
        "Table-aware retrieval over SEC filings. Every answer is decomposed into "
        "claims, each carrying a verbatim snippet, chunk id and page number that "
        "are re-verified against the source text - and against the filer's own "
        "inline-XBRL tagging - before the answer is returned."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What was operating income in fiscal 2025?"])
    k: int | None = Field(default=None, ge=1, le=50)
    ticker: str | None = None
    fiscal_year: int | None = None
    form_type: str | None = None
    tables_only: bool = False

    def where(self) -> dict[str, Any] | None:
        filters: dict[str, Any] = {}
        if self.ticker:
            filters["ticker"] = self.ticker
        if self.fiscal_year:
            filters["fiscal_year"] = self.fiscal_year
        if self.form_type:
            filters["form_type"] = self.form_type
        if self.tables_only:
            filters["content_type"] = "table"
        return filters or None


class SearchRequest(AskRequest):
    question: str = Field(min_length=1, alias="query", examples=["goodwill impairment"])

    model_config = {"populate_by_name": True}


class IngestRequest(BaseModel):
    path: str = Field(description="File or directory on the server's filesystem")
    company: str | None = None
    ticker: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    pattern: str = "*"
    persist: bool = True

    def metadata(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("company", self.company),
                ("ticker", self.ticker),
                ("form_type", self.form_type),
                ("fiscal_year", self.fiscal_year),
            )
            if value is not None
        }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/health")
def health(
    pipeline: SecFilingRagPipeline = Depends(get_pipeline),
    auth: ApiKeyAuth = Depends(get_auth),
    limiter: RateLimiter = Depends(get_limiter),
) -> dict[str, Any]:
    """Unauthenticated on purpose: a load balancer has to be able to call it."""
    return {
        "status": "ok",
        **pipeline.describe(),
        "auth_enabled": auth.enabled,
        "rate_limit": limiter.limit if limiter.enabled else None,
    }


@app.get("/usage", dependencies=[Depends(guard)])
def usage(tracker: UsageTracker = Depends(get_usage)) -> dict[str, Any]:
    """Token spend and latency since start-up, with an estimated cost."""
    return tracker.snapshot()


@app.post("/search", dependencies=[Depends(guard)])
def search(
    request: SearchRequest, pipeline: SecFilingRagPipeline = Depends(get_pipeline)
) -> dict[str, Any]:
    """Retrieval only - no model call, no cost. Useful for tuning the index."""
    results = pipeline.retrieve(request.question, k=request.k, where=request.where())
    return {
        "query": request.question,
        "results": [
            {
                "chunk_id": sc.chunk_id,
                "score": round(sc.score, 6),
                "page": sc.chunk.page,
                "content_type": sc.chunk.content_type.value,
                "section": sc.chunk.section,
                "label": sc.chunk.citation_label(),
                "text": sc.chunk.text,
            }
            for sc in results
        ],
    }


@app.post("/ask", dependencies=[Depends(guard)])
def ask(
    request: AskRequest,
    pipeline: SecFilingRagPipeline = Depends(get_pipeline),
    tracker: UsageTracker = Depends(get_usage),
) -> dict[str, Any]:
    """Answer a question with verified citations.

    A guardrail block is returned as a 200 with `answerable: false` and the
    failure detail, not as an error: the caller asked a legitimate question and
    the honest response is that the answer could not be proven from the filing.
    """
    if pipeline.store.size == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No filings are indexed. POST /ingest first.",
        )

    started = time.perf_counter()
    try:
        result = pipeline.answer(request.question, k=request.k, where=request.where())
    except GenerationError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    tracker.record(result.usage, seconds=time.perf_counter() - started, blocked=result.blocked)
    return result.to_dict()


@app.post("/ingest", status_code=status.HTTP_201_CREATED, dependencies=[Depends(guard)])
def ingest(
    request: IngestRequest, pipeline: SecFilingRagPipeline = Depends(get_pipeline)
) -> dict[str, Any]:
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Path not found: {path}"
        )

    metadata = request.metadata()
    try:
        if path.is_dir():
            stats = pipeline.ingest_directory(path, pattern=request.pattern, **metadata)
        else:
            stats = [pipeline.ingest(path, **metadata)]
    except ValueError as exc:  # unsupported file type
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc

    if request.persist:
        pipeline.persist()

    return {
        "documents": stats,
        "chunks_indexed": pipeline.store.size,
        "xbrl_facts": len(pipeline.facts),
    }
