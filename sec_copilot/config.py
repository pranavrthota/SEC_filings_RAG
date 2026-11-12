"""Service configuration.

One frozen settings object is threaded through the pipeline rather than read
from the environment at point of use, so a deployment's behaviour is fully
described by a single serialisable value.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

# Opus 5 is the default; it is the model the guardrail prompt was tuned against.
DEFAULT_MODEL = "claude-opus-5"
# Used when SEC_COPILOT_PROVIDER=gemini and no model is named. A pinned version
# rather than the `gemini-flash-latest` alias: on the free tier the alias points
# at the busiest shared pool and returns a persistent 503, while pinned versions
# answer immediately. The cost of pinning is that Google eventually retires a
# version (gemini-2.5-flash closed to new keys during development), which fails
# loudly as a 404 naming its replacement - a better failure than an outage.
# Must match gemini_client.DEFAULT_MODEL; a literal rather than an import
# because the client imports this module. test_gemini.py guards the drift.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Load KEY=VALUE lines from a .env file into the environment.

    Existing variables win, so a real environment always overrides the file.
    Deliberately tiny - this exists so an API key can live in a gitignored file
    instead of being pasted into a shell, not to reimplement python-dotenv.
    """
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunk sizing, in characters (cheap, deterministic, model-agnostic)."""

    max_chars: int = 1400
    overlap_chars: int = 200
    # A table is split by rows only past this size; the header row is repeated
    # on every part so no figure is ever orphaned from its column label.
    table_max_chars: int = 3000
    min_chars: int = 60


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 8
    # Candidates pulled from each retriever before reciprocal-rank fusion.
    candidate_k: int = 30
    # Weights for reciprocal rank fusion (dense semantic vs. sparse BM25).
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_k: int = 60
    # Numeric questions ("what was total revenue") are answered by tables;
    # this multiplies the fused score of table chunks when the query looks
    # numeric. 1.0 disables the bias.
    table_boost: float = 1.25


@dataclass(frozen=True)
class GuardrailConfig:
    """How strictly an answer must be provable before it is returned."""

    # Refuse to return an answer whose citations cannot be verified verbatim.
    block_on_failed_citation: bool = True
    # Refuse to return an answer containing a figure not present in context.
    block_on_ungrounded_number: bool = True
    # Refuse an answer that attributes a real figure to the wrong fiscal period.
    # This is a distinct failure from a fabricated number and, in a multi-year
    # statement, a far more likely one.
    block_on_period_mismatch: bool = True
    # Relative tolerance when comparing an answer's figure to a source figure.
    # Rounding is legitimate ("133,050 million" -> "$133.05 billion"), so exact
    # equality would reject correct work.
    numeric_tolerance: float = 0.005
    # Fuzzy-match tolerance for verbatim snippet checking, 1.0 = exact after
    # whitespace normalisation.
    snippet_match_threshold: float = 0.9
    # Ignore small integers when numeric-grounding; years and one/two digit
    # counts appear in prose too often to be meaningful signals.
    numeric_min_digits: int = 3


@dataclass(frozen=True)
class GenerationConfig:
    # "claude" or "gemini". The guardrails are provider-independent, so this
    # changes which model writes the answer and nothing else.
    provider: str = "claude"
    model: str = DEFAULT_MODEL
    max_tokens: int = 16000
    effort: str = "high"  # low | medium | high | xhigh | max
    thinking: bool = True
    max_context_chars: int = 60000


@dataclass(frozen=True)
class Settings:
    """Root configuration object."""

    index_dir: Path = Path("index")
    # Extract inline-XBRL facts when ingesting EDGAR HTML. The filer's own
    # tagging carries exact periods, units and scales, which the numeric
    # guardrail uses as authoritative ground truth.
    extract_xbrl: bool = True
    embedding_backend: str = "auto"  # auto | sentence-transformers | hashing
    # Chosen by measurement, not reputation: on the three-filing eval corpus
    # mpnet reaches answerable@8 = 1.00 where MiniLM manages 0.75. See
    # `evals/retrieval_eval.py` and the table in the README.
    # BAAI/bge-small-en-v1.5 is the fast alternative (a@8 = 1.00 filtered, a
    # third of the size); MiniLM is materially worse and no longer the default.
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 512  # only used by the hashing backend
    vector_store: str = "in-memory"  # in-memory | chroma
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        """Build settings from SEC_COPILOT_* environment variables."""
        load_dotenv()
        provider = os.getenv("SEC_COPILOT_PROVIDER", "claude").lower()
        default_model = DEFAULT_MODEL if provider == "claude" else DEFAULT_GEMINI_MODEL
        base = cls(
            index_dir=Path(os.getenv("SEC_COPILOT_INDEX_DIR", "index")),
            embedding_backend=os.getenv("SEC_COPILOT_EMBEDDING_BACKEND", "auto"),
            embedding_model=os.getenv(
                "SEC_COPILOT_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
            ),
            vector_store=os.getenv("SEC_COPILOT_VECTOR_STORE", "in-memory"),
            generation=GenerationConfig(
                provider=provider,
                model=os.getenv("SEC_COPILOT_MODEL", default_model),
                effort=os.getenv("SEC_COPILOT_EFFORT", "high"),
            ),
        )
        return replace(base, **overrides) if overrides else base

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["index_dir"] = str(self.index_dir)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
