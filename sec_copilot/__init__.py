"""SEC filing RAG service - table-aware retrieval with verified citations."""

from .config import (
    ChunkingConfig,
    GenerationConfig,
    GuardrailConfig,
    RetrievalConfig,
    Settings,
)
from .models import (
    AnswerResult,
    Block,
    BlockType,
    Chunk,
    Citation,
    Claim,
    Document,
    GroundedAnswer,
    NumericStatus,
    ScoredChunk,
    Table,
    VerificationReport,
)
from .pipeline import SecFilingRagPipeline

__all__ = [
    "AnswerResult",
    "Block",
    "BlockType",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "Claim",
    "Document",
    "GenerationConfig",
    "GroundedAnswer",
    "GuardrailConfig",
    "NumericStatus",
    "RetrievalConfig",
    "ScoredChunk",
    "SecFilingRagPipeline",
    "Settings",
    "Table",
    "VerificationReport",
]

__version__ = "0.1.0"
