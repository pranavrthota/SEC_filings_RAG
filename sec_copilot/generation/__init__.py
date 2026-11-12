"""Answer generation and the guardrails that check it."""

from __future__ import annotations

from ..config import GenerationConfig
from .claude_client import AnswerGenerator, ClaudeAnswerGenerator, EchoAnswerGenerator, GenerationError


def build_generator(config: GenerationConfig) -> AnswerGenerator:
    """Pick a generator by provider.

    The provider is the only thing that changes: every generator returns a
    `GroundedAnswer`, and the citation, period and scale guardrails run
    identically over whichever produced it.
    """
    provider = (config.provider or "claude").lower()
    if provider == "claude":
        return ClaudeAnswerGenerator(config)
    if provider == "gemini":
        from .gemini_client import GeminiAnswerGenerator

        return GeminiAnswerGenerator(config)
    if provider == "echo":
        return EchoAnswerGenerator()
    raise ValueError(
        f"Unknown generation provider {config.provider!r}. Use claude, gemini or echo."
    )


__all__ = [
    "AnswerGenerator",
    "ClaudeAnswerGenerator",
    "EchoAnswerGenerator",
    "GenerationError",
    "build_generator",
]
