"""Answer generation against the Claude API.

The generator is an ABC with one Claude implementation, so the pipeline and its
tests never depend on a live API key. Two things about the Claude call are
load-bearing for the zero-hallucination goal:

* `messages.parse` with a pydantic `output_format` makes `GroundedAnswer` the
  only representable response. A prose answer with no `claims` array cannot be
  returned, so citation is structural rather than a request the model may
  quietly decline.
* The system prompt is a frozen constant with a cache breakpoint on it, so
  repeated questions against the same filing pay for the guardrail preamble
  once.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..config import GenerationConfig
from ..models import GroundedAnswer, ScoredChunk
from .prompts import SYSTEM_PROMPT, build_user_message, render_context

logger = logging.getLogger(__name__)


class AnswerGenerator(ABC):
    """Turns a question plus retrieved passages into a `GroundedAnswer`."""

    @abstractmethod
    def generate(
        self,
        question: str,
        results: Sequence[ScoredChunk],
        *,
        extra_instruction: str | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        """Return the answer and a usage/telemetry dict."""


class ClaudeAnswerGenerator(AnswerGenerator):
    """Claude-backed generator using structured outputs."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self._client = client

    @property
    def client(self) -> Any:
        """Lazily constructed SDK client.

        Deferred so importing the package - or running the ingest path - does
        not require credentials to be present.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def generate(
        self,
        question: str,
        results: Sequence[ScoredChunk],
        *,
        extra_instruction: str | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        import anthropic

        context = render_context(results, max_chars=self.config.max_context_chars)
        user_message = build_user_message(question, context)
        if extra_instruction:
            user_message = f"{user_message}\n\n{extra_instruction}"

        request: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            # A frozen system prompt with a cache breakpoint: every question
            # against the corpus shares this prefix.
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_message}],
            "output_format": GroundedAnswer,
            "output_config": {"effort": self.config.effort},
        }
        if self.config.thinking:
            # Adaptive thinking: reconciling figures across tables is exactly
            # the kind of work that degrades without it.
            request["thinking"] = {"type": "adaptive"}

        try:
            response = self.client.messages.parse(**request)
        except anthropic.BadRequestError as exc:
            raise GenerationError(f"Claude rejected the request: {exc.message}") from exc
        except anthropic.AuthenticationError as exc:
            raise GenerationError(
                "Claude authentication failed. Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after", "60")
            raise GenerationError(f"Rate limited by Claude; retry after {retry_after}s.") from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError("Could not reach the Claude API.") from exc

        usage = self._usage(response)

        # A safety refusal is not an answer - surface it as an unanswerable
        # result rather than reading `content` that may not be there.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            reason = getattr(detail, "explanation", None) or "the model declined this request"
            return (
                GroundedAnswer(answerable=False, refusal_reason=f"Model refusal: {reason}"),
                usage,
            )

        answer = response.parsed_output
        if answer is None:
            raise GenerationError(
                f"Claude returned no parsed output (stop_reason={response.stop_reason})."
            )
        return answer, usage

    @staticmethod
    def _usage(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "model": getattr(response, "model", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "request_id": getattr(response, "_request_id", None),
        }


class GenerationError(RuntimeError):
    """Raised when the model could not produce a usable answer."""


class EchoAnswerGenerator(AnswerGenerator):
    """A deterministic offline generator, for tests and for demos without a key.

    It quotes the first line of the top passage. That is a deliberately dumb
    answer, but it is a *correctly cited* one - which makes it useful for
    exercising the guardrail path end to end.
    """

    def generate(
        self,
        question: str,
        results: Sequence[ScoredChunk],
        *,
        extra_instruction: str | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        from ..models import Citation, Claim

        if not results:
            return (
                GroundedAnswer(
                    answerable=False,
                    refusal_reason="No filing excerpts were retrieved for this question.",
                ),
                {"generator": "echo"},
            )

        top = results[0].chunk
        snippet = next(
            (
                line.strip()
                for line in top.text.splitlines()
                # Skip the bracketed provenance header the chunker prepends.
                if len(line.strip()) > 20 and not line.lstrip().startswith("[")
            ),
            top.text[:200].strip(),
        )
        return (
            GroundedAnswer(
                answerable=True,
                answer=f"Most relevant passage for {question!r}: {snippet}",
                claims=[
                    Claim(
                        text=snippet,
                        citations=[
                            Citation(chunk_id=top.chunk_id, page=top.page, snippet=snippet)
                        ],
                    )
                ],
            ),
            {"generator": "echo"},
        )
