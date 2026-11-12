"""Answer generation against the Google Gemini API.

A peer of `ClaudeAnswerGenerator`, not a replacement: both implement
`AnswerGenerator`, both return a `GroundedAnswer`, and the citation, period and
scale guardrails apply identically to whichever produced the answer. Gemini has
a genuinely free tier, which makes it the practical choice for running the
pipeline end to end without a paid key.

Two things needed care:

* **Schema flattening.** Gemini's `responseSchema` is an OpenAPI subset. It
  rejects the `$ref`/`$defs` indirection pydantic emits for nested models, and
  ignores validation keywords like `minLength`. `to_gemini_schema` inlines the
  references and keeps only the fields Gemini honours, so `GroundedAnswer`
  remains the only representable response - which is what makes citation
  structural rather than a request the model may decline.

* **No SDK.** The call is plain HTTPS via the standard library, matching how
  BM25 and the hashing embedder are handled: an optional capability should not
  add a dependency for people who do not use it.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from pydantic import ValidationError

from ..config import GenerationConfig
from ..models import GroundedAnswer, ScoredChunk
from .claude_client import AnswerGenerator, GenerationError
from .prompts import SYSTEM_PROMPT, build_user_message, render_context

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.6-flash"

# Tried in order when the preferred model is unavailable. Ordered by capability,
# with the lite models last: a lite answer that verifies beats no answer, and
# the guardrail judges them all by the same standard either way.
FALLBACK_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
)

# Transient on a free tier: 429 is quota, 503 is the shared pool being busy,
# 500/502/504 are ordinary upstream blips. All clear on their own.
_RETRYABLE = {429, 500, 502, 503, 504}

# Keywords Gemini's Schema type understands. Everything else - title, default,
# $ref, additionalProperties, minLength - is dropped rather than sent, because
# an unknown key is a 400 rather than a warning.
_KEPT = {"type", "description", "enum", "items", "properties", "required", "nullable"}

_TYPES = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def to_gemini_schema(schema: dict[str, Any], defs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert a pydantic JSON schema into Gemini's OpenAPI subset.

    References are inlined because Gemini has no `$defs`; the nesting here is
    shallow and acyclic (answer -> claims -> citations), so inlining terminates.
    """
    defs = defs if defs is not None else schema.get("$defs", {})

    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        target = defs.get(name)
        if target is None:
            raise GenerationError(f"Cannot resolve schema reference {schema['$ref']!r}")
        return to_gemini_schema(target, defs)

    # `anyOf`/`allOf` appear for optional fields; take the first concrete branch.
    for combinator in ("anyOf", "allOf", "oneOf"):
        if combinator in schema:
            for branch in schema[combinator]:
                if branch.get("type") != "null":
                    return to_gemini_schema(branch, defs)

    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _KEPT:
            continue
        if key == "type":
            out["type"] = _TYPES.get(value, "STRING")
        elif key == "properties":
            out["properties"] = {k: to_gemini_schema(v, defs) for k, v in value.items()}
        elif key == "items":
            out["items"] = to_gemini_schema(value, defs)
        else:
            out[key] = value

    if out.get("type") == "OBJECT" and "properties" not in out:
        out["properties"] = {}
    return out


def _is_transient(error: Exception) -> bool:
    """True for an availability failure, as opposed to a bad request."""
    text = str(error)
    return (
        any(f"error {code}" in text for code in _RETRYABLE)
        or "404" in text
        or "connection failed" in text
        or "Could not reach" in text
    )


class GeminiAnswerGenerator(AnswerGenerator):
    """Gemini-backed generator using schema-constrained JSON output."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff_seconds: float = 2.0,
        fallbacks: Sequence[str] | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        # GenerationConfig defaults to a Claude model id; sending that to
        # Gemini is a 404, so fall back unless a Gemini model was named.
        named = model or self.config.model
        self.model = named if named and "gemini" in named.lower() else DEFAULT_MODEL
        self._api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        # The preferred model first, then the rest, without duplicates.
        self.candidates: tuple[str, ...] = tuple(
            dict.fromkeys((self.model, *(fallbacks or FALLBACK_MODELS)))
        )

    @property
    def api_key(self) -> str:
        key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise GenerationError(
                "No Gemini API key. Set GEMINI_API_KEY in the environment or in a "
                ".env file (get one free at https://aistudio.google.com/apikey)."
            )
        return key

    def generate(
        self,
        question: str,
        results: Sequence[ScoredChunk],
        *,
        extra_instruction: str | None = None,
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        context = render_context(results, max_chars=self.config.max_context_chars)
        user_message = build_user_message(question, context)
        if extra_instruction:
            user_message = f"{user_message}\n\n{extra_instruction}"

        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": to_gemini_schema(GroundedAnswer.model_json_schema()),
                "maxOutputTokens": self.config.max_tokens,
                # Deterministic: an auditable answer should not vary run to run.
                "temperature": 0.0,
            },
        }

        data, model = self._post_with_fallback(payload)
        return self._parse(data, model)

    def _post_with_fallback(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Try each candidate model, moving on when one is unavailable.

        Free-tier capacity is per model and moves around: during development
        `gemini-flash-latest` and `gemini-3.8-flash` both returned a persistent
        503 while three other models answered the identical payload in seconds.
        Retrying one model harder does not help; trying the next one does.
        """
        last: Exception | None = None
        for model in self.candidates:
            try:
                data = self._post(f"{API_ROOT}/models/{model}:generateContent", payload)
                if model != self.model:
                    logger.info("Gemini: %s unavailable, answered with %s", self.model, model)
                return data, model
            except GenerationError as exc:
                if not _is_transient(exc):
                    raise
                last = exc
                logger.info("Gemini: %s unavailable, trying the next model", model)

        assert last is not None
        raise GenerationError(
            f"Every Gemini model was unavailable ({', '.join(self.candidates)}). "
            f"The free tier is congested rather than misconfigured - retry shortly. "
            f"Last error: {last}"
        )

    # -- internals --------------------------------------------------------- #

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with backoff on the transient failures a free tier actually hits.

        503 (model overloaded) and 429 (quota) are both routine on the free
        tier and both clear on their own, so they are retried; a bad key or a
        malformed request never will be, and fails immediately.
        """
        body = json.dumps(payload).encode("utf-8")
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                if exc.code in (401, 403):
                    raise GenerationError(
                        f"Gemini rejected the API key ({exc.code}). {detail}"
                    ) from exc
                if exc.code not in _RETRYABLE:
                    raise GenerationError(f"Gemini API error {exc.code}: {detail}") from exc
                last = GenerationError(f"Gemini API error {exc.code}: {detail}")
            except urllib.error.URLError as exc:
                last = GenerationError(f"Could not reach the Gemini API: {exc.reason}")
            except (TimeoutError, OSError) as exc:
                # A reset or dropped connection mid-read is not wrapped by
                # urllib and reached the caller as an unhandled crash.
                last = GenerationError(f"Gemini connection failed: {exc!r}")

            if attempt < self.max_retries:
                delay = self.backoff_seconds * (2**attempt)
                logger.info(
                    "Gemini call failed (%s); retrying in %.1fs [%d/%d]",
                    last,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)

        assert last is not None
        raise last

    def _parse(
        self, data: dict[str, Any], model: str | None = None
    ) -> tuple[GroundedAnswer, dict[str, Any]]:
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "generator": "gemini",
            "model": model or self.model,
            "input_tokens": usage_meta.get("promptTokenCount"),
            "output_tokens": usage_meta.get("candidatesTokenCount"),
            "total_tokens": usage_meta.get("totalTokenCount"),
        }

        candidates = data.get("candidates") or []
        if not candidates:
            # A prompt-level block returns no candidate at all.
            reason = (data.get("promptFeedback") or {}).get("blockReason", "unknown")
            return (
                GroundedAnswer(
                    answerable=False, refusal_reason=f"Gemini blocked the request: {reason}"
                ),
                usage,
            )

        candidate = candidates[0]
        finish = candidate.get("finishReason", "")
        usage["finish_reason"] = finish

        if finish in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
            return (
                GroundedAnswer(
                    answerable=False, refusal_reason=f"Gemini declined to answer ({finish})."
                ),
                usage,
            )

        text = "".join(
            part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
        ).strip()

        if not text:
            if finish == "MAX_TOKENS":
                raise GenerationError(
                    "Gemini hit the output token limit before emitting an answer. "
                    "Raise generation.max_tokens or lower retrieval.top_k."
                )
            raise GenerationError(f"Gemini returned no content (finishReason={finish!r}).")

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GenerationError(
                f"Gemini returned malformed JSON despite the response schema: {text[:200]!r}"
            ) from exc

        try:
            return GroundedAnswer.model_validate(parsed), usage
        except ValidationError as exc:
            # Gemini's schema subset cannot express every constraint the model
            # declares (length keywords are dropped), so a shape error here is
            # a provider limitation rather than a bug. Report which field failed
            # instead of surfacing a bare stack trace.
            problems = "; ".join(
                ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
                for error in exc.errors()[:3]
            )
            raise GenerationError(
                f"Gemini output did not match GroundedAnswer "
                f"({exc.error_count()} error(s)): {problems}"
            ) from exc
