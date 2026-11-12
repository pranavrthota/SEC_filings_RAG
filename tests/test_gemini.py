"""Gemini generator tests.

No network: every HTTP call is stubbed. What is tested is the part that
actually broke in practice - schema flattening, model fallback under a
free-tier 503, and turning provider quirks into clean failures rather than
stack traces.
"""

from __future__ import annotations

import json

import pytest

from sec_copilot.config import GenerationConfig
from sec_copilot.generation import build_generator
from sec_copilot.generation.claude_client import GenerationError
from sec_copilot.generation.gemini_client import (
    FALLBACK_MODELS,
    GeminiAnswerGenerator,
    to_gemini_schema,
)
from sec_copilot.models import BlockType, Chunk, GroundedAnswer, ScoredChunk

TABLE = (
    "[Acme | 10-K | Statements Of Operations]\n"
    "| Line item | FY2023 | FY2022 |\n| --- | --- | --- |\n"
    "| Operating income | 2,879 | 2,113 |\n"
)


@pytest.fixture
def sources():
    return [ScoredChunk(chunk=Chunk("c1", "acme", TABLE, 4, BlockType.TABLE), score=1.0)]


def _reply(answer: dict, tokens: tuple[int, int] = (100, 20)) -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"text": json.dumps(answer)}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": tokens[0], "candidatesTokenCount": tokens[1]},
    }


GOOD = {
    "answerable": True,
    "answer": "Operating income was 2,879 million in FY2023.",
    "claims": [
        {
            "text": "Operating income was 2,879 million in FY2023.",
            "citations": [
                {"chunk_id": "c1", "page": 4, "snippet": "| Operating income | 2,879 | 2,113 |"}
            ],
            "derived_from": [],
        }
    ],
    "refusal_reason": "",
}


# --------------------------------------------------------------------------- #
# Schema flattening
# --------------------------------------------------------------------------- #


def test_nested_models_are_inlined_because_gemini_has_no_defs():
    schema = to_gemini_schema(GroundedAnswer.model_json_schema())
    text = json.dumps(schema)

    assert "$ref" not in text and "$defs" not in text
    citation = schema["properties"]["claims"]["items"]["properties"]["citations"]["items"]
    assert citation["properties"]["chunk_id"]["type"] == "STRING"
    assert citation["properties"]["page"]["type"] == "INTEGER"


def test_types_are_uppercased_and_unsupported_keywords_dropped():
    schema = to_gemini_schema(GroundedAnswer.model_json_schema())
    text = json.dumps(schema)

    assert schema["type"] == "OBJECT"
    assert schema["properties"]["answerable"]["type"] == "BOOLEAN"
    # Gemini rejects unknown keys outright rather than ignoring them.
    for unsupported in ("additionalProperties", "minLength", "maxLength", "title", "default"):
        assert unsupported not in text


def test_descriptions_survive_because_they_instruct_the_model():
    schema = to_gemini_schema(GroundedAnswer.model_json_schema())
    citation = schema["properties"]["claims"]["items"]["properties"]["citations"]["items"]
    assert "verbatim" in citation["properties"]["snippet"]["description"].lower()


def test_optional_fields_resolve_to_their_concrete_branch():
    flattened = to_gemini_schema(
        {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "maybe"}
    )
    assert flattened["type"] == "STRING"


def test_an_unresolvable_reference_is_reported_clearly():
    with pytest.raises(GenerationError, match="Cannot resolve schema reference"):
        to_gemini_schema({"$ref": "#/$defs/Missing"}, {})


# --------------------------------------------------------------------------- #
# Model selection and fallback
# --------------------------------------------------------------------------- #


def test_a_claude_model_id_is_not_sent_to_gemini():
    # GenerationConfig defaults to claude-opus-5; sending it would be a 404.
    generator = GeminiAnswerGenerator(GenerationConfig(provider="gemini"))
    assert "gemini" in generator.model
    assert generator.candidates[0] == generator.model


def test_a_named_gemini_model_leads_the_candidate_list():
    generator = GeminiAnswerGenerator(GenerationConfig(provider="gemini", model="gemini-3.6-flash"))
    assert generator.candidates[0] == "gemini-3.6-flash"
    assert len(set(generator.candidates)) == len(generator.candidates), "no duplicates"
    assert set(FALLBACK_MODELS) <= set(generator.candidates)


def test_an_unavailable_model_falls_through_to_the_next(monkeypatch, sources):
    """The failure that actually happened: one model 503s, another answers."""
    tried: list[str] = []

    def fake_post(self, url, payload):
        model = url.rsplit("/", 1)[-1].split(":")[0]
        tried.append(model)
        if len(tried) < 3:
            raise GenerationError("Gemini API error 503: UNAVAILABLE")
        return _reply(GOOD)

    monkeypatch.setattr(GeminiAnswerGenerator, "_post", fake_post)
    generator = GeminiAnswerGenerator(GenerationConfig(provider="gemini"), api_key="x")

    answer, usage = generator.generate("q", sources)
    assert answer.answerable
    assert len(tried) == 3, "moved on after each unavailable model"
    assert usage["model"] == tried[-1], "usage reports the model that actually answered"


def test_a_bad_request_is_not_retried_across_models(monkeypatch, sources):
    calls: list[str] = []

    def fake_post(self, url, payload):
        calls.append(url)
        raise GenerationError("Gemini API error 400: INVALID_ARGUMENT")

    monkeypatch.setattr(GeminiAnswerGenerator, "_post", fake_post)
    generator = GeminiAnswerGenerator(api_key="x")

    with pytest.raises(GenerationError, match="400"):
        generator.generate("q", sources)
    assert len(calls) == 1, "a malformed request will never succeed on another model"


def test_exhausting_every_model_says_so_plainly(monkeypatch, sources):
    monkeypatch.setattr(
        GeminiAnswerGenerator,
        "_post",
        lambda self, url, payload: (_ for _ in ()).throw(
            GenerationError("Gemini API error 503: UNAVAILABLE")
        ),
    )
    generator = GeminiAnswerGenerator(api_key="x")
    with pytest.raises(GenerationError, match="Every Gemini model was unavailable"):
        generator.generate("q", sources)


# --------------------------------------------------------------------------- #
# Response handling
# --------------------------------------------------------------------------- #


def test_a_well_formed_reply_becomes_a_grounded_answer(monkeypatch, sources):
    monkeypatch.setattr(GeminiAnswerGenerator, "_post", lambda self, url, payload: _reply(GOOD))
    answer, usage = GeminiAnswerGenerator(api_key="x").generate("q", sources)

    assert answer.claims[0].citations[0].chunk_id == "c1"
    assert usage["input_tokens"] == 100 and usage["output_tokens"] == 20
    assert usage["generator"] == "gemini"


def test_an_over_long_snippet_is_truncated_rather_than_rejected(monkeypatch, sources):
    """Gemini cannot enforce maxLength, so a long quote must not lose the answer."""
    long_answer = json.loads(json.dumps(GOOD))
    long_answer["claims"][0]["citations"][0]["snippet"] = "x" * 900

    monkeypatch.setattr(
        GeminiAnswerGenerator, "_post", lambda self, url, payload: _reply(long_answer)
    )
    answer, _ = GeminiAnswerGenerator(api_key="x").generate("q", sources)
    assert len(answer.claims[0].citations[0].snippet) == 400


def test_a_safety_block_becomes_an_unanswerable_result(monkeypatch, sources):
    monkeypatch.setattr(
        GeminiAnswerGenerator,
        "_post",
        lambda self, url, payload: {"candidates": [{"finishReason": "SAFETY", "content": {}}]},
    )
    answer, _ = GeminiAnswerGenerator(api_key="x").generate("q", sources)
    assert not answer.answerable
    assert "declined" in answer.refusal_reason.lower()


def test_a_prompt_level_block_becomes_an_unanswerable_result(monkeypatch, sources):
    monkeypatch.setattr(
        GeminiAnswerGenerator,
        "_post",
        lambda self, url, payload: {"promptFeedback": {"blockReason": "OTHER"}},
    )
    answer, _ = GeminiAnswerGenerator(api_key="x").generate("q", sources)
    assert not answer.answerable


def test_hitting_the_output_limit_is_reported_as_such(monkeypatch, sources):
    monkeypatch.setattr(
        GeminiAnswerGenerator,
        "_post",
        lambda self, url, payload: {
            "candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]
        },
    )
    with pytest.raises(GenerationError, match="output token limit"):
        GeminiAnswerGenerator(api_key="x").generate("q", sources)


def test_malformed_json_is_reported_not_raised_raw(monkeypatch, sources):
    monkeypatch.setattr(
        GeminiAnswerGenerator,
        "_post",
        lambda self, url, payload: {
            "candidates": [{"content": {"parts": [{"text": "{not json"}]}, "finishReason": "STOP"}]
        },
    )
    with pytest.raises(GenerationError, match="malformed JSON"):
        GeminiAnswerGenerator(api_key="x").generate("q", sources)


def test_a_shape_mismatch_names_the_offending_field(monkeypatch, sources):
    broken = {"answerable": True, "answer": "x", "claims": [{"text": "y"}]}  # no citations
    monkeypatch.setattr(GeminiAnswerGenerator, "_post", lambda self, url, payload: _reply(broken))
    with pytest.raises(GenerationError, match="citations"):
        GeminiAnswerGenerator(api_key="x").generate("q", sources)


def test_a_missing_key_is_a_clear_message_not_a_401(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="No Gemini API key"):
        _ = GeminiAnswerGenerator().api_key


# --------------------------------------------------------------------------- #
# Provider switching
# --------------------------------------------------------------------------- #


def test_the_provider_selects_the_generator():
    from sec_copilot.generation.claude_client import ClaudeAnswerGenerator, EchoAnswerGenerator

    assert isinstance(build_generator(GenerationConfig(provider="claude")), ClaudeAnswerGenerator)
    assert isinstance(build_generator(GenerationConfig(provider="echo")), EchoAnswerGenerator)
    assert isinstance(build_generator(GenerationConfig(provider="gemini")), GeminiAnswerGenerator)


def test_the_two_gemini_defaults_do_not_drift():
    # config cannot import the client (the client imports config), so the
    # default is duplicated as a literal and guarded here instead.
    from sec_copilot.config import DEFAULT_GEMINI_MODEL
    from sec_copilot.generation.gemini_client import DEFAULT_MODEL

    assert DEFAULT_GEMINI_MODEL == DEFAULT_MODEL


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown generation provider"):
        build_generator(GenerationConfig(provider="llama"))


def test_guardrails_apply_identically_whichever_provider_answered(monkeypatch, sources):
    """The point of the ABC: verification does not know or care who generated."""
    from sec_copilot.generation.verifier import CitationVerifier

    fabricated = json.loads(json.dumps(GOOD))
    fabricated["claims"][0]["text"] = "Operating income was 9,999 million in FY2023."
    fabricated["answer"] = fabricated["claims"][0]["text"]

    monkeypatch.setattr(
        GeminiAnswerGenerator, "_post", lambda self, url, payload: _reply(fabricated)
    )
    answer, _ = GeminiAnswerGenerator(api_key="x").generate("q", sources)

    report = CitationVerifier().verify(answer, sources)
    assert not report.passed
    assert "9,999" in [n.value for n in report.ungrounded_numbers]
