"""Authentication, rate limiting and usage accounting."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sec_copilot.api import app, get_auth, get_limiter, get_pipeline, get_usage
from sec_copilot.service import ApiKeyAuth, RateLimiter, UsageTracker, client_key


class _Request:
    """Minimal stand-in for a Starlette request."""

    def __init__(self, headers=None, host="1.2.3.4"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})() if host else None


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


def test_auth_is_disabled_when_no_keys_are_configured():
    auth = ApiKeyAuth(keys=frozenset())
    assert not auth.enabled
    auth.check(_Request())  # must not raise


def test_a_valid_key_is_accepted_and_a_wrong_one_rejected():
    auth = ApiKeyAuth(keys=frozenset({"secret-a", "secret-b"}))
    auth.check(_Request({"X-API-Key": "secret-b"}))

    with pytest.raises(HTTPException) as exc:
        auth.check(_Request({"X-API-Key": "secret-c"}))
    assert exc.value.status_code == 401


def test_a_missing_key_is_rejected_when_auth_is_enabled():
    auth = ApiKeyAuth(keys=frozenset({"secret"}))
    with pytest.raises(HTTPException) as exc:
        auth.check(_Request())
    assert exc.value.status_code == 401


def test_keys_are_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("SEC_COPILOT_API_KEYS", " one , two ,, ")
    auth = ApiKeyAuth.from_env()
    assert auth.keys == {"one", "two"}


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


def test_requests_are_allowed_up_to_the_limit_then_refused():
    limiter = RateLimiter(limit=3, window_seconds=60)
    for _ in range(3):
        limiter.check("client")

    with pytest.raises(HTTPException) as exc:
        limiter.check("client")
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_clients_are_limited_independently():
    limiter = RateLimiter(limit=1, window_seconds=60)
    limiter.check("a")
    limiter.check("b")  # b has its own budget
    with pytest.raises(HTTPException):
        limiter.check("a")


def test_the_window_slides(monkeypatch):
    limiter = RateLimiter(limit=1, window_seconds=10)
    clock = {"now": 1000.0}
    monkeypatch.setattr("sec_copilot.service.time.monotonic", lambda: clock["now"])

    limiter.check("c")
    with pytest.raises(HTTPException):
        limiter.check("c")

    clock["now"] += 11
    limiter.check("c")  # the earlier hit has aged out


def test_a_zero_limit_disables_limiting():
    limiter = RateLimiter(limit=0)
    assert not limiter.enabled
    for _ in range(100):
        limiter.check("anyone")


def test_the_api_key_identifies_the_client_ahead_of_the_address():
    assert client_key(_Request({"X-API-Key": "abcdefghijkl"})).startswith("key:")
    assert client_key(_Request()) == "ip:1.2.3.4"
    assert client_key(_Request(host=None)) == "ip:unknown"


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #


def test_usage_accumulates_tokens_and_cost():
    tracker = UsageTracker()
    tracker.record(
        {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 0},
        seconds=2.0,
        blocked=False,
    )
    snapshot = tracker.snapshot()

    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] == 1_000_000
    # 1M input at $5 + 0.1M output at $25 = $7.50
    assert snapshot["estimated_cost_usd"] == pytest.approx(7.50)
    assert snapshot["mean_latency_seconds"] == pytest.approx(2.0)


def test_usage_counts_both_attempts_of_a_retried_answer():
    tracker = UsageTracker()
    tracker.record(
        {
            "first_attempt": {"input_tokens": 100, "output_tokens": 10},
            "retry": {"input_tokens": 120, "output_tokens": 12},
        },
        seconds=1.0,
        blocked=True,
    )
    snapshot = tracker.snapshot()
    assert snapshot["input_tokens"] == 220
    assert snapshot["output_tokens"] == 22
    assert snapshot["blocked_by_guardrail"] == 1


def test_usage_tolerates_a_generator_that_reports_nothing():
    tracker = UsageTracker()
    tracker.record({"generator": "echo"}, seconds=0.1, blocked=False)
    tracker.record(None, seconds=0.1, blocked=False)
    assert tracker.snapshot()["requests"] == 2


# --------------------------------------------------------------------------- #
# Wired into the app
# --------------------------------------------------------------------------- #


@pytest.fixture
def secured_client(pipeline):
    # One instance each, not a factory: the limiter and the tracker are stateful
    # across requests, which is the whole point of them.
    auth = ApiKeyAuth(keys=frozenset({"good-key"}))
    limiter = RateLimiter(limit=3, window_seconds=60)
    tracker = UsageTracker()

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_auth] = lambda: auth
    app.dependency_overrides[get_limiter] = lambda: limiter
    app.dependency_overrides[get_usage] = lambda: tracker
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_protected_routes_require_a_key(secured_client):
    assert secured_client.post("/search", json={"query": "net sales"}).status_code == 401
    ok = secured_client.post(
        "/search", json={"query": "net sales"}, headers={"X-API-Key": "good-key"}
    )
    assert ok.status_code == 200


def test_health_stays_open_for_load_balancers(secured_client):
    body = secured_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["auth_enabled"] is True
    assert body["rate_limit"] == 3


def test_the_rate_limit_is_enforced_over_http(secured_client):
    headers = {"X-API-Key": "good-key"}
    codes = [
        secured_client.post("/search", json={"query": "net sales"}, headers=headers).status_code
        for _ in range(4)
    ]
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429


def test_usage_is_reported_after_asking(secured_client):
    headers = {"X-API-Key": "good-key"}
    answered = secured_client.post(
        "/ask", json={"question": "operating income 2023"}, headers=headers
    )
    assert answered.status_code == 200
    snapshot = secured_client.get("/usage", headers=headers).json()
    assert snapshot["requests"] == 1
    assert "estimated_cost_usd" in snapshot
