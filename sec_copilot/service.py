"""Service concerns: authentication, rate limiting, and request telemetry.

Kept separate from `api.py` so the routes stay readable and so each policy can
be tested on its own. All three are configured from the environment and are
inert unless switched on, which keeps local development friction-free while
making a deployment's posture explicit.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


@dataclass
class ApiKeyAuth:
    """Shared-secret auth via the `X-API-Key` header.

    Disabled when no keys are configured, so a local run needs no setup. Keys
    are compared with `secrets.compare_digest` - a plain `==` on a secret leaks
    its prefix through timing, and this is the one place that matters.
    """

    keys: frozenset[str] = frozenset()
    header: str = "X-API-Key"

    @classmethod
    def from_env(cls, variable: str = "SEC_COPILOT_API_KEYS") -> "ApiKeyAuth":
        raw = os.getenv(variable, "")
        keys = frozenset(k.strip() for k in raw.split(",") if k.strip())
        if not keys:
            logger.warning(
                "%s is unset - the API is unauthenticated. Set it to a comma-separated "
                "list of keys before exposing this service.",
                variable,
            )
        return cls(keys=keys)

    @property
    def enabled(self) -> bool:
        return bool(self.keys)

    def check(self, request: Request) -> None:
        if not self.enabled:
            return
        presented = request.headers.get(self.header, "")
        if not any(secrets.compare_digest(presented, key) for key in self.keys):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing or invalid {self.header}",
                headers={"WWW-Authenticate": self.header},
            )


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


@dataclass
class RateLimiter:
    """Fixed-window-free sliding log limiter, per client.

    A sliding log rather than a fixed window because a fixed window lets a
    caller fire two full quotas either side of a boundary - which, for an
    endpoint that spends money on model calls, is the case worth preventing.
    Per-process and in-memory: correct for one instance, and the point at which
    you outgrow it is the point at which you want Redis.
    """

    limit: int = 60
    window_seconds: float = 60.0
    _hits: dict[str, deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    @classmethod
    def from_env(cls) -> "RateLimiter":
        return cls(
            limit=int(os.getenv("SEC_COPILOT_RATE_LIMIT", "60")),
            window_seconds=float(os.getenv("SEC_COPILOT_RATE_WINDOW", "60")),
        )

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def check(self, client: str) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(hits[0] + self.window_seconds - now) + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit of {self.limit} requests per "
                    f"{self.window_seconds:.0f}s exceeded",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def client_key(request: Request) -> str:
    """Identify the caller: API key if present, else peer address.

    Keying on the API key rather than the address means one credential cannot
    multiply its quota across source addresses, and callers behind a shared NAT
    are not throttled as one.
    """
    key = request.headers.get("X-API-Key")
    if key:
        return f"key:{key[:8]}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


# --------------------------------------------------------------------------- #
# Telemetry
# --------------------------------------------------------------------------- #


@dataclass
class UsageTracker:
    """Running totals of what the service has spent.

    Token counts come back from the Claude response; the dollar figure is
    computed from the published Opus 5 rates, so it is an estimate for
    orientation rather than a bill.
    """

    #: USD per million tokens, Claude Opus 5.
    input_per_mtok: float = 5.0
    output_per_mtok: float = 25.0
    cache_read_per_mtok: float = 0.50

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    blocked: int = 0
    total_seconds: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def record(self, usage: dict | None, *, seconds: float, blocked: bool) -> None:
        with self._lock:
            self.requests += 1
            self.total_seconds += seconds
            self.blocked += int(blocked)
            for entry in self._flatten(usage):
                self.input_tokens += int(entry.get("input_tokens") or 0)
                self.output_tokens += int(entry.get("output_tokens") or 0)
                self.cache_read_tokens += int(entry.get("cache_read_input_tokens") or 0)

    @staticmethod
    def _flatten(usage: dict | None) -> list[dict]:
        """A retried answer nests one usage dict per attempt."""
        if not usage:
            return []
        if any(isinstance(v, dict) for v in usage.values()):
            return [v for v in usage.values() if isinstance(v, dict)]
        return [usage]

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.input_tokens / 1e6 * self.input_per_mtok
            + self.output_tokens / 1e6 * self.output_per_mtok
            + self.cache_read_tokens / 1e6 * self.cache_read_per_mtok
        )

    def snapshot(self) -> dict:
        with self._lock:
            requests = self.requests or 1
            return {
                "requests": self.requests,
                "blocked_by_guardrail": self.blocked,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_input_tokens": self.cache_read_tokens,
                "cache_hit_ratio": round(
                    self.cache_read_tokens / max(1, self.input_tokens + self.cache_read_tokens),
                    4,
                ),
                "estimated_cost_usd": round(self.estimated_cost_usd, 4),
                "estimated_cost_per_request_usd": round(
                    self.estimated_cost_usd / requests, 5
                ),
                "mean_latency_seconds": round(self.total_seconds / requests, 3),
            }
