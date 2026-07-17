"""Minimal in-memory rate limiting for FastAPI routes.

This is a small fixed-window limiter over a process-local dict — no Redis,
no external dependency. That means it is **per-process**: if this app is
ever run with multiple uvicorn workers (or multiple instances behind a load
balancer), each process enforces its own independent limit rather than
sharing a single global count, so the *effective* limit for a caller scales
with the number of processes. That's an accepted tradeoff for this in-memory
demo app (storage elsewhere in the app is already per-process/in-memory) —
it still meaningfully blocks single-process abuse and brute force without
adding an external moving part.

Per-IP keying (`_client_ip` below) trusts `X-Forwarded-For` only when
`settings.TRUST_PROXY_HEADERS` is explicitly enabled, and even then only its
rightmost entry. See `_client_ip`'s docstring for why: an untrusted
`X-Forwarded-For` lets a caller manufacture a fresh rate-limit bucket on
every request, silently defeating IP-based limiting.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Annotated, Callable

from fastapi import Depends, HTTPException, Request

from app.auth.dependencies import get_current_user
from app.config import settings
from app.models.user_model import User

# Hard cap on distinct keys tracked at once. Without this, an attacker who
# varies their identity per request (e.g. a spoofed X-Forwarded-For value)
# could grow the tracking dict without bound, trading a rate-limit hole for
# a memory-leak hole. When the cap is hit, the least-recently-touched key is
# evicted to make room for the new one.
DEFAULT_MAX_TRACKED_KEYS = 10_000


@dataclass
class _Window:
    """Fixed-window counter state for a single key."""

    count: int
    window_start: float


class FixedWindowRateLimiter:
    """A minimal fixed-window rate limiter keyed by an arbitrary string.

    Each key gets its own rolling window: the first hit for a key opens a
    `window_seconds`-wide window, subsequent hits within that window
    increment its counter, and the window resets once it expires. The limit
    itself is passed in on each `check()` call (rather than fixed at
    construction) so callers can read it from live config (e.g.
    `app.config.settings`) and tests can adjust it at runtime.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        max_tracked_keys: int = DEFAULT_MAX_TRACKED_KEYS,
    ) -> None:
        """Initialize the limiter.

        Args:
            window_seconds: Width of each key's rolling window, in seconds.
            clock: Monotonic time source. Defaults to `time.monotonic` so
                wall-clock changes (NTP adjustments, DST, manual clock
                changes) can't confuse the window. Injectable for tests via
                `set_clock`.
            max_tracked_keys: Memory bound — the most distinct keys tracked
                at once before the oldest-touched is evicted.
        """
        self._window_seconds = window_seconds
        self._clock = clock
        self._max_tracked_keys = max_tracked_keys
        self._buckets: "OrderedDict[str, _Window]" = OrderedDict()

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Swap the time source.

        Intended for tests that need to fast-forward past a window without `time.sleep()`.

        Args:
            clock: The replacement zero-argument time source.
        """
        self._clock = clock

    def reset(self) -> None:
        """Clear all tracked state. Intended for test isolation between tests."""
        self._buckets.clear()

    @property
    def tracked_key_count(self) -> int:
        """Number of distinct keys currently tracked (for tests/inspection)."""
        return len(self._buckets)

    def check(self, key: str, limit_per_minute: int) -> None:
        """Record one hit for `key`, raising 429 if it exceeds `limit_per_minute`.

        Args:
            key: Identity string for the caller (e.g. an IP address or user id).
            limit_per_minute: Max hits allowed per `window_seconds`. A value
                of 0 (or negative) disables limiting entirely — useful for
                tests and local dev.

        Raises:
            HTTPException: 429 with a `Retry-After` header (seconds) when the
                caller has exceeded the limit for the current window.
        """
        if limit_per_minute <= 0:
            return

        now = self._clock()

        window = self._buckets.get(key)
        if window is None or now - window.window_start >= self._window_seconds:
            window = _Window(count=0, window_start=now)

        window.count += 1
        self._buckets[key] = window
        self._buckets.move_to_end(key)

        # Bound memory: evict the least-recently-touched key(s) once we're
        # over the cap, rather than letting the dict grow forever.
        while len(self._buckets) > self._max_tracked_keys:
            self._buckets.popitem(last=False)

        if window.count > limit_per_minute:
            remaining = self._window_seconds - (now - window.window_start)
            retry_after = max(1, math.ceil(remaining))
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down and try again shortly.",
                headers={"Retry-After": str(retry_after)},
            )


def _client_ip(request: Request) -> str:
    """Best-effort caller IP for unauthenticated rate limiting.

    Keying is gated by `settings.TRUST_PROXY_HEADERS` (default False):

    - False (default): uses `request.client.host` only. `X-Forwarded-For` is
      ignored entirely. Nothing in this deployment strips or rewrites an
      inbound `X-Forwarded-For` header, so trusting it here would let a
      direct caller send a fresh, arbitrary value on every request and land
      in a brand-new rate-limit bucket each time -- defeating the very
      brute-force protection this limiter exists to provide. The tradeoff:
      if the app *is* actually behind a real reverse proxy while this flag
      is off, every request arrives with the same `request.client.host` (the
      proxy's IP), so all callers share one bucket. That's a deliberate
      fail-CLOSED choice (over-limiting real distinct users) rather than
      fail-OPEN (no effective limiting at all) -- the correct default when
      we don't know whether a trusted proxy is in front of us.
    - True: only set this when the app is actually deployed behind a
      reverse proxy that overwrites/appends to `X-Forwarded-For` itself
      (e.g. Azure App Service). In that case, use the RIGHTMOST entry of
      `X-Forwarded-For`, not the leftmost. The rightmost entry is the one
      appended by the nearest trusted proxy hop and therefore cannot be
      forged by the external client; every entry to its left (including the
      traditional "original client" leftmost entry) is attacker-controlled
      free text the client can set to anything, including a fresh random
      value per request. Do not "fix" this back to the leftmost entry --
      that reintroduces the spoofing bypass. Falls back to
      `request.client.host` when the header is absent or empty.

    Falls back to a constant key if `request.client` is None, which
    Starlette allows (e.g. some test clients/transports).

    NOTE: even the rightmost `X-Forwarded-For` entry is only as trustworthy
    as the proxy in front of us; this is abuse-mitigation, not a security
    boundary, and must never be used for authentication/authorization.

    Args:
        request: The incoming request.

    Returns:
        A best-effort identity string for the caller.
    """
    if settings.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            entries = [entry.strip() for entry in forwarded.split(",")]
            non_empty = [entry for entry in entries if entry]
            if non_empty:
                return non_empty[-1]
    if request.client is not None:
        return request.client.host
    return "unknown-client"


def rate_limit_by_ip(limiter: FixedWindowRateLimiter, limit_per_minute: Callable[[], int]) -> Callable[[Request], None]:
    """Build a per-IP rate-limiting FastAPI dependency.

    Intended for unauthenticated routes, where the caller's IP is the only
    identity available.

    Args:
        limiter: The `FixedWindowRateLimiter` instance tracking this route's state.
        limit_per_minute: Called on each request to read the current limit
            (e.g. `lambda: settings.RATE_LIMIT_AUTH_PER_MINUTE`) rather than
            baking one in at import time, so config changes and test
            monkeypatches take effect immediately.

    Returns:
        A dependency callable suitable for a route's `dependencies=[...]`.
    """

    def _dependency(request: Request) -> None:
        limiter.check(_client_ip(request), limit_per_minute())

    return _dependency


def rate_limit_by_user(limiter: FixedWindowRateLimiter, limit_per_minute: Callable[[], int]) -> Callable[..., User]:
    """Build a per-user rate-limiting FastAPI dependency, composed on top of auth.

    Keys the limiter by `User.id` rather than IP: it survives NAT (many
    users behind one IP don't share a bucket) and can't be spoofed the way a
    header can. Depends on `get_current_user`, so applying this dependency
    to a route both enforces authentication and adds rate limiting in one
    line — no separate `Depends(get_current_user)` needed alongside it.

    Args:
        limiter: The `FixedWindowRateLimiter` instance tracking this route's state.
        limit_per_minute: Called on each request to read the current limit,
            same rationale as `rate_limit_by_ip`.

    Returns:
        A dependency callable suitable for a route's `dependencies=[...]`
        (or as a regular parameter dependency, since it also returns the user).
    """

    def _dependency(current_user: Annotated[User, Depends(get_current_user)]) -> User:
        limiter.check(current_user.id, limit_per_minute())
        return current_user

    return _dependency
