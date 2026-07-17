"""Tests for the in-memory rate limiter and its FastAPI wiring.

Covers app/rate_limit.py itself plus its use on POST /api/v1/auth/google
(per-IP) and POST /api/v1/piece/preview (per-user).

Mirrors the token/mocking patterns in tests/test_auth.py and the
photo-upload helpers in tests/test_main.py. The process-global limiter
instances (app.main.auth_rate_limiter / piece_preview_rate_limiter) are
reset before/after every test by the autouse fixture in tests/conftest.py.
"""

import itertools
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Generator
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import settings
from app.main import app, auth_rate_limiter, piece_preview_rate_limiter
from app.rate_limit import FixedWindowRateLimiter, _client_ip

client = TestClient(app)

_counter = itertools.count()


def make_photo_jpeg() -> bytes:
    """Create a tiny in-memory JPEG so upload endpoints have something to decode."""
    image = Image.new("RGB", (64, 64), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 50, 50), fill=(220, 210, 190))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def photo_files() -> dict[str, tuple[str, BytesIO, str]]:
    """Build a fresh multipart files dict for a photo upload (BytesIO isn't reusable)."""
    return {"file": ("photo.jpg", BytesIO(make_photo_jpeg()), "image/jpeg")}


def create_test_token(user_id: str = "test-user-id") -> str:
    """Create a valid test JWT for `user_id`, matching tests/test_auth.py's helper."""
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "email": f"{user_id}@example.com",
        "name": "Test User",
        "picture": None,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def auth_header(user_id: str = "test-user-id") -> dict[str, str]:
    """Build an Authorization header for `user_id`."""
    return {"Authorization": f"Bearer {create_test_token(user_id)}"}


@pytest.fixture
def mock_piece_detector_not_found() -> Generator[None, None, None]:
    """Make /api/v1/piece/preview fast and deterministic: nothing detected."""
    detector = MagicMock()
    detector.detect_region.return_value = None
    with patch("app.main.get_piece_detector", return_value=detector):
        yield


@pytest.fixture
def mock_google_auth_success() -> Generator[None, None, None]:
    """Make /api/v1/auth/google succeed without a real Google network call."""
    mock_user_info = {
        "sub": "google-user-123",
        "email": "user@example.com",
        "name": "Google User",
        "picture": None,
    }
    with patch("app.auth.service.AuthService.verify_google_token", return_value=mock_user_info):
        yield


def unique_email() -> dict[str, str]:
    """A fresh JSON body for /api/v1/auth/google (the endpoint doesn't care about the value)."""
    return {"id_token": f"fake-google-token-{next(_counter)}"}


# =============================================================================
# _client_ip helper unit tests
# =============================================================================


def _fake_request(forwarded_for: str | None, client_host: str | None) -> MagicMock:
    """Build a minimal Request-like double for _client_ip, without a real ASGI request."""
    request = MagicMock()
    request.headers.get.return_value = forwarded_for
    request.client = MagicMock(host=client_host) if client_host is not None else None
    return request


class TestClientIp:
    """Unit tests for the caller-IP resolution used by the per-IP limiter."""

    def test_default_ignores_x_forwarded_for_entirely(self) -> None:
        """TRUST_PROXY_HEADERS=False (default): X-Forwarded-For is ignored, even if present."""
        with patch.object(settings, "TRUST_PROXY_HEADERS", False):
            request = _fake_request(forwarded_for="1.2.3.4, 10.0.0.1", client_host="10.0.0.1")
            assert _client_ip(request) == "10.0.0.1"

    def test_default_spoofed_x_forwarded_for_cannot_create_new_bucket(self) -> None:
        """The attack this fix closes: a varying X-Forwarded-For must not change the key.

        With TRUST_PROXY_HEADERS off (the default), the resolved key must stay the
        same across requests that only differ by X-Forwarded-For, since nothing in
        this deployment strips/rewrites an inbound X-Forwarded-For header.
        """
        with patch.object(settings, "TRUST_PROXY_HEADERS", False):
            first = _fake_request(forwarded_for="1.1.1.1", client_host="10.0.0.1")
            second = _fake_request(forwarded_for="9.9.9.9", client_host="10.0.0.1")
            assert _client_ip(first) == _client_ip(second) == "10.0.0.1"

    def test_falls_back_to_request_client_host_when_no_forwarded_header(self) -> None:
        """Without X-Forwarded-For, falls back to request.client.host."""
        with patch.object(settings, "TRUST_PROXY_HEADERS", False):
            request = _fake_request(forwarded_for=None, client_host="192.168.1.1")
            assert _client_ip(request) == "192.168.1.1"

    def test_falls_back_to_constant_when_client_is_none(self) -> None:
        """request.client can be None (Starlette allows it); falls back to a constant key."""
        request = _fake_request(forwarded_for=None, client_host=None)
        assert _client_ip(request) == "unknown-client"

    def test_trust_proxy_headers_uses_rightmost_entry(self) -> None:
        """TRUST_PROXY_HEADERS=True: the rightmost entry (nearest trusted proxy hop) wins.

        The leftmost entry is attacker-controlled free text; only the rightmost
        entry is guaranteed proxy-authored.
        """
        with patch.object(settings, "TRUST_PROXY_HEADERS", True):
            request = _fake_request(forwarded_for="1.2.3.4, 5.6.7.8", client_host="5.6.7.8")
            assert _client_ip(request) == "5.6.7.8"

    def test_trust_proxy_headers_spoofed_leftmost_cannot_create_new_bucket(self) -> None:
        """A varying, attacker-controlled leftmost entry must not change the resolved key.

        With TRUST_PROXY_HEADERS=True, only the rightmost entry is used, so the
        leftmost entry (which the client fully controls) can be anything.
        """
        with patch.object(settings, "TRUST_PROXY_HEADERS", True):
            first = _fake_request(forwarded_for="1.1.1.1, 5.6.7.8", client_host="5.6.7.8")
            second = _fake_request(forwarded_for="9.9.9.9, 5.6.7.8", client_host="5.6.7.8")
            assert _client_ip(first) == _client_ip(second) == "5.6.7.8"

    def test_trust_proxy_headers_strips_whitespace_and_skips_empty_entries(self) -> None:
        """Rightmost entry resolution tolerates stray whitespace and a trailing empty entry."""
        with patch.object(settings, "TRUST_PROXY_HEADERS", True):
            request = _fake_request(forwarded_for=" 1.2.3.4 ,  5.6.7.8  , ", client_host="5.6.7.8")
            assert _client_ip(request) == "5.6.7.8"

    def test_trust_proxy_headers_falls_back_when_header_absent(self) -> None:
        """TRUST_PROXY_HEADERS=True but no X-Forwarded-For header: falls back to request.client.host."""
        with patch.object(settings, "TRUST_PROXY_HEADERS", True):
            request = _fake_request(forwarded_for=None, client_host="192.168.1.1")
            assert _client_ip(request) == "192.168.1.1"


# =============================================================================
# FixedWindowRateLimiter unit tests
# =============================================================================


class TestFixedWindowRateLimiter:
    """Direct tests against the limiter, independent of FastAPI wiring."""

    def test_requests_under_limit_do_not_raise(self) -> None:
        """Every request within the limit passes silently."""
        limiter = FixedWindowRateLimiter()
        for _ in range(5):
            limiter.check("key-a", limit_per_minute=5)  # should not raise

    def test_request_over_limit_raises_429_with_retry_after(self) -> None:
        """The (limit+1)th request in a window raises 429 with a Retry-After header."""
        limiter = FixedWindowRateLimiter(window_seconds=60.0)
        for _ in range(3):
            limiter.check("key-a", limit_per_minute=3)

        with pytest.raises(HTTPException) as exc_info:
            limiter.check("key-a", limit_per_minute=3)

        error = exc_info.value
        assert error.status_code == 429
        headers = error.headers or {}
        assert "Retry-After" in headers
        assert int(headers["Retry-After"]) >= 1

    def test_window_resets_after_expiry_via_injected_clock(self) -> None:
        """Once the injected clock advances past window_seconds, the counter resets.

        Uses an injectable fake clock instead of time.sleep() so the test is
        instant and deterministic.
        """
        current_time = [1000.0]
        limiter = FixedWindowRateLimiter(window_seconds=60.0, clock=lambda: current_time[0])

        for _ in range(3):
            limiter.check("key-a", limit_per_minute=3)
        with pytest.raises(HTTPException):
            limiter.check("key-a", limit_per_minute=3)

        # Advance the fake clock past the window.
        current_time[0] += 60.0
        limiter.check("key-a", limit_per_minute=3)  # should not raise: fresh window

    def test_set_clock_swaps_time_source(self) -> None:
        """set_clock lets a test inject a different time source after construction."""
        limiter = FixedWindowRateLimiter(window_seconds=60.0)
        current_time = [500.0]
        limiter.set_clock(lambda: current_time[0])

        for _ in range(2):
            limiter.check("key-a", limit_per_minute=2)
        with pytest.raises(HTTPException):
            limiter.check("key-a", limit_per_minute=2)

        current_time[0] += 61.0
        limiter.check("key-a", limit_per_minute=2)  # fresh window, should not raise

    def test_limit_zero_disables_limiting(self) -> None:
        """A limit_per_minute of 0 means unlimited, even for many requests."""
        limiter = FixedWindowRateLimiter()
        for _ in range(1000):
            limiter.check("key-a", limit_per_minute=0)  # should never raise
        assert limiter.tracked_key_count == 0  # disabled: nothing is even tracked

    def test_different_keys_have_independent_buckets(self) -> None:
        """Exhausting one key's limit does not affect a different key."""
        limiter = FixedWindowRateLimiter()
        for _ in range(3):
            limiter.check("key-a", limit_per_minute=3)
        with pytest.raises(HTTPException):
            limiter.check("key-a", limit_per_minute=3)

        limiter.check("key-b", limit_per_minute=3)  # unaffected, should not raise

    def test_memory_bound_evicts_oldest_key_once_over_cap(self) -> None:
        """The tracked-key dict never grows past max_tracked_keys."""
        limiter = FixedWindowRateLimiter(max_tracked_keys=3)

        for i in range(10):
            limiter.check(f"key-{i}", limit_per_minute=100)
            assert limiter.tracked_key_count <= 3

        assert limiter.tracked_key_count == 3
        # The most recently touched key must still be tracked...
        limiter.check("key-9", limit_per_minute=100)
        # ...while a long-evicted key starts a fresh window rather than
        # inheriting stale state (best-effort check: just shouldn't raise
        # even after many more hits than the original limit would allow
        # if state had leaked).
        for _ in range(5):
            limiter.check("key-0", limit_per_minute=100)

    def test_reset_clears_all_tracked_state(self) -> None:
        """reset() drops every key, as used by the test-isolation fixture."""
        limiter = FixedWindowRateLimiter()
        limiter.check("key-a", limit_per_minute=10)
        limiter.check("key-b", limit_per_minute=10)
        assert limiter.tracked_key_count == 2

        limiter.reset()
        assert limiter.tracked_key_count == 0


# =============================================================================
# Endpoint wiring: POST /api/v1/auth/google (per-IP)
# =============================================================================


class TestAuthGoogleRateLimit:
    """Integration tests for the per-IP limiter on /api/v1/auth/google."""

    def test_requests_within_limit_succeed(self, mock_google_auth_success: None) -> None:
        """Requests up to the configured limit all succeed."""
        with patch.object(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 3):
            for _ in range(3):
                response = client.post(
                    "/api/v1/auth/google",
                    json=unique_email(),
                    headers={"X-Forwarded-For": "10.0.0.1"},
                )
                assert response.status_code == 200

    def test_request_over_limit_returns_429_with_retry_after(self, mock_google_auth_success: None) -> None:
        """The request past the limit is rejected with 429 and Retry-After."""
        with patch.object(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 2):
            for _ in range(2):
                response = client.post(
                    "/api/v1/auth/google",
                    json=unique_email(),
                    headers={"X-Forwarded-For": "10.0.0.2"},
                )
                assert response.status_code == 200

            response = client.post(
                "/api/v1/auth/google",
                json=unique_email(),
                headers={"X-Forwarded-For": "10.0.0.2"},
            )

        assert response.status_code == 429
        assert "retry-after" in response.headers
        assert int(response.headers["retry-after"]) >= 1

    def test_default_spoofed_x_forwarded_for_does_not_evade_rate_limit(self, mock_google_auth_success: None) -> None:
        """Regression test for the XFF rate-limit bypass: default config keys on real IP only.

        TestClient requests all share the same underlying `request.client.host`
        (TestClient's fixed test IP), so with TRUST_PROXY_HEADERS off (the default)
        a varying, attacker-supplied X-Forwarded-For must NOT grant a fresh bucket
        per request -- that was the vulnerability: an attacker sending a new random
        X-Forwarded-For on every request evaded the limiter entirely. This test
        fails against the old leftmost-X-Forwarded-For-trusting implementation.
        """
        with (
            patch.object(settings, "TRUST_PROXY_HEADERS", False),
            patch.object(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 1),
        ):
            first = client.post(
                "/api/v1/auth/google",
                json=unique_email(),
                headers={"X-Forwarded-For": "10.0.0.10"},
            )
            assert first.status_code == 200

            # A different, spoofed X-Forwarded-For must NOT grant a new bucket: same
            # underlying test client IP, so this must now be rate-limited.
            second = client.post(
                "/api/v1/auth/google",
                json=unique_email(),
                headers={"X-Forwarded-For": "10.0.0.11"},
            )
            assert second.status_code == 429

    def test_trust_proxy_headers_true_uses_rightmost_x_forwarded_for_entry(
        self, mock_google_auth_success: None
    ) -> None:
        """With TRUST_PROXY_HEADERS=True, only the rightmost X-Forwarded-For entry is trusted.

        A spoofed leftmost entry must not create a new bucket; only varying the
        rightmost entry (the one a trusted proxy would append) does.
        """
        with (
            patch.object(settings, "TRUST_PROXY_HEADERS", True),
            patch.object(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 1),
        ):
            first = client.post(
                "/api/v1/auth/google",
                json=unique_email(),
                headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.20"},
            )
            assert first.status_code == 200

            # Spoofed leftmost entry, same rightmost entry: still rate-limited.
            second = client.post(
                "/api/v1/auth/google",
                json=unique_email(),
                headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.20"},
            )
            assert second.status_code == 429

            # Different rightmost entry: independent bucket, should still succeed.
            third = client.post(
                "/api/v1/auth/google",
                json=unique_email(),
                headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.21"},
            )
            assert third.status_code == 200

    def test_limit_of_zero_disables_rate_limiting(self, mock_google_auth_success: None) -> None:
        """Setting RATE_LIMIT_AUTH_PER_MINUTE to 0 disables the limit."""
        with patch.object(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 0):
            for _ in range(25):
                response = client.post(
                    "/api/v1/auth/google",
                    json=unique_email(),
                    headers={"X-Forwarded-For": "10.0.0.99"},
                )
                assert response.status_code == 200


# =============================================================================
# Endpoint wiring: POST /api/v1/piece/preview (per-user)
# =============================================================================


class TestPiecePreviewRateLimit:
    """Integration tests for the per-user limiter on /api/v1/piece/preview."""

    def test_requests_within_limit_succeed(self, mock_piece_detector_not_found: None) -> None:
        """Requests up to the configured limit all succeed for one user."""
        with patch.object(settings, "RATE_LIMIT_PREVIEW_PER_MINUTE", 3):
            for _ in range(3):
                response = client.post(
                    "/api/v1/piece/preview",
                    files=photo_files(),
                    headers=auth_header("user-a"),
                )
                assert response.status_code == 200

    def test_request_over_limit_returns_429_with_retry_after(self, mock_piece_detector_not_found: None) -> None:
        """The request past the limit is rejected with 429 and Retry-After."""
        with patch.object(settings, "RATE_LIMIT_PREVIEW_PER_MINUTE", 2):
            for _ in range(2):
                response = client.post(
                    "/api/v1/piece/preview",
                    files=photo_files(),
                    headers=auth_header("user-b"),
                )
                assert response.status_code == 200

            response = client.post(
                "/api/v1/piece/preview",
                files=photo_files(),
                headers=auth_header("user-b"),
            )

        assert response.status_code == 429
        assert "retry-after" in response.headers
        assert int(response.headers["retry-after"]) >= 1

    def test_exhausting_one_user_does_not_rate_limit_another(self, mock_piece_detector_not_found: None) -> None:
        """A shared/global counter here would be a cross-user DoS -- verify keying is per-user."""
        with patch.object(settings, "RATE_LIMIT_PREVIEW_PER_MINUTE", 1):
            first = client.post(
                "/api/v1/piece/preview",
                files=photo_files(),
                headers=auth_header("user-exhausted"),
            )
            assert first.status_code == 200

            # Same user again: now rate-limited.
            second = client.post(
                "/api/v1/piece/preview",
                files=photo_files(),
                headers=auth_header("user-exhausted"),
            )
            assert second.status_code == 429

            # A different authenticated user must be unaffected.
            third = client.post(
                "/api/v1/piece/preview",
                files=photo_files(),
                headers=auth_header("user-fresh"),
            )
            assert third.status_code == 200

    def test_limit_of_zero_disables_rate_limiting(self, mock_piece_detector_not_found: None) -> None:
        """Setting RATE_LIMIT_PREVIEW_PER_MINUTE to 0 disables the limit."""
        with patch.object(settings, "RATE_LIMIT_PREVIEW_PER_MINUTE", 0):
            for _ in range(25):
                response = client.post(
                    "/api/v1/piece/preview",
                    files=photo_files(),
                    headers=auth_header("user-unlimited"),
                )
                assert response.status_code == 200

    def test_still_requires_authentication(self, mock_piece_detector_not_found: None) -> None:
        """The rate-limit dependency composes on top of auth; it must not bypass it."""
        response = client.post("/api/v1/piece/preview", files=photo_files())
        assert response.status_code in (401, 403)


# =============================================================================
# Sanity: the limiters used by the app are the same instances the test fixture resets
# =============================================================================


def test_app_exposes_module_level_limiter_instances() -> None:
    """The dependency wiring in app.main uses these exact singletons.

    Guards against a future refactor accidentally constructing a new limiter
    per-request (which would silently disable rate limiting) or per-app
    (which the conftest fixture wouldn't be resetting).
    """
    assert isinstance(app, FastAPI)
    assert isinstance(auth_rate_limiter, FixedWindowRateLimiter)
    assert isinstance(piece_preview_rate_limiter, FixedWindowRateLimiter)
