"""Tests for the Ravensburger CDN client (app/services/ravensburger_client.py).

The CDN is never contacted: `httpx.AsyncClient` is replaced with a mock
supporting the async-context-manager protocol, per the suite's convention of
mocking at the outbound boundary with unittest.mock (no respx). The suite has
no pytest-asyncio, so async methods are driven with `asyncio.run`.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.config import settings
from app.services.ravensburger_client import RavensburgerClient

REAL_IMAGE_BYTES = b"x" * (settings.RAVENSBURGER_PLACEHOLDER_MAX_BYTES + 1)
PLACEHOLDER_BYTES = b"x" * 458


def mock_async_client(response: Optional[MagicMock] = None, exc: Optional[Exception] = None) -> MagicMock:
    """Build a stand-in for the httpx.AsyncClient class.

    Args:
        response: The response `get()` should resolve to.
        exc: An exception `get()` should raise instead.

    Returns:
        A MagicMock usable as a replacement for the AsyncClient constructor,
        whose instances work as async context managers.
    """
    inner = MagicMock()
    inner.get = AsyncMock(side_effect=exc) if exc else AsyncMock(return_value=response)
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=inner)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=context_manager)


def fetch(article_number: str) -> Optional[bytes]:
    """Run fetch_box_image synchronously.

    Args:
        article_number: The article number to fetch.

    Returns:
        The fetch result.
    """
    return asyncio.run(RavensburgerClient().fetch_box_image(article_number))


class TestFetchBoxImage:
    """Hit/miss behavior of RavensburgerClient.fetch_box_image."""

    def test_real_image_returned(self) -> None:
        """A 200 with a body above the placeholder threshold is a hit."""
        response = MagicMock(status_code=200, content=REAL_IMAGE_BYTES)
        with patch("app.services.ravensburger_client.httpx.AsyncClient", mock_async_client(response)):
            assert fetch("05009") == REAL_IMAGE_BYTES

    def test_placeholder_sized_body_is_a_miss(self) -> None:
        """A 200 with a placeholder-sized body (unknown article) is a miss."""
        response = MagicMock(status_code=200, content=PLACEHOLDER_BYTES)
        with patch("app.services.ravensburger_client.httpx.AsyncClient", mock_async_client(response)):
            assert fetch("99999") is None

    def test_non_200_is_a_miss(self) -> None:
        """Non-200 statuses are misses regardless of body size."""
        response = MagicMock(status_code=503, content=REAL_IMAGE_BYTES)
        with patch("app.services.ravensburger_client.httpx.AsyncClient", mock_async_client(response)):
            assert fetch("05009") is None

    def test_timeout_is_a_miss(self) -> None:
        """Transport timeouts are swallowed into a miss, never raised."""
        constructor = mock_async_client(exc=httpx.TimeoutException("timed out"))
        with patch("app.services.ravensburger_client.httpx.AsyncClient", constructor):
            assert fetch("05009") is None

    def test_http_error_is_a_miss(self) -> None:
        """Generic httpx errors are swallowed into a miss, never raised."""
        constructor = mock_async_client(exc=httpx.ConnectError("connection refused"))
        with patch("app.services.ravensburger_client.httpx.AsyncClient", constructor):
            assert fetch("05009") is None

    def test_cdn_url_uses_configured_size(self) -> None:
        """The CDN URL embeds the configured dimensions and the article number."""
        assert RavensburgerClient().cdn_url("05009") == (
            "https://ravensburger.cloud/images/produktseiten/520x445/05009.webp"
        )
