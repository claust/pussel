"""Tests for the Ravensburger image client (app/services/ravensburger_client.py).

ravensburger.org is never contacted: `httpx.AsyncClient` is replaced with a
mock supporting the async-context-manager protocol, per the suite's
convention of mocking at the outbound boundary with unittest.mock (no respx).
The suite has no pytest-asyncio, so async methods are driven with
`asyncio.run`.
"""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from app.config import settings
from app.services.ravensburger_client import RavensburgerClient, RavensburgerImage

MOTIF_BYTES = b"motif-jpeg-bytes"
BOX_BYTES = b"box-jpeg-bytes"
NOT_FOUND = MagicMock(status_code=404, content=b"")


def mock_async_client(responses: Optional[list[MagicMock]] = None, exc: Optional[Exception] = None) -> MagicMock:
    """Build a stand-in for the httpx.AsyncClient class.

    Args:
        responses: The responses successive `get()` calls should resolve to.
        exc: An exception `get()` should raise instead.

    Returns:
        A MagicMock usable as a replacement for the AsyncClient constructor,
        whose instances work as async context managers.
    """
    inner = MagicMock()
    inner.get = AsyncMock(side_effect=exc) if exc else AsyncMock(side_effect=responses)
    context_manager = MagicMock()
    context_manager.__aenter__ = AsyncMock(return_value=inner)
    context_manager.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=context_manager)


def fetch(article_number: str) -> Optional[RavensburgerImage]:
    """Run fetch_puzzle_image synchronously.

    Args:
        article_number: The article number to fetch.

    Returns:
        The fetch result.
    """
    return asyncio.run(RavensburgerClient().fetch_puzzle_image(article_number))


class TestFetchPuzzleImage:
    """Hit/miss and fallback behavior of RavensburgerClient.fetch_puzzle_image."""

    def test_motif_returned_when_present(self) -> None:
        """A 200 on the _1 motif URL is a hit; the box URL is never tried."""
        response = MagicMock(status_code=200, content=MOTIF_BYTES)
        constructor = mock_async_client([response])
        with patch("app.services.ravensburger_client.httpx.AsyncClient", constructor):
            assert fetch("12000877") == RavensburgerImage(content=MOTIF_BYTES, kind="motif")

    def test_falls_back_to_box_when_motif_missing(self) -> None:
        """A 404 on the motif falls back to the box shot."""
        box = MagicMock(status_code=200, content=BOX_BYTES)
        with patch("app.services.ravensburger_client.httpx.AsyncClient", mock_async_client([NOT_FOUND, box])):
            assert fetch("05009") == RavensburgerImage(content=BOX_BYTES, kind="box")

    def test_unknown_article_is_a_miss(self) -> None:
        """404s on both the motif and box URLs (unknown article) are a miss."""
        with patch("app.services.ravensburger_client.httpx.AsyncClient", mock_async_client([NOT_FOUND, NOT_FOUND])):
            assert fetch("99999") is None

    def test_empty_body_is_a_miss(self) -> None:
        """A 200 with an empty body is not treated as an image."""
        empty = MagicMock(status_code=200, content=b"")
        with patch("app.services.ravensburger_client.httpx.AsyncClient", mock_async_client([empty, empty])):
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

    def test_requests_motif_then_box_urls(self) -> None:
        """The motif (_1) URL is requested first, the bare box URL second."""
        constructor = mock_async_client([NOT_FOUND, NOT_FOUND])
        with patch("app.services.ravensburger_client.httpx.AsyncClient", constructor):
            fetch("12000877")
        inner = constructor.return_value.__aenter__.return_value
        requested = [call.args[0] for call in inner.get.await_args_list]
        size = settings.RAVENSBURGER_IMAGE_SIZE
        assert requested == [
            f"https://www.ravensburger.org/produktseiten/{size}/12000877_1.jpg",
            f"https://www.ravensburger.org/produktseiten/{size}/12000877.jpg",
        ]

    def test_image_url_uses_configured_size_bucket(self) -> None:
        """The image URL embeds the configured size bucket and the article number."""
        with patch.object(settings, "RAVENSBURGER_IMAGE_SIZE", 360):
            assert RavensburgerClient().image_url("05009", "_1") == (
                "https://www.ravensburger.org/produktseiten/360/05009_1.jpg"
            )
