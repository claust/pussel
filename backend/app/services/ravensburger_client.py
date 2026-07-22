"""HTTP client for Ravensburger's public box-image CDN.

The CDN serves box art at
``https://ravensburger.cloud/images/produktseiten/{W}x{H}/{article}.webp``
keyed by bare article number, at a fixed set of pre-rendered sizes (see
`Settings.RAVENSBURGER_IMAGE_WIDTH`/`_HEIGHT`). It never 404s: unknown
articles — and unsupported sizes — get HTTP 200 with a tiny (~458-byte)
placeholder, so a real hit is distinguished by payload size
(`Settings.RAVENSBURGER_PLACEHOLDER_MAX_BYTES`), not status.
"""

import logging
from functools import lru_cache
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class RavensburgerClient:
    """Fetches box images from the Ravensburger CDN, treating placeholders as misses."""

    def cdn_url(self, article_number: str) -> str:
        """Build the CDN URL for an article number.

        Args:
            article_number: The Ravensburger article number (5 or 8 digits).

        Returns:
            The image URL at the configured size.
        """
        size = f"{settings.RAVENSBURGER_IMAGE_WIDTH}x{settings.RAVENSBURGER_IMAGE_HEIGHT}"
        return f"https://ravensburger.cloud/images/produktseiten/{size}/{article_number}.webp"

    async def fetch_box_image(self, article_number: str) -> Optional[bytes]:
        """Fetch the box image for an article number, if one exists.

        Args:
            article_number: The candidate Ravensburger article number.

        Returns:
            The image bytes (webp) on a real hit, or None when the CDN
            returns a non-200, a placeholder-sized body, or any transport
            error/timeout — a miss is never an exception.
        """
        try:
            async with httpx.AsyncClient(timeout=settings.RAVENSBURGER_CDN_TIMEOUT_SECONDS) as client:
                response = await client.get(self.cdn_url(article_number))
        except httpx.HTTPError as exc:
            logger.warning("Ravensburger CDN fetch failed for article %s: %s", article_number, exc)
            return None

        if response.status_code != 200:
            return None
        if len(response.content) <= settings.RAVENSBURGER_PLACEHOLDER_MAX_BYTES:
            return None
        return response.content


@lru_cache()
def get_ravensburger_client() -> RavensburgerClient:
    """Get or create the singleton RavensburgerClient instance.

    Returns:
        The shared RavensburgerClient instance.
    """
    return RavensburgerClient()
