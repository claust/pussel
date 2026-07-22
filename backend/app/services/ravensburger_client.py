"""HTTP client for Ravensburger's public product-image host.

The Ravensburger website serves product images as static files at
``https://www.ravensburger.org/produktseiten/{size}/{article}{suffix}.jpg``
keyed by bare article number, at a fixed set of pre-rendered size buckets
(75, 100, 240, 360, 1024 — see `Settings.RAVENSBURGER_IMAGE_SIZE`). The
suffix selects the shot: no suffix is the box, ``_1`` is the clean puzzle
motif (the artwork without box, logo, or piece-count text), ``_2``/``_3``
are lifestyle photos. Misses — unknown articles, absent suffixes,
unsupported sizes — are genuine HTTP 404s (verified live 2026-07-22),
unlike the old ravensburger.cloud CDN which answered everything with a
200 placeholder.

The motif is preferred because it doubles as the reference image for piece
matching; the box shot is only a fallback for products without a ``_1``.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

MOTIF_SUFFIX = "_1"
BOX_SUFFIX = ""

ImageKind = Literal["motif", "box"]

# Fetch order: the motif when the product has one, else the box shot.
_SHOT_ATTEMPTS: tuple[tuple[str, ImageKind], ...] = ((MOTIF_SUFFIX, "motif"), (BOX_SUFFIX, "box"))


@dataclass(frozen=True)
class RavensburgerImage:
    """A fetched product image and which shot it is.

    Attributes:
        content: The raw JPEG bytes as served.
        kind: "motif" for the clean puzzle artwork (``_1``), "box" for the
            box-shot fallback.
    """

    content: bytes
    kind: ImageKind


class RavensburgerClient:
    """Fetches product images from ravensburger.org, preferring the puzzle motif."""

    def image_url(self, article_number: str, suffix: str = BOX_SUFFIX) -> str:
        """Build the image URL for an article number and shot suffix.

        Args:
            article_number: The Ravensburger article number (5 or 8 digits).
            suffix: The shot selector ("" for the box, "_1" for the motif).

        Returns:
            The image URL at the configured size bucket.
        """
        size = settings.RAVENSBURGER_IMAGE_SIZE
        return f"https://www.ravensburger.org/produktseiten/{size}/{article_number}{suffix}.jpg"

    async def fetch_puzzle_image(self, article_number: str) -> Optional[RavensburgerImage]:
        """Fetch the best available product image for an article number.

        Tries the clean puzzle motif (``_1``) first and falls back to the
        box shot, so a hit on either proves the article exists.

        Args:
            article_number: The candidate Ravensburger article number.

        Returns:
            The motif image when it exists, else the box image, else None
            when neither exists or on any transport error/timeout — a miss
            is never an exception.
        """
        try:
            async with httpx.AsyncClient(timeout=settings.RAVENSBURGER_CDN_TIMEOUT_SECONDS) as client:
                for suffix, kind in _SHOT_ATTEMPTS:
                    response = await client.get(self.image_url(article_number, suffix))
                    if response.status_code == 200 and response.content:
                        return RavensburgerImage(content=response.content, kind=kind)
        except httpx.HTTPError as exc:
            logger.warning("Ravensburger image fetch failed for article %s: %s", article_number, exc)
        return None


@lru_cache()
def get_ravensburger_client() -> RavensburgerClient:
    """Get or create the singleton RavensburgerClient instance.

    Returns:
        The shared RavensburgerClient instance.
    """
    return RavensburgerClient()
