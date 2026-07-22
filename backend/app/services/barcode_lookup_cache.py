"""In-memory cache of barcode-lookup results, with FIFO eviction.

Mirrors `app.services.puzzle_store`: an `OrderedDict` capped at `MAX_ENTRIES`
with oldest-first eviction, exposed through an `@lru_cache()` singleton
getter. Both hits and misses are cached, so a barcode held in front of the
camera doesn't re-probe the Ravensburger CDN on every stable read.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# Cap on cached lookups. Each hit holds one ~1000px JPEG (~100-300 KB), so
# 100 entries bounds memory at a few tens of MB in the worst case.
MAX_ENTRIES = 100


@dataclass(frozen=True)
class BarcodeLookupRecord:
    """The outcome of one EAN lookup, hit or miss.

    Attributes:
        found: Whether a product image was found for the EAN.
        image_jpeg: The JPEG-converted product image when found, else None.
        image_kind: "motif" (clean puzzle artwork) or "box" when found,
            else None.
        article_number: The resolved article number when found, else None.
    """

    found: bool
    image_jpeg: Optional[bytes]
    image_kind: Optional[str]
    article_number: Optional[str]


class BarcodeLookupCache:
    """In-memory lookup results keyed by EAN, with FIFO eviction at `MAX_ENTRIES`."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._records: "OrderedDict[str, BarcodeLookupRecord]" = OrderedDict()

    def get(self, ean: str) -> Optional[BarcodeLookupRecord]:
        """Look up a cached result by EAN.

        Args:
            ean: The EAN-13 barcode payload.

        Returns:
            The cached record (hit or miss), or None if never looked up
            (or evicted).
        """
        return self._records.get(ean)

    def put(self, ean: str, record: BarcodeLookupRecord) -> None:
        """Cache a lookup result, evicting the oldest entry if over capacity.

        Args:
            ean: The EAN-13 barcode payload.
            record: The lookup outcome to cache.
        """
        self._records[ean] = record
        if len(self._records) > MAX_ENTRIES:
            self._records.popitem(last=False)


@lru_cache()
def get_barcode_lookup_cache() -> BarcodeLookupCache:
    """Get or create the singleton BarcodeLookupCache instance.

    Returns:
        The shared BarcodeLookupCache instance.
    """
    return BarcodeLookupCache()
