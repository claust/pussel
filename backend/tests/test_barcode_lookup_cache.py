"""Tests for the in-memory barcode lookup cache (app/services/barcode_lookup_cache.py)."""

from app.services.barcode_lookup_cache import MAX_ENTRIES, BarcodeLookupCache, BarcodeLookupRecord

HIT = BarcodeLookupRecord(found=True, box_image_jpeg=b"jpeg-bytes", article_number="05009")
MISS = BarcodeLookupRecord(found=False, box_image_jpeg=None, article_number=None)


class TestBarcodeLookupCache:
    """Roundtrip, miss caching, and FIFO eviction."""

    def test_get_returns_none_for_unknown_ean(self) -> None:
        """An EAN never looked up returns None (distinct from a cached miss)."""
        assert BarcodeLookupCache().get("4005556050093") is None

    def test_put_get_roundtrip(self) -> None:
        """A cached hit comes back intact."""
        cache = BarcodeLookupCache()
        cache.put("4005556050093", HIT)
        assert cache.get("4005556050093") == HIT

    def test_misses_are_cached(self) -> None:
        """A cached miss is returned as a record, not None."""
        cache = BarcodeLookupCache()
        cache.put("4006381333931", MISS)
        assert cache.get("4006381333931") == MISS

    def test_fifo_eviction_at_capacity(self) -> None:
        """Exceeding MAX_ENTRIES evicts the oldest entry, keeping the rest."""
        cache = BarcodeLookupCache()
        for index in range(MAX_ENTRIES + 1):
            cache.put(f"ean-{index}", MISS)
        assert cache.get("ean-0") is None
        assert cache.get("ean-1") == MISS
        assert cache.get(f"ean-{MAX_ENTRIES}") == MISS
