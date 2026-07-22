"""Tests for GET /api/v1/puzzle/barcode/{ean} (Ravensburger product-image lookup).

ravensburger.org is never contacted: `app.main.get_ravensburger_client` is
patched with a fake whose `fetch_puzzle_image` is an AsyncMock. The
process-global lookup cache singleton is cleared around every test so cached
hits/misses can't leak between tests.

Mirrors the token helper in tests/test_rate_limit.py.
"""

import io
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app
from app.services.barcode_lookup_cache import get_barcode_lookup_cache
from app.services.ravensburger_client import ImageKind, RavensburgerImage

client = TestClient(app)

FROZEN_EAN = "4005556050093"  # standard line, article 05009
ADULT_EAN = "4005555006220"  # adult line, payload 00622, article 12000622
OTHER_BRAND_EAN = "4006381333931"  # valid checksum, non-Ravensburger prefix


def make_jpeg_bytes() -> bytes:
    """Create a tiny in-memory JPEG standing in for a real product image."""
    image = Image.new("RGB", (64, 64), (30, 60, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


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


def auth_headers(user_id: str = "test-user-id") -> dict[str, str]:
    """Build Authorization headers with a valid test token."""
    return {"Authorization": f"Bearer {create_test_token(user_id)}"}


def make_fake_ravensburger_client(hit_articles: dict[str, bytes], kind: ImageKind = "motif") -> MagicMock:
    """Build a fake RavensburgerClient hitting only the given article numbers.

    Args:
        hit_articles: Article number -> image bytes for articles that exist.
        kind: The image kind ("motif" or "box") every hit reports.

    Returns:
        A MagicMock whose async `fetch_puzzle_image` resolves per
        `hit_articles`.
    """

    async def fetch_puzzle_image(article_number: str) -> Optional[RavensburgerImage]:
        content = hit_articles.get(article_number)
        if content is None:
            return None
        return RavensburgerImage(content=content, kind=kind)

    fake = MagicMock()
    fake.fetch_puzzle_image = AsyncMock(side_effect=fetch_puzzle_image)
    return fake


@pytest.fixture(autouse=True)
def _reset_barcode_cache() -> Generator[None, None, None]:
    """Clear the process-global lookup cache singleton around every test."""
    get_barcode_lookup_cache.cache_clear()
    yield
    get_barcode_lookup_cache.cache_clear()


class TestBarcodeLookupEndpoint:
    """Behavior of GET /api/v1/puzzle/barcode/{ean}."""

    def test_requires_authentication(self) -> None:
        """An unauthenticated request is rejected."""
        response = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}")
        assert response.status_code in (401, 403)

    def test_malformed_ean_rejected(self) -> None:
        """Wrong length, non-digit, and bad-checksum EANs all yield 400."""
        for bad_ean in ("12345", "400555605009X", "4005556050094"):
            response = client.get(f"/api/v1/puzzle/barcode/{bad_ean}", headers=auth_headers())
            assert response.status_code == 400, bad_ean

    def test_standard_line_hit(self) -> None:
        """A standard-line EAN resolves to its payload article with a JPEG data URL."""
        fake = make_fake_ravensburger_client({"05009": make_jpeg_bytes()})
        with patch("app.main.get_ravensburger_client", return_value=fake):
            response = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=auth_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["found"] is True
        assert body["article_number"] == "05009"
        assert body["box_image"].startswith("data:image/jpeg;base64,")
        assert body["image_kind"] == "motif"
        fake.fetch_puzzle_image.assert_awaited_once_with("05009")

    def test_box_fallback_kind_reported(self) -> None:
        """When only the box shot exists, the response reports image_kind='box'."""
        fake = make_fake_ravensburger_client({"05009": make_jpeg_bytes()}, kind="box")
        with patch("app.main.get_ravensburger_client", return_value=fake):
            response = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=auth_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["found"] is True
        assert body["image_kind"] == "box"

    def test_standard_line_miss(self) -> None:
        """A standard-line EAN whose article has no CDN image returns found=False."""
        fake = make_fake_ravensburger_client({})
        with patch("app.main.get_ravensburger_client", return_value=fake):
            response = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=auth_headers())
        assert response.status_code == 200
        assert response.json() == {"found": False, "box_image": None, "image_kind": None, "article_number": None}

    def test_adult_line_probes_series_prefixes(self) -> None:
        """An adult-line EAN probes series candidates and resolves the hit's article."""
        fake = make_fake_ravensburger_client({"12000622": make_jpeg_bytes()})
        with patch("app.main.get_ravensburger_client", return_value=fake):
            response = client.get(f"/api/v1/puzzle/barcode/{ADULT_EAN}", headers=auth_headers())
        assert response.status_code == 200
        body = response.json()
        assert body["found"] is True
        assert body["article_number"] == "12000622"
        probed = [call.args[0] for call in fake.fetch_puzzle_image.await_args_list]
        assert probed == [f"{series}00622" for series in settings.RAVENSBURGER_SERIES_PREFIXES]

    def test_non_ravensburger_prefix_never_probes_cdn(self) -> None:
        """A valid EAN with a foreign GS1 prefix is a miss without any CDN call."""
        fake = make_fake_ravensburger_client({"anything": make_jpeg_bytes()})
        with patch("app.main.get_ravensburger_client", return_value=fake):
            response = client.get(f"/api/v1/puzzle/barcode/{OTHER_BRAND_EAN}", headers=auth_headers())
        assert response.status_code == 200
        assert response.json()["found"] is False
        fake.fetch_puzzle_image.assert_not_awaited()

    def test_second_lookup_served_from_cache(self) -> None:
        """Repeating a lookup (hit or miss) doesn't re-probe the CDN."""
        fake = make_fake_ravensburger_client({"05009": make_jpeg_bytes()})
        with patch("app.main.get_ravensburger_client", return_value=fake):
            first = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=auth_headers())
            second = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=auth_headers())
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert fake.fetch_puzzle_image.await_count == 1

    def test_undecodable_cdn_image_is_a_miss(self) -> None:
        """Bytes the CDN returns that aren't a decodable image yield found=False."""
        fake = make_fake_ravensburger_client({"05009": b"not-an-image" * 400})
        with patch("app.main.get_ravensburger_client", return_value=fake):
            response = client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=auth_headers())
        assert response.status_code == 200
        assert response.json()["found"] is False

    def test_rate_limited_per_user(self) -> None:
        """Requests beyond the per-user limit are rejected with 429."""
        fake = make_fake_ravensburger_client({})
        with (
            patch("app.main.get_ravensburger_client", return_value=fake),
            patch.object(settings, "RATE_LIMIT_BARCODE_PER_MINUTE", 2),
        ):
            headers = auth_headers("rate-limit-user")
            assert client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=headers).status_code == 200
            assert client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=headers).status_code == 200
            assert client.get(f"/api/v1/puzzle/barcode/{FROZEN_EAN}", headers=headers).status_code == 429
