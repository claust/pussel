"""API tests for the piece-geometry endpoints and the preview include_quality flag."""

import os
import shutil
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Generator
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from piece_geometry_fixtures import (
    PIECE_A_COLORS,
    PIECE_A_EDGE_TYPES,
    PIECE_B_COLORS,
    PIECE_B_EDGE_TYPES,
    deterministic_config,
    encode_png,
    make_piece_rgba,
    rasterize_piece,
)
from PIL import Image, ImageDraw

from app.main import app, settings
from app.services.piece_detector import DetectedRegion
from app.services.piece_geometry import store as store_module
from app.services.piece_geometry.contour import mask_to_contour
from app.services.piece_geometry.service import PieceGeometryService

client = TestClient(app)


@pytest.fixture(autouse=True)
def _ensure_upload_dir() -> Generator[None, None, None]:
    """Recreate UPLOAD_DIR before each test (another module's fixture may have removed it)."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield
    shutil.rmtree(settings.UPLOAD_DIR, ignore_errors=True)


def create_test_token() -> str:
    """Create a valid test JWT token for authentication (mirrors tests/test_main.py)."""
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    payload = {
        "sub": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "picture": None,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def get_auth_header() -> dict[str, str]:
    """Create an authorization header with a valid test token."""
    return {"Authorization": f"Bearer {create_test_token()}"}


def make_photo_jpeg() -> bytes:
    """Create an in-memory JPEG photo with a bright rectangle on a dark background."""
    image = Image.new("RGB", (800, 600), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 80, 700, 520), fill=(220, 210, 190))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def upload_test_puzzle() -> str:
    """Upload a throwaway puzzle and return its puzzle_id."""
    response = client.post(
        "/api/v1/puzzle/upload",
        files={"file": ("photo.jpg", BytesIO(make_photo_jpeg()), "image/jpeg")},
        headers=get_auth_header(),
    )
    assert response.status_code == 200
    return str(response.json()["puzzle_id"])


def mocked_geometry_service() -> PieceGeometryService:
    """A PieceGeometryService whose mocked background remover echoes the uploaded RGBA PNG.

    The test uploads are RGBA PNGs whose alpha channel IS the piece mask, so
    "segmentation" is simply decoding the upload — this keeps the mocked-rembg
    pattern while letting different uploads carry different pieces.
    """
    remover = MagicMock()
    remover.remove_background.side_effect = lambda image_bytes: Image.open(BytesIO(image_bytes)).convert("RGBA")
    return PieceGeometryService(background_remover=remover)


def geometry_files(
    edge_types: list[str] = PIECE_A_EDGE_TYPES,
    colors: list[tuple[int, int, int]] = PIECE_A_COLORS,
) -> dict[str, tuple[str, BytesIO, str]]:
    """Build the multipart files dict for a piece-geometry photo upload.

    Args:
        edge_types: The synthetic piece's 4 edge types.
        colors: The synthetic piece's 4 quadrant BGR colors.

    Returns:
        The multipart files dict.
    """
    rgba = make_piece_rgba(edge_types, colors)
    return {"file": ("piece.png", BytesIO(encode_png(rgba)), "image/png")}


class TestUploadPieceGeometry:
    """Tests for POST /api/v1/puzzle/{puzzle_id}/piece/geometry."""

    def test_requires_auth(self) -> None:
        """The endpoint rejects unauthenticated requests."""
        puzzle_id = upload_test_puzzle()

        response = client.post(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", files=geometry_files())

        assert response.status_code in (401, 403)

    def test_unknown_puzzle_returns_404(self) -> None:
        """A geometry upload against a nonexistent puzzle_id returns 404."""
        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            response = client.post(
                "/api/v1/puzzle/does-not-exist/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )

        assert response.status_code == 404

    def test_no_file_returns_400(self) -> None:
        """Omitting the file returns 400."""
        puzzle_id = upload_test_puzzle()

        response = client.post(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=get_auth_header())

        assert response.status_code == 400

    def test_invalid_image_bytes_returns_400(self) -> None:
        """Non-image bytes are rejected before the pipeline runs."""
        puzzle_id = upload_test_puzzle()

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
            files={"file": ("piece.png", BytesIO(b"not an image"), "image/png")},
            headers=get_auth_header(),
        )

        assert response.status_code == 400

    def test_first_upload_returns_new_and_valid_record(self) -> None:
        """The first photo of a piece is enrolled as new, with a valid geometric record."""
        puzzle_id = upload_test_puzzle()

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            response = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "new"
        assert result["piece_id"] == "p001"
        assert result["match_piece_id"] is None
        assert result["lockable"] is True
        assert result["quality"]["is_clean"] is True
        assert result["quality"]["corner_disagreement"] is False
        assert len(result["record"]["corners"]) == 4
        assert len(result["record"]["edges"]) == 4
        for edge in result["record"]["edges"]:
            assert edge["type"] in ("tab", "blank", "flat")
            assert len(edge["polyline"]) == 100
        # include_contour defaults to false: the (large) contour is omitted.
        assert result["record"]["contour"] is None

    def test_include_contour_true_adds_the_contour_polyline(self) -> None:
        """include_contour=true adds the full contour to the response."""
        puzzle_id = upload_test_puzzle()

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            response = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry?include_contour=true",
                files=geometry_files(),
                headers=get_auth_header(),
            )

        assert response.status_code == 200
        contour = response.json()["record"]["contour"]
        assert contour is not None
        assert len(contour) > 0

    def test_same_piece_photographed_twice_returns_matched_with_same_piece_id(self) -> None:
        """Posting the identical synthetic piece a second time matches the first enrollment."""
        puzzle_id = upload_test_puzzle()

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            first = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )
            second = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )

        assert first.json()["status"] == "new"
        first_piece_id = first.json()["piece_id"]

        assert second.status_code == 200
        second_body = second.json()
        assert second_body["status"] == "matched"
        assert second_body["piece_id"] == first_piece_id
        assert second_body["match_piece_id"] == first_piece_id
        assert second_body["z_score"] is not None
        assert second_body["z_score"] < 0

    def test_uncertain_default_reports_without_enrolling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A gray-zone verdict with the default on_uncertain=report is not enrolled (unchanged behavior)."""
        puzzle_id = upload_test_puzzle()

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            first = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )
            assert first.json()["status"] == "new"

            # Force every comparison into the gray zone so the genuinely
            # different piece B lands there deterministically.
            monkeypatch.setattr(store_module.settings, "PIECE_GEOMETRY_T_ACCEPT", -1e9)
            monkeypatch.setattr(store_module.settings, "PIECE_GEOMETRY_T_NEW", 1e9)
            response = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(PIECE_B_EDGE_TYPES, PIECE_B_COLORS),
                headers=get_auth_header(),
            )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "uncertain"
        assert result["piece_id"] is None
        assert result["match_piece_id"] == "p001"
        assert result["z_score"] is not None

        listing = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=get_auth_header())
        assert len(listing.json()["pieces"]) == 1  # not enrolled

    def test_on_uncertain_enroll_enrolls_and_is_matchable_afterwards(self) -> None:
        """on_uncertain=enroll turns a gray-zone verdict into an enrollment; a re-scan then matches it."""
        puzzle_id = upload_test_puzzle()

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            first = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )
            assert first.json()["piece_id"] == "p001"

            # Gray-zone thresholds only for the escape-hatch post.
            with pytest.MonkeyPatch.context() as gray:
                gray.setattr(store_module.settings, "PIECE_GEOMETRY_T_ACCEPT", -1e9)
                gray.setattr(store_module.settings, "PIECE_GEOMETRY_T_NEW", 1e9)
                enrolled = client.post(
                    f"/api/v1/puzzle/{puzzle_id}/piece/geometry?on_uncertain=enroll",
                    files=geometry_files(PIECE_B_EDGE_TYPES, PIECE_B_COLORS),
                    headers=get_auth_header(),
                )

            assert enrolled.status_code == 200
            enrolled_body = enrolled.json()
            assert enrolled_body["status"] == "new"
            assert enrolled_body["piece_id"] == "p002"
            assert enrolled_body["z_score"] is not None

            # With normal thresholds restored, an identical re-scan of piece B
            # matches the piece the escape hatch enrolled.
            rescan = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(PIECE_B_EDGE_TYPES, PIECE_B_COLORS),
                headers=get_auth_header(),
            )

        rescan_body = rescan.json()
        assert rescan_body["status"] == "matched"
        assert rescan_body["piece_id"] == "p002"
        assert rescan_body["match_piece_id"] == "p002"

        listing = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=get_auth_header())
        assert [p["piece_id"] for p in listing.json()["pieces"]] == ["p001", "p002"]


class TestListPieceGeometry:
    """Tests for GET /api/v1/puzzle/{puzzle_id}/piece/geometry."""

    def test_requires_auth(self) -> None:
        """The endpoint rejects unauthenticated requests."""
        puzzle_id = upload_test_puzzle()

        response = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry")

        assert response.status_code in (401, 403)

    def test_unknown_puzzle_returns_404(self) -> None:
        """Listing a nonexistent puzzle_id returns 404."""
        response = client.get("/api/v1/puzzle/does-not-exist/piece/geometry", headers=get_auth_header())

        assert response.status_code == 404

    def test_empty_puzzle_has_no_pieces(self) -> None:
        """A freshly uploaded puzzle with no geometry scans yet has an empty piece list."""
        puzzle_id = upload_test_puzzle()

        response = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=get_auth_header())

        assert response.status_code == 200
        assert response.json() == {"puzzle_id": puzzle_id, "pieces": []}

    def test_lists_enrolled_pieces_without_polylines(self) -> None:
        """After one upload, the list endpoint reports the enrolled piece's summary (no polylines)."""
        puzzle_id = upload_test_puzzle()
        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=get_auth_header(),
            )

        response = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=get_auth_header())

        assert response.status_code == 200
        result = response.json()
        assert result["puzzle_id"] == puzzle_id
        assert len(result["pieces"]) == 1
        piece = result["pieces"][0]
        assert piece["piece_id"] == "p001"
        assert len(piece["edge_types"]) == 4
        assert piece["is_clean"] is True
        assert piece["corner_disagreement"] is False
        assert "polyline" not in piece


class TestPreviewIncludeQuality:
    """Tests for the /api/v1/piece/preview include_quality flag."""

    def _clean_piece_polygon(self) -> list[tuple[float, float]]:
        """A normalized polygon closely tracing a clean synthetic piece contour."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None
        height, width = mask.shape
        return [(float(x) / width, float(y) / height) for x, y in contour[::7]]

    def test_default_response_is_unchanged(self) -> None:
        """Without include_quality, lockable/corner_disagreement are null and detection is unaffected."""
        detector = MagicMock()
        detector.detect_region.return_value = DetectedRegion(
            polygon=[(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)],
            bbox=(0.1, 0.2, 0.7, 0.7),
            confidence=0.87,
        )

        with patch("app.main.get_piece_detector", return_value=detector):
            response = client.post(
                "/api/v1/piece/preview",
                files={"file": ("photo.jpg", BytesIO(make_photo_jpeg()), "image/jpeg")},
                headers=get_auth_header(),
            )

        assert response.status_code == 200
        result = response.json()
        assert result["found"] is True
        assert result["confidence"] == 0.87
        assert result["lockable"] is None
        assert result["corner_disagreement"] is None

    def test_include_quality_true_adds_flags_for_a_clean_region(self) -> None:
        """include_quality=true adds lockable=True/corner_disagreement=False for a clean piece polygon."""
        detector = MagicMock()
        detector.detect_region.return_value = DetectedRegion(
            polygon=self._clean_piece_polygon(),
            bbox=(0.1, 0.2, 0.7, 0.7),
            confidence=0.9,
        )

        with patch("app.main.get_piece_detector", return_value=detector):
            response = client.post(
                "/api/v1/piece/preview?include_quality=true",
                files={"file": ("photo.jpg", BytesIO(make_photo_jpeg()), "image/jpeg")},
                headers=get_auth_header(),
            )

        assert response.status_code == 200
        result = response.json()
        assert result["found"] is True
        assert result["corner_disagreement"] is False
        assert result["lockable"] is True

    def test_include_quality_true_with_not_found_stays_false(self) -> None:
        """When no region is detected, include_quality doesn't add flags (found=False short-circuits)."""
        detector = MagicMock()
        detector.detect_region.return_value = None

        with patch("app.main.get_piece_detector", return_value=detector):
            response = client.post(
                "/api/v1/piece/preview?include_quality=true",
                files={"file": ("photo.jpg", BytesIO(make_photo_jpeg()), "image/jpeg")},
                headers=get_auth_header(),
            )

        assert response.status_code == 200
        result = response.json()
        assert result["found"] is False
        assert result["lockable"] is None
        assert result["corner_disagreement"] is None
