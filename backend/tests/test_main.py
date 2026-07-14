"""Test module for the FastAPI puzzle solver application."""

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from io import BufferedReader, BytesIO
from pathlib import Path
from typing import Generator, cast
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from typing_extensions import TypeAlias

from app.main import app, settings
from app.models.puzzle_model import PieceResponse, Position
from app.services.piece_detector import DetectedRegion

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Third-party imports

# Local imports

client = TestClient(app)

# Type alias for file upload tuple
FileUpload: TypeAlias = tuple[str, BufferedReader, str]


def create_test_token() -> str:
    """Create a valid test JWT token for authentication."""
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


@pytest.fixture(autouse=True)
def setup_and_cleanup() -> Generator[None, None, None]:
    """Set up test environment and clean up after tests."""
    # Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    yield

    # Cleanup after tests
    shutil.rmtree(settings.UPLOAD_DIR)


def test_health_check() -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_upload_puzzle() -> None:
    """Test uploading a puzzle image."""
    # Create a test image
    test_image_path = "test_puzzle.jpg"
    with open(test_image_path, "wb") as f:
        f.write(b"fake image content")

    try:
        with open(test_image_path, "rb") as f:
            files = {"file": ("test_puzzle.jpg", f, "image/jpeg")}
            response = client.post("/api/v1/puzzle/upload", files=files, headers=get_auth_header())

        assert response.status_code == 200
        assert "puzzle_id" in response.json()
        puzzle_id = response.json()["puzzle_id"]
        puzzle_path = os.path.join(settings.UPLOAD_DIR, f"{puzzle_id}.jpg")
        assert os.path.exists(puzzle_path)

    finally:
        # Cleanup test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)


def test_process_piece_invalid_puzzle() -> None:
    """Test processing a piece with an invalid puzzle ID."""
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        temp_file.write(b"fake piece content")
        temp_file.seek(0)
        file_tuple = cast(
            FileUpload,
            ("test_piece.jpg", temp_file, "image/jpeg"),
        )
        files = {"file": file_tuple}
        response = client.post("/api/v1/puzzle/invalid-id/piece", files=files, headers=get_auth_header())
        assert response.status_code == 404


def test_process_piece() -> None:
    """Test processing a valid puzzle piece."""
    # Create mock response
    mock_response = PieceResponse(
        position=Position(x=0.25, y=0.75),
        position_confidence=0.85,
        rotation=90,
        rotation_confidence=0.90,
    )

    # Mock the image processor
    mock_processor = MagicMock()
    mock_processor.process_piece = AsyncMock(return_value=mock_response)

    # First upload a puzzle
    with open("test_puzzle.jpg", "wb") as f:
        f.write(b"fake image content")

    try:
        # Upload puzzle
        with open("test_puzzle.jpg", "rb") as f:
            files = {"file": ("test_puzzle.jpg", f, "image/jpeg")}
            response = client.post("/api/v1/puzzle/upload", files=files, headers=get_auth_header())
        puzzle_id = response.json()["puzzle_id"]

        # Test piece processing with mocked image processor
        with patch(
            "app.main.get_image_processor",
            return_value=mock_processor,
        ):
            with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
                temp_file.write(b"fake piece content")
                temp_file.seek(0)
                file_tuple = cast(
                    FileUpload,
                    ("test_piece.jpg", temp_file, "image/jpeg"),
                )
                files = {"file": file_tuple}
                response = client.post(
                    f"/api/v1/puzzle/{puzzle_id}/piece",
                    files=files,
                    headers=get_auth_header(),
                )

        assert response.status_code == 200
        result = response.json()
        assert "position" in result
        assert "x" in result["position"]
        assert "y" in result["position"]
        assert "position_confidence" in result
        assert "rotation" in result
        assert "rotation_confidence" in result

        # Verify the mocked values are returned
        assert result["position"]["x"] == 0.25
        assert result["position"]["y"] == 0.75
        assert result["position_confidence"] == 0.85
        assert result["rotation"] == 90
        assert result["rotation_confidence"] == 0.90

    finally:
        # Cleanup test image
        if os.path.exists("test_puzzle.jpg"):
            os.remove("test_puzzle.jpg")


def make_photo_jpeg() -> bytes:
    """Create an in-memory JPEG photo with a bright rectangle on a dark background."""
    image = Image.new("RGB", (800, 600), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 80, 700, 520), fill=(220, 210, 190))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def photo_files(content: bytes) -> dict[str, tuple[str, BytesIO, str]]:
    """Build the multipart files dict for a photo upload."""
    return {"file": ("photo.jpg", BytesIO(content), "image/jpeg")}


def test_detect_frame_requires_auth() -> None:
    """Detect-frame rejects unauthenticated requests."""
    response = client.post("/api/v1/puzzle/detect-frame", files=photo_files(make_photo_jpeg()))
    assert response.status_code in (401, 403)


def test_detect_frame_no_file() -> None:
    """Detect-frame returns 400 when no file is provided."""
    response = client.post("/api/v1/puzzle/detect-frame", headers=get_auth_header())
    assert response.status_code == 400


def test_detect_frame_detects_rectangle() -> None:
    """Detect-frame finds the rectangle and returns a trimmed preview."""
    response = client.post(
        "/api/v1/puzzle/detect-frame",
        files=photo_files(make_photo_jpeg()),
        headers=get_auth_header(),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["trimmed_image"].startswith("data:image/jpeg;base64,")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence"] > 0.5
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        corner = result["corners"][name]
        assert 0.0 <= corner["x"] <= 1.0
        assert 0.0 <= corner["y"] <= 1.0
    # Detected corners should be close to the drawn rectangle (100,80)-(700,520) in 800x600
    assert abs(result["corners"]["top_left"]["x"] - 100 / 800) < 0.03
    assert abs(result["corners"]["top_left"]["y"] - 80 / 600) < 0.03


def test_detect_frame_manual_corners() -> None:
    """Detect-frame uses manually supplied corners with confidence 1.0."""
    manual_corners = {
        "top_left": {"x": 0.125, "y": 0.1333},
        "top_right": {"x": 0.875, "y": 0.1333},
        "bottom_right": {"x": 0.875, "y": 0.8667},
        "bottom_left": {"x": 0.125, "y": 0.8667},
    }

    response = client.post(
        "/api/v1/puzzle/detect-frame",
        files=photo_files(make_photo_jpeg()),
        data={"corners": json.dumps(manual_corners)},
        headers=get_auth_header(),
    )

    assert response.status_code == 200
    result = response.json()
    assert result["confidence"] == 1.0
    assert result["corners"] == manual_corners
    assert result["trimmed_image"].startswith("data:image/jpeg;base64,")


def test_detect_frame_invalid_corners_json() -> None:
    """Detect-frame returns 400 for a malformed corners payload."""
    response = client.post(
        "/api/v1/puzzle/detect-frame",
        files=photo_files(make_photo_jpeg()),
        data={"corners": "not valid json"},
        headers=get_auth_header(),
    )
    assert response.status_code == 400


def test_detect_frame_invalid_image_bytes() -> None:
    """Detect-frame returns 400 when the file is not a decodable image."""
    response = client.post(
        "/api/v1/puzzle/detect-frame",
        files=photo_files(b"fake image content"),
        headers=get_auth_header(),
    )
    assert response.status_code == 400


def test_piece_preview_requires_auth() -> None:
    """Piece preview rejects unauthenticated requests."""
    response = client.post("/api/v1/piece/preview", files=photo_files(make_photo_jpeg()))
    assert response.status_code in (401, 403)


def test_piece_preview_no_file() -> None:
    """Piece preview returns 400 when no file is provided."""
    response = client.post("/api/v1/piece/preview", headers=get_auth_header())
    assert response.status_code == 400


def test_piece_preview_invalid_image_bytes() -> None:
    """Piece preview returns 400 for non-image bytes."""
    response = client.post(
        "/api/v1/piece/preview",
        files=photo_files(b"fake image content"),
        headers=get_auth_header(),
    )
    assert response.status_code == 400


def test_piece_preview_rejects_oversized_file() -> None:
    """Piece preview returns 413 when the frame exceeds the upload size limit."""
    oversized = b"\x00" * (settings.MAX_UPLOAD_SIZE + 1)
    response = client.post(
        "/api/v1/piece/preview",
        files=photo_files(oversized),
        headers=get_auth_header(),
    )
    assert response.status_code == 413


def test_piece_preview_returns_region() -> None:
    """Piece preview maps a detected region to polygon and bbox."""
    detector = MagicMock()
    detector.detect_region.return_value = DetectedRegion(
        polygon=[(0.1, 0.2), (0.8, 0.2), (0.8, 0.9), (0.1, 0.9)],
        bbox=(0.1, 0.2, 0.7, 0.7),
        confidence=0.87,
    )

    with patch("app.main.get_piece_detector", return_value=detector):
        response = client.post(
            "/api/v1/piece/preview",
            files=photo_files(make_photo_jpeg()),
            headers=get_auth_header(),
        )

    assert response.status_code == 200
    result = response.json()
    assert result["found"] is True
    assert len(result["polygon"]) == 4
    assert result["polygon"][0] == {"x": 0.1, "y": 0.2}
    assert result["bbox"] == {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.7}
    assert result["confidence"] == 0.87


def test_piece_preview_not_found() -> None:
    """Piece preview reports found=False when nothing is detected."""
    detector = MagicMock()
    detector.detect_region.return_value = None

    with patch("app.main.get_piece_detector", return_value=detector):
        response = client.post(
            "/api/v1/piece/preview",
            files=photo_files(make_photo_jpeg()),
            headers=get_auth_header(),
        )

    assert response.status_code == 200
    result = response.json()
    assert result["found"] is False
    assert result["polygon"] == []
    assert result["confidence"] == 0.0
    assert result["bbox"] is None
