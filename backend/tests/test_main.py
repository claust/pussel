"""Test module for the FastAPI puzzle solver application."""

import os
import shutil
import sys
import tempfile
from io import BufferedReader
from pathlib import Path
from typing import Generator, cast

import pytest
from fastapi.testclient import TestClient
from typing_extensions import TypeAlias

from app.main import app, settings

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

# Third-party imports

# Local imports

client = TestClient(app)

# Type alias for file upload tuple
FileUpload: TypeAlias = tuple[str, BufferedReader, str]


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
            response = client.post("/api/v1/puzzle/upload", files=files)

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
        response = client.post("/api/v1/puzzle/invalid-id/piece", files=files)
        assert response.status_code == 404


def test_process_piece() -> None:
    """Test processing a valid puzzle piece."""
    # First upload a puzzle
    with open("test_puzzle.jpg", "wb") as f:
        f.write(b"fake image content")

    try:
        # Upload puzzle
        with open("test_puzzle.jpg", "rb") as f:
            files = {"file": ("test_puzzle.jpg", f, "image/jpeg")}
            response = client.post("/api/v1/puzzle/upload", files=files)
        puzzle_id = response.json()["puzzle_id"]

        # Test piece processing
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
            )

        assert response.status_code == 200
        result = response.json()
        assert "position" in result
        assert "x" in result["position"]
        assert "y" in result["position"]
        assert "confidence" in result
        assert "rotation" in result

        # Verify confidence is between 0.5 and 1.0
        assert 0.5 <= result["confidence"] <= 1.0

        # Verify rotation is one of the expected values
        assert result["rotation"] in [0, 90, 180, 270]

    finally:
        # Cleanup test image
        if os.path.exists("test_puzzle.jpg"):
            os.remove("test_puzzle.jpg")
