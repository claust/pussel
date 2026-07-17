"""Tests for puzzle ownership (IDOR fix): every {puzzle_id} endpoint is owner-scoped.

Mirrors the token/auth patterns in tests/test_auth.py and tests/test_main.py,
and the mocked piece-geometry pipeline in tests/test_piece_geometry_api.py.
"""

import os
import shutil
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi.testclient import TestClient
from piece_geometry_fixtures import PIECE_A_COLORS, PIECE_A_EDGE_TYPES, encode_png, make_piece_rgba
from PIL import Image, ImageDraw

from app.config import settings
from app.main import app
from app.models.puzzle_model import PieceResponse, Position
from app.services.piece_geometry.service import PieceGeometryService
from app.services.puzzle_store import MAX_PUZZLES, PuzzleRecord, PuzzleStore, get_puzzle_store

client = TestClient(app)


@pytest.fixture(autouse=True)
def _ensure_upload_dir() -> Generator[None, None, None]:
    """Recreate UPLOAD_DIR before each test and reset the puzzle store singleton after.

    Resetting avoids this file's uploads (owners "user-a"/"user-b") lingering
    in the shared, process-wide `get_puzzle_store()` singleton across tests.
    """
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield
    shutil.rmtree(settings.UPLOAD_DIR, ignore_errors=True)
    get_puzzle_store.cache_clear()


def create_test_token(user_id: str) -> str:
    """Create a valid test JWT token for the given user (mirrors tests/test_auth.py)."""
    expire = datetime.now(timezone.utc) + timedelta(hours=1)
    payload = {
        "sub": user_id,
        "email": f"{user_id}@example.com",
        "name": user_id,
        "picture": None,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def auth_header(user_id: str) -> dict[str, str]:
    """Create an authorization header for the given user."""
    return {"Authorization": f"Bearer {create_test_token(user_id)}"}


USER_A = "user-a"
USER_B = "user-b"


def make_photo_jpeg() -> bytes:
    """Create an in-memory JPEG photo with a bright rectangle on a dark background."""
    image = Image.new("RGB", (800, 600), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 80, 700, 520), fill=(220, 210, 190))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def upload_puzzle_as(user_id: str) -> str:
    """Upload a puzzle as the given user and return its puzzle_id."""
    response = client.post(
        "/api/v1/puzzle/upload",
        files={"file": ("photo.jpg", BytesIO(make_photo_jpeg()), "image/jpeg")},
        headers=auth_header(user_id),
    )
    assert response.status_code == 200
    return str(response.json()["puzzle_id"])


def mocked_geometry_service() -> PieceGeometryService:
    """A PieceGeometryService whose mocked background remover echoes the uploaded RGBA PNG."""
    remover = MagicMock()
    remover.remove_background.side_effect = lambda image_bytes: Image.open(BytesIO(image_bytes)).convert("RGBA")
    return PieceGeometryService(background_remover=remover)


def geometry_files() -> dict[str, tuple[str, BytesIO, str]]:
    """Build the multipart files dict for a piece-geometry photo upload."""
    rgba = make_piece_rgba(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
    return {"file": ("piece.png", BytesIO(encode_png(rgba)), "image/png")}


def piece_files() -> dict[str, tuple[str, BytesIO, str]]:
    """Build the multipart files dict for a piece-processing upload."""
    return {"file": ("piece.jpg", BytesIO(b"fake piece content"), "image/jpeg")}


class TestProcessPieceOwnership:
    """Tests for POST /api/v1/puzzle/{puzzle_id}/piece."""

    def test_owner_can_process_piece(self) -> None:
        """User A can process a piece against their own puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        mock_response = PieceResponse(
            position=Position(x=0.5, y=0.5), position_confidence=0.5, rotation=0, rotation_confidence=0.5
        )
        mock_matcher = MagicMock()
        mock_matcher.process_piece = AsyncMock(return_value=mock_response)

        with patch("app.main.get_classical_matcher", return_value=mock_matcher):
            response = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece",
                files=piece_files(),
                headers=auth_header(USER_A),
            )

        assert response.status_code == 200

    def test_non_owner_gets_404(self) -> None:
        """User B gets 404 (not 403) trying to process a piece against user A's puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/piece",
            files=piece_files(),
            headers=auth_header(USER_B),
        )

        assert response.status_code == 404


class TestPieceGeometryOwnership:
    """Tests for POST/GET /api/v1/puzzle/{puzzle_id}/piece/geometry."""

    def test_owner_can_upload_geometry(self) -> None:
        """User A can upload piece geometry against their own puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            response = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=auth_header(USER_A),
            )

        assert response.status_code == 200
        assert response.json()["status"] == "new"

    def test_non_owner_gets_404_on_upload(self) -> None:
        """User B gets 404 uploading geometry against user A's puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        with patch("app.main.get_piece_geometry_service", return_value=mocked_geometry_service()):
            response = client.post(
                f"/api/v1/puzzle/{puzzle_id}/piece/geometry",
                files=geometry_files(),
                headers=auth_header(USER_B),
            )

        assert response.status_code == 404

    def test_owner_can_list_geometry(self) -> None:
        """User A can list piece geometry for their own puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=auth_header(USER_A))

        assert response.status_code == 200

    def test_non_owner_gets_404_on_list(self) -> None:
        """User B gets 404 listing geometry for user A's puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.get(f"/api/v1/puzzle/{puzzle_id}/piece/geometry", headers=auth_header(USER_B))

        assert response.status_code == 404


class TestGeneratePieceOwnership:
    """Tests for POST /api/v1/puzzle/{puzzle_id}/generate-piece."""

    def test_owner_can_generate_piece(self) -> None:
        """User A can generate a piece from their own puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/generate-piece",
            json={"center_x": 0.5, "center_y": 0.5},
            headers=auth_header(USER_A),
        )

        assert response.status_code == 200
        assert response.json()["piece_image"].startswith("data:image/png;base64,")

    def test_non_owner_gets_404(self) -> None:
        """User B gets 404 generating a piece from user A's puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/generate-piece",
            json={"center_x": 0.5, "center_y": 0.5},
            headers=auth_header(USER_B),
        )

        assert response.status_code == 404


class TestCutAllOwnership:
    """Tests for POST /api/v1/puzzle/{puzzle_id}/cut-all — previously unauthenticated."""

    def test_owner_can_cut_puzzle(self) -> None:
        """User A can cut their own puzzle into pieces."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/cut-all",
            json={"rows": 2, "cols": 2},
            headers=auth_header(USER_A),
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["pieces"]) == 4

    def test_non_owner_gets_404(self) -> None:
        """User B gets 404 cutting user A's puzzle."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/cut-all",
            json={"rows": 2, "cols": 2},
            headers=auth_header(USER_B),
        )

        assert response.status_code == 404

    def test_no_token_returns_401(self) -> None:
        """Regression test: cut-all previously had no auth dependency at all."""
        puzzle_id = upload_puzzle_as(USER_A)

        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/cut-all",
            json={"rows": 2, "cols": 2},
        )

        assert response.status_code == 401


class TestListPuzzles:
    """Tests for GET /api/v1/puzzles."""

    def test_returns_only_callers_own_puzzles(self) -> None:
        """The listing includes the caller's puzzles and excludes other users'."""
        puzzle_a1 = upload_puzzle_as(USER_A)
        puzzle_a2 = upload_puzzle_as(USER_A)
        puzzle_b = upload_puzzle_as(USER_B)

        response = client.get("/api/v1/puzzles", headers=auth_header(USER_A))

        assert response.status_code == 200
        puzzle_ids = {p["puzzle_id"] for p in response.json()["puzzles"]}
        assert puzzle_ids == {puzzle_a1, puzzle_a2}
        assert puzzle_b not in puzzle_ids

    def test_requires_auth(self) -> None:
        """Listing puzzles requires authentication."""
        response = client.get("/api/v1/puzzles")

        assert response.status_code in (401, 403)


class TestPuzzleStoreEviction:
    """Tests for PuzzleStore's FIFO eviction cap (MAX_PUZZLES)."""

    def test_fifo_eviction_drops_oldest_and_cleans_up(self, tmp_path: Path) -> None:
        """Adding past MAX_PUZZLES evicts the oldest record and cleans up its resources."""
        store = PuzzleStore()
        mock_classical = MagicMock()
        mock_image_processor = MagicMock()
        mock_geometry_store = MagicMock()

        with (
            patch("app.services.classical_matcher.get_classical_matcher", return_value=mock_classical),
            patch("app.services.image_processor.get_image_processor", return_value=mock_image_processor),
            patch("app.services.piece_geometry.store.get_piece_geometry_store", return_value=mock_geometry_store),
        ):
            for i in range(MAX_PUZZLES + 1):
                puzzle_id = f"puzzle-{i}"
                file_path = tmp_path / f"{puzzle_id}.jpg"
                file_path.write_bytes(b"fake")
                store.add(PuzzleRecord(puzzle_id=puzzle_id, owner_id=USER_A, file_path=str(file_path), grid=None))

        # The oldest puzzle was evicted; the rest (including the newest) remain.
        assert store.get("puzzle-0") is None
        assert store.get("puzzle-1") is not None
        assert store.get(f"puzzle-{MAX_PUZZLES}") is not None
        assert len(store.list_for_owner(USER_A)) == MAX_PUZZLES

        # Cleanup happened for the evicted puzzle: file removed, caches cleared.
        assert not (tmp_path / "puzzle-0.jpg").exists()
        mock_classical.clear_puzzle_cache.assert_any_call("puzzle-0")
        mock_image_processor.clear_puzzle_cache.assert_any_call("puzzle-0")
        mock_geometry_store.drop.assert_any_call("puzzle-0")

    def test_eviction_tolerates_missing_file(self, tmp_path: Path) -> None:
        """Eviction doesn't blow up when the puzzle's file is already gone."""
        store = PuzzleStore()
        missing_path = tmp_path / "already-gone.jpg"

        with (
            patch("app.services.classical_matcher.get_classical_matcher", return_value=MagicMock()),
            patch("app.services.image_processor.get_image_processor", return_value=MagicMock()),
            patch("app.services.piece_geometry.store.get_piece_geometry_store", return_value=MagicMock()),
        ):
            store.add(PuzzleRecord(puzzle_id="evictee", owner_id=USER_A, file_path=str(missing_path), grid=None))
            for i in range(MAX_PUZZLES):
                store.add(
                    PuzzleRecord(puzzle_id=f"filler-{i}", owner_id=USER_A, file_path=str(missing_path), grid=None)
                )

        assert store.get("evictee") is None
