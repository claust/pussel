"""Tests for the classical SIFT->NCC hybrid puzzle-piece matcher."""

import asyncio
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import jwt
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import settings
from app.main import app
from app.models.puzzle_model import PieceResponse, PieceSpan, Position
from app.services import classical_matcher as cm_module
from app.services.classical_matcher import ClassicalMatcher, PuzzleFeatures, get_classical_matcher

client = TestClient(app)

# A fixed sub-rectangle of the synthetic puzzle used as the "true" piece
# location for the happy-path tests below.
PUZZLE_SIZE = 512
CROP_X1, CROP_Y1 = 150, 150
CROP_SIZE = 150
CROP_X2, CROP_Y2 = CROP_X1 + CROP_SIZE, CROP_Y1 + CROP_SIZE
EXPECTED_NX = (CROP_X1 + CROP_X2) / 2 / PUZZLE_SIZE
EXPECTED_NY = (CROP_Y1 + CROP_Y2) / 2 / PUZZLE_SIZE
POSITION_TOLERANCE = 0.15

# make_piece_canvas pads the crop with a black border on each side; the full
# piece frame (crop + padding) is what piece_span should describe.
CANVAS_PAD = 20
CANVAS_SIDE = CROP_SIZE + 2 * CANVAS_PAD
EXPECTED_SPAN = CANVAS_SIDE / PUZZLE_SIZE
SPAN_TOLERANCE = 0.3  # generous: SIFT/NCC scale estimation is approximate


def make_synthetic_puzzle(size: int = PUZZLE_SIZE, seed: int = 42) -> Image.Image:
    """Build a textured RGB puzzle image with enough structure for SIFT keypoints.

    A smooth gradient background plus hundreds of small colored shapes gives
    SIFT plenty of corner-like features; pure noise is SIFT-hostile.
    """
    rng = np.random.default_rng(seed)
    coords = np.linspace(0, 255, size)
    xv, yv = np.meshgrid(coords, coords)
    channels = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)
    img = Image.fromarray(channels, mode="RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(600):
        cx, cy = (int(v) for v in rng.integers(0, size, size=2))
        rad = int(rng.integers(4, 22))
        rc, gc, bc = (int(c) for c in rng.integers(0, 255, size=3))
        color: tuple[int, int, int] = (rc, gc, bc)
        shape_type = int(rng.integers(0, 3))
        if shape_type == 0:
            draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=color)
        elif shape_type == 1:
            draw.rectangle((cx - rad, cy - rad, cx + rad, cy + rad), fill=color)
        else:
            pts = [(int(cx + rng.integers(-rad, rad)), int(cy + rng.integers(-rad, rad))) for _ in range(5)]
            draw.polygon(pts, fill=color)
    return img


def make_piece_canvas(puzzle_rgb: "np.ndarray", pad: int = 20) -> "np.ndarray":
    """Crop the fixed test region and composite it on a black square canvas."""
    crop = puzzle_rgb[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
    canvas = np.zeros((CROP_SIZE + 2 * pad, CROP_SIZE + 2 * pad, 3), dtype=np.uint8)
    canvas[pad : pad + CROP_SIZE, pad : pad + CROP_SIZE] = crop
    return canvas


def encode_jpeg(rgb: "np.ndarray") -> bytes:
    """Encode an RGB array as JPEG bytes."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def make_upload(content: bytes) -> UploadFile:
    """Wrap raw bytes in an UploadFile for process_piece."""
    return UploadFile(filename="piece.jpg", file=io.BytesIO(content))


@pytest.fixture()
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point settings.UPLOAD_DIR at a throwaway directory for the duration of a test."""
    monkeypatch.setattr(cm_module.settings, "UPLOAD_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def puzzle_id(upload_dir: Path) -> str:
    """Write a synthetic puzzle image to the (patched) upload dir and return its ID."""
    new_id = str(uuid4())
    make_synthetic_puzzle().save(upload_dir / f"{new_id}.jpg", format="JPEG", quality=95)
    return new_id


@pytest.fixture()
def matcher() -> ClassicalMatcher:
    """A fresh ClassicalMatcher instance (its own cache, independent of the singleton)."""
    return ClassicalMatcher()


def test_sift_happy_path(matcher: ClassicalMatcher, puzzle_id: str) -> None:
    """A crop of the puzzle, uploaded unrotated, is located near its true center via SIFT."""
    puzzle_rgb = np.array(make_synthetic_puzzle())
    canvas = make_piece_canvas(puzzle_rgb)

    result = asyncio.run(matcher.process_piece(make_upload(encode_jpeg(canvas)), puzzle_id, remove_background=False))

    assert abs(result.position.x - EXPECTED_NX) < POSITION_TOLERANCE
    assert abs(result.position.y - EXPECTED_NY) < POSITION_TOLERANCE
    assert result.rotation == 0
    assert result.position_confidence > 0
    assert result.rotation_confidence > 0
    assert result.piece_span is not None
    assert abs(result.piece_span.width - EXPECTED_SPAN) < SPAN_TOLERANCE
    assert abs(result.piece_span.height - EXPECTED_SPAN) < SPAN_TOLERANCE


def test_sift_rotation(matcher: ClassicalMatcher, puzzle_id: str) -> None:
    """A piece rotated 90 degrees (np.rot90 k=1) is reported with the exp25 rotation convention.

    exp25's convention maps a counter-clockwise ``np.rot90(arr, k)`` rotation of
    the *observed* piece to ``rotation_degrees = (-k * 90) % 360`` (verified
    empirically against the ported ``_predict_sift``): k=1 -> 270 degrees.
    """
    puzzle_rgb = np.array(make_synthetic_puzzle())
    canvas = make_piece_canvas(puzzle_rgb)
    rotated = np.ascontiguousarray(np.rot90(canvas, k=1))

    result = asyncio.run(matcher.process_piece(make_upload(encode_jpeg(rotated)), puzzle_id, remove_background=False))

    assert result.rotation == 270
    assert result.position_confidence > 0


def test_sift_fail_falls_back_to_ncc(
    matcher: ClassicalMatcher, puzzle_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When SIFT can't match, NCC still produces a non-neutral prediction."""
    monkeypatch.setattr(matcher, "_predict_sift", lambda *_args, **_kwargs: None)
    puzzle_rgb = np.array(make_synthetic_puzzle())
    canvas = make_piece_canvas(puzzle_rgb)

    result = asyncio.run(matcher.process_piece(make_upload(encode_jpeg(canvas)), puzzle_id, remove_background=False))

    assert result.position_confidence > 0
    assert result.rotation_confidence > 0
    assert not (result.position.x == 0.5 and result.position.y == 0.5 and result.position_confidence == 0.0)
    # The NCC template is square in overview pixel space, so the span should be
    # square-ish even when the puzzle overview itself isn't perfectly square.
    assert result.piece_span is not None
    assert result.piece_span.width > 0
    assert result.piece_span.height > 0
    assert abs(result.piece_span.width - result.piece_span.height) < 0.05


def test_both_fail_returns_neutral(matcher: ClassicalMatcher, puzzle_id: str) -> None:
    """A fully black piece (nothing above MASK_THRESHOLD) yields the exact neutral response."""
    black_piece = np.zeros((150, 150, 3), dtype=np.uint8)

    result = asyncio.run(
        matcher.process_piece(make_upload(encode_jpeg(black_piece)), puzzle_id, remove_background=False)
    )

    assert result.position.x == 0.5
    assert result.position.y == 0.5
    assert result.position_confidence == 0.0
    assert result.rotation == 0
    assert result.rotation_confidence == 0.0
    assert result.cleaned_image is None
    assert result.piece_span is None


def test_unexpected_exception_returns_neutral_fallback(
    matcher: ClassicalMatcher, puzzle_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unexpected error during matching degrades to the neutral fallback, not a crash."""
    monkeypatch.setattr(matcher, "_predict_sift", MagicMock(side_effect=RuntimeError("boom")))
    puzzle_rgb = np.array(make_synthetic_puzzle())
    canvas = make_piece_canvas(puzzle_rgb)

    result = asyncio.run(matcher.process_piece(make_upload(encode_jpeg(canvas)), puzzle_id, remove_background=False))

    assert result.position.x == 0.5
    assert result.position.y == 0.5
    assert result.position_confidence == 0.0
    assert result.rotation == 0
    assert result.rotation_confidence == 0.0
    assert result.cleaned_image is None
    assert result.piece_span is None


def test_cache_populated_after_process_piece(matcher: ClassicalMatcher, puzzle_id: str) -> None:
    """process_piece populates the per-puzzle feature cache."""
    puzzle_rgb = np.array(make_synthetic_puzzle())
    canvas = make_piece_canvas(puzzle_rgb)

    asyncio.run(matcher.process_piece(make_upload(encode_jpeg(canvas)), puzzle_id, remove_background=False))

    assert puzzle_id in matcher._puzzle_cache
    assert isinstance(matcher._puzzle_cache[puzzle_id], PuzzleFeatures)


def test_clear_puzzle_cache_single(matcher: ClassicalMatcher, puzzle_id: str) -> None:
    """clear_puzzle_cache(puzzle_id) removes only that entry."""
    matcher._load_puzzle_features(puzzle_id)
    other_id = "not-cached-but-present"
    matcher._puzzle_cache[other_id] = matcher._puzzle_cache[puzzle_id]

    matcher.clear_puzzle_cache(puzzle_id)

    assert puzzle_id not in matcher._puzzle_cache
    assert other_id in matcher._puzzle_cache


def test_clear_puzzle_cache_all(matcher: ClassicalMatcher, puzzle_id: str) -> None:
    """clear_puzzle_cache() with no argument clears every entry."""
    matcher._load_puzzle_features(puzzle_id)

    matcher.clear_puzzle_cache()

    assert matcher._puzzle_cache == {}


def test_load_puzzle_features_unknown_uuid_raises(matcher: ClassicalMatcher, upload_dir: Path) -> None:
    """A well-formed but nonexistent puzzle UUID raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        matcher._load_puzzle_features(str(uuid4()))


def test_load_puzzle_features_invalid_id_raises(matcher: ClassicalMatcher, upload_dir: Path) -> None:
    """A non-UUID puzzle ID raises FileNotFoundError before touching the filesystem."""
    with pytest.raises(FileNotFoundError):
        matcher._load_puzzle_features("../etc/passwd")


def test_process_piece_reraises_file_not_found(matcher: ClassicalMatcher, upload_dir: Path) -> None:
    """process_piece re-raises FileNotFoundError instead of returning a neutral fallback."""
    black_piece = np.zeros((150, 150, 3), dtype=np.uint8)

    with pytest.raises(FileNotFoundError):
        asyncio.run(matcher.process_piece(make_upload(encode_jpeg(black_piece)), str(uuid4()), remove_background=False))


def test_get_classical_matcher_singleton() -> None:
    """get_classical_matcher returns the same instance across calls."""
    assert get_classical_matcher() is get_classical_matcher()


# --- Endpoint switch (settings.MATCHER selects the matcher service) ---


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


def _upload_puzzle(upload_dir: Path) -> str:
    """Upload a throwaway puzzle image through the API and return its puzzle_id."""
    files = {"file": ("test_puzzle.jpg", io.BytesIO(b"fake image content"), "image/jpeg")}
    response = client.post("/api/v1/puzzle/upload", files=files, headers=get_auth_header())
    return cast(str, response.json()["puzzle_id"])


def test_endpoint_uses_classical_matcher_when_configured(upload_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With settings.MATCHER == "classical", the piece endpoint calls get_classical_matcher."""
    monkeypatch.setattr(settings, "MATCHER", "classical")
    puzzle_id = _upload_puzzle(upload_dir)
    mock_matcher = MagicMock()
    mock_matcher.process_piece = AsyncMock(
        return_value=PieceResponse(
            position=Position(x=0.1, y=0.2),
            position_confidence=0.5,
            rotation=90,
            rotation_confidence=0.5,
            piece_span=PieceSpan(width=0.3, height=0.35),
        )
    )

    with (
        patch("app.main.get_classical_matcher", return_value=mock_matcher) as mock_get_classical,
        patch("app.main.get_image_processor") as mock_get_cnn,
    ):
        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/piece",
            files={"file": ("piece.jpg", io.BytesIO(b"fake piece content"), "image/jpeg")},
            headers=get_auth_header(),
        )

    assert response.status_code == 200
    mock_get_classical.assert_called_once()
    mock_get_cnn.assert_not_called()
    body = response.json()
    assert body["rotation"] == 90
    # Confirms the wire shape: snake_case "piece_span" with "width"/"height" sub-keys.
    assert body["piece_span"] == {"width": 0.3, "height": 0.35}


def test_endpoint_uses_cnn_processor_when_configured(upload_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With settings.MATCHER == "cnn", the piece endpoint calls get_image_processor."""
    monkeypatch.setattr(settings, "MATCHER", "cnn")
    puzzle_id = _upload_puzzle(upload_dir)
    mock_processor = MagicMock()
    mock_processor.process_piece = AsyncMock(
        return_value=PieceResponse(
            position=Position(x=0.3, y=0.4), position_confidence=0.6, rotation=180, rotation_confidence=0.6
        )
    )

    with (
        patch("app.main.get_image_processor", return_value=mock_processor) as mock_get_cnn,
        patch("app.main.get_classical_matcher") as mock_get_classical,
    ):
        response = client.post(
            f"/api/v1/puzzle/{puzzle_id}/piece",
            files={"file": ("piece.jpg", io.BytesIO(b"fake piece content"), "image/jpeg")},
            headers=get_auth_header(),
        )

    assert response.status_code == 200
    mock_get_cnn.assert_called_once()
    mock_get_classical.assert_not_called()
    assert response.json()["rotation"] == 180
