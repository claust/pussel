"""Tests for the image processor's graceful degradation without a checkpoint."""

import asyncio
import io
from unittest.mock import patch

import pytest

from fastapi import UploadFile
from PIL import Image

from app.services import image_processor as ip_module
from app.services.image_processor import ImageProcessor


def make_upload(content: bytes) -> UploadFile:
    """Wrap raw bytes in an UploadFile for process_piece."""
    return UploadFile(filename="piece.jpg", file=io.BytesIO(content))


def make_jpeg() -> bytes:
    """Create a small in-memory JPEG."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_missing_checkpoint_loads_no_model() -> None:
    """With no checkpoint file, the processor loads no model instead of raising."""
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    assert processor.model is None


def test_missing_checkpoint_returns_neutral_fallback() -> None:
    """process_piece returns a neutral prediction when no model is available."""
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    result = asyncio.run(processor.process_piece(make_upload(make_jpeg()), "some-puzzle-id", remove_background=False))

    assert result.position.x == 0.5
    assert result.position.y == 0.5
    assert result.position_confidence == 0.0
    assert result.rotation == 0
    assert result.rotation_confidence == 0.0


def test_load_puzzle_tensor_rejects_non_uuid_puzzle_id() -> None:
    """_load_puzzle_tensor rejects path-like puzzle IDs before building a file path."""
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    with pytest.raises(FileNotFoundError):
        processor._load_puzzle_tensor("../etc/passwd")
