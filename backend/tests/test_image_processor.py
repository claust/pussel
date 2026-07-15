"""Tests for the image processor's graceful degradation without a checkpoint."""

import asyncio
import io
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fastapi import UploadFile
from PIL import Image

from app.services import image_processor as ip_module
from app.services.image_processor import ImageProcessor, _extract_state_dict, _load_checkpoint


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


def test_extract_state_dict_accepts_raw_state_dict() -> None:
    """A raw state_dict (exp20 checkpoint_best.pt) is returned unchanged."""
    raw = {"layer.weight": torch.zeros(2, 2)}

    assert _extract_state_dict(raw) is raw


def test_extract_state_dict_unwraps_model_state_dict() -> None:
    """A wrapped {"model_state_dict": ...} checkpoint (exp18-style) is unwrapped."""
    weights = {"layer.weight": torch.zeros(2, 2)}
    wrapped = {"model_state_dict": weights, "epoch": 5, "optimizer_state_dict": {}}

    assert _extract_state_dict(wrapped) is weights


def test_load_checkpoint_reraises_without_unsafe_optin() -> None:
    """When the safe load fails and the escape hatch is off, the error propagates.

    The unsafe ``weights_only=False`` retry must not run implicitly, so
    ``torch.load`` is expected to be called exactly once.
    """
    safe_error = RuntimeError("weights_only load rejected pickled object")

    with patch.object(ip_module, "ALLOW_UNSAFE_CHECKPOINT_LOAD", False):
        with patch.object(ip_module.torch, "load", side_effect=safe_error) as mock_load:
            with pytest.raises(RuntimeError, match="weights_only load rejected"):
                _load_checkpoint("some.pt", torch.device("cpu"))

    assert mock_load.call_count == 1
    assert mock_load.call_args.kwargs["weights_only"] is True


def test_load_checkpoint_unsafe_optin_retries_full_load() -> None:
    """With the escape hatch on, a failed safe load retries with a full pickle load."""
    sentinel = {"layer.weight": torch.zeros(1)}

    with patch.object(ip_module, "ALLOW_UNSAFE_CHECKPOINT_LOAD", True):
        with patch.object(ip_module.torch, "load", side_effect=[RuntimeError("unsafe"), sentinel]) as mock_load:
            result = _load_checkpoint("some.pt", torch.device("cpu"))

    assert result is sentinel
    assert mock_load.call_count == 2
    assert mock_load.call_args_list[1].kwargs["weights_only"] is False


def test_corrupt_checkpoint_falls_back_to_no_model(tmp_path: Path) -> None:
    """A checkpoint that exists but fails to load yields no model, not a crash."""
    bad_ckpt = tmp_path / "corrupt.pt"
    bad_ckpt.write_bytes(b"not a real torch checkpoint")

    with patch.object(ip_module, "CHECKPOINT_PATH", str(bad_ckpt)):
        processor = ImageProcessor()

    assert processor.model is None


def test_load_puzzle_tensor_rejects_non_uuid_puzzle_id() -> None:
    """_load_puzzle_tensor rejects path-like puzzle IDs before building a file path."""
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    with pytest.raises(FileNotFoundError):
        processor._load_puzzle_tensor("../etc/passwd")
