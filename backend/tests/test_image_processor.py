"""Tests for the image processor's graceful degradation without a checkpoint."""

import asyncio
import base64
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from fastapi import UploadFile
from PIL import Image
from torch import Tensor

from app.services import image_processor as ip_module
from app.services.image_processor import (
    PIECE_SIZE,
    PUZZLE_SIZE,
    ImageProcessor,
    _bound_longest_side,
    _composite_on_black,
    _extract_state_dict,
    _load_checkpoint,
    _pad_to_square,
)


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


def test_composite_on_black_renders_transparent_regions_black() -> None:
    """Transparent pixels become black while opaque pixels keep their color."""
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgba.putpixel((1, 1), (200, 50, 25, 255))

    rgb = _composite_on_black(rgba)

    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (0, 0, 0)
    assert rgb.getpixel((1, 1)) == (200, 50, 25)


def test_pad_to_square_centers_content_on_black() -> None:
    """A non-square image is centered on a black square canvas of the longer side."""
    img = Image.new("RGB", (10, 4), (255, 255, 255))

    square = _pad_to_square(img)

    assert square.size == (10, 10)
    assert square.getpixel((5, 0)) == (0, 0, 0)  # padding row
    assert square.getpixel((5, 5)) == (255, 255, 255)  # original content


def test_pad_to_square_returns_square_input_unchanged() -> None:
    """An already-square image is returned as-is."""
    img = Image.new("RGB", (8, 8), (1, 2, 3))

    assert _pad_to_square(img) is img


def test_bound_longest_side_downscales_preserving_aspect() -> None:
    """An oversized image is shrunk to the limit with its aspect ratio intact."""
    img = Image.new("RGB", (8000, 500), (255, 255, 255))

    bounded = _bound_longest_side(img, limit=PIECE_SIZE)

    assert max(bounded.size) == PIECE_SIZE
    assert bounded.size == (128, 8)  # 500 * 128/8000 = 8


def test_bound_longest_side_leaves_small_image_unchanged() -> None:
    """An image already within the limit passes through untouched.

    Upscaling small crops stays with the inference resize, so this must not
    enlarge anything.
    """
    img = Image.new("RGB", (40, 20), (9, 9, 9))

    assert _bound_longest_side(img, limit=PIECE_SIZE) is img


class _FakeModel:
    """Stand-in model returning fixed predictions for pipeline tests."""

    def __call__(self, piece_tensor: Tensor, puzzle_tensor: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return a fixed (position, rotation_logits, attention_map) triple."""
        return (
            torch.tensor([[0.25, 0.75]]),
            torch.tensor([[5.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.9]]),
        )


@contextmanager
def capture_model_input(processor: ImageProcessor) -> Iterator[list[Image.Image]]:
    """Run process_piece against a fake model, capturing the image it is fed.

    Yields a list that receives the image handed to ``piece_transform`` — the
    prepared model input — for any process_piece call made inside the block.

    Args:
        processor: The processor whose model input should be captured.

    Yields:
        The list of captured model inputs.
    """
    captured: list[Image.Image] = []
    original_transform = processor.piece_transform

    def spy_transform(img: Image.Image) -> Tensor:
        captured.append(img)
        return cast(Tensor, original_transform(img))

    with (
        patch.object(processor, "model", _FakeModel()),
        patch.object(processor, "piece_transform", spy_transform),
        patch.object(processor, "_load_puzzle_tensor", return_value=torch.zeros(3, PUZZLE_SIZE, PUZZLE_SIZE)),
    ):
        yield captured


def test_process_piece_model_input_is_black_composited_square() -> None:
    """With background removal, the model sees the cutout on black, padded square.

    This mirrors the exp20 training pieces and the exp25 north-star eval prep
    (black composite + pad-to-square), not the previous white composite that
    squashed non-square crops.
    """
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    # Fake rembg output: a wide opaque red strip on a transparent canvas, so
    # the alpha crop is wider than tall and square padding must add black rows.
    rgba = Image.new("RGBA", (100, 60), (0, 0, 0, 0))
    rgba.paste(Image.new("RGBA", (80, 20), (255, 0, 0, 255)), (10, 20))

    class FakeRemover:
        """Background remover stub returning the prebuilt RGBA cutout."""

        def remove_background(self, contents: bytes) -> Image.Image:
            """Return the fake rembg output regardless of input."""
            return rgba

    with (
        patch.object(ip_module, "get_background_remover", return_value=FakeRemover()),
        patch.object(ip_module.settings, "ENABLE_BACKGROUND_REMOVAL", True),
        capture_model_input(processor) as captured,
    ):
        result = asyncio.run(processor.process_piece(make_upload(make_jpeg()), "some-puzzle-id"))

    assert len(captured) == 1
    model_input = captured[0]
    assert model_input.mode == "RGB"
    assert model_input.width == model_input.height
    # Padding rows and in-crop transparent regions are black; the piece keeps its color
    assert model_input.getpixel((model_input.width // 2, 0)) == (0, 0, 0)
    center = model_input.width // 2
    assert model_input.getpixel((center, center)) == (255, 0, 0)

    # The fake model's predictions flow through unchanged
    assert result.position.x == pytest.approx(0.25)
    assert result.position.y == pytest.approx(0.75)
    assert result.rotation == 0

    # The cleaned image for the frontend is still an RGBA PNG cutout
    assert result.cleaned_image is not None
    assert result.cleaned_image.startswith("data:image/png;base64,")
    cleaned = Image.open(io.BytesIO(base64.b64decode(result.cleaned_image.split(",", 1)[1])))
    assert cleaned.mode == "RGBA"


def test_process_piece_without_background_removal_pads_to_square() -> None:
    """Without background removal, a non-square upload is padded, not squashed."""
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    buffer = io.BytesIO()
    Image.new("RGB", (90, 30), (0, 255, 0)).save(buffer, format="JPEG")

    with capture_model_input(processor) as captured:
        asyncio.run(processor.process_piece(make_upload(buffer.getvalue()), "some-puzzle-id", remove_background=False))

    assert len(captured) == 1
    model_input = captured[0]
    assert model_input.size == (90, 90)
    assert model_input.getpixel((45, 0)) == (0, 0, 0)  # black padding row


def test_process_piece_bounds_pad_canvas_for_elongated_upload() -> None:
    """An elongated upload is downscaled before padding, bounding the canvas.

    MAX_UPLOAD_SIZE caps bytes, not pixels, so a small file can decode to
    extreme dimensions; padding those to square unbounded would allocate a
    canvas of max(width, height) per side.
    """
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    buffer = io.BytesIO()
    Image.new("RGB", (4000, 200), (0, 0, 255)).save(buffer, format="JPEG")

    with capture_model_input(processor) as captured:
        asyncio.run(processor.process_piece(make_upload(buffer.getvalue()), "some-puzzle-id", remove_background=False))

    assert len(captured) == 1
    # Without the bound this canvas would be 4000x4000 (~48MB) instead of 128x128
    assert captured[0].size == (PIECE_SIZE, PIECE_SIZE)


def test_load_puzzle_tensor_rejects_non_uuid_puzzle_id() -> None:
    """_load_puzzle_tensor rejects path-like puzzle IDs before building a file path."""
    with patch.object(ip_module, "CHECKPOINT_PATH", "/nonexistent/checkpoint.pt"):
        processor = ImageProcessor()

    with pytest.raises(FileNotFoundError):
        processor._load_puzzle_tensor("../etc/passwd")
