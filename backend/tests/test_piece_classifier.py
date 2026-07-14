"""Tests for the piece classifier service."""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from PIL import Image

from app.models.piece_classifier_model import PieceClassifier
from app.services.piece_classifier import PieceClassifierService, get_piece_classifier, prepare_classifier_input


def make_rgba_with_region(
    size: tuple[int, int] = (320, 240),
    region: tuple[int, int, int, int] = (60, 40, 160, 120),
    color: tuple[int, int, int] = (80, 120, 200),
) -> Image.Image:
    """Create a transparent RGBA image with one opaque rectangle."""
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    left, top, right, bottom = region
    rgba[top:bottom, left:right] = (*color, 255)
    return Image.fromarray(rgba, mode="RGBA")


def make_service(checkpoint_path: Optional[Path], monkeypatch: pytest.MonkeyPatch) -> PieceClassifierService:
    """Build a service instance loading from the given checkpoint path."""
    from app.services import piece_classifier as module

    path = str(checkpoint_path) if checkpoint_path is not None else "/nonexistent/checkpoint_best.pt"
    monkeypatch.setattr(module, "CHECKPOINT_PATH", path)
    return PieceClassifierService()


class TestPrepareClassifierInput:
    """Tests for the shared input-preparation protocol."""

    def test_crop_is_square_with_black_background(self) -> None:
        """The crop is square and transparent areas are composited to black."""
        crop = prepare_classifier_input(make_rgba_with_region())

        assert crop is not None
        assert crop.mode == "RGB"
        assert crop.width == crop.height
        pixels = np.asarray(crop)
        # Corners are padding (the region is wider than tall) and must be black
        assert pixels[0, 0].tolist() == [0, 0, 0]
        # The subject color survives in the middle
        assert pixels[crop.height // 2, crop.width // 2].tolist() == [80, 120, 200]

    def test_fully_transparent_image_returns_none(self) -> None:
        """No opaque region means no crop."""
        assert prepare_classifier_input(Image.new("RGBA", (100, 100), (0, 0, 0, 0))) is None

    def test_non_rgba_image_returns_none(self) -> None:
        """The protocol only accepts rembg RGBA output."""
        assert prepare_classifier_input(Image.new("RGB", (100, 100), (10, 20, 30))) is None

    def test_crop_targets_largest_component(self) -> None:
        """With several blobs, the crop covers the largest one."""
        rgba = np.zeros((240, 320, 4), dtype=np.uint8)
        rgba[10:30, 10:30] = (200, 0, 0, 255)  # small blob
        rgba[100:200, 100:220] = (0, 200, 0, 255)  # large blob (120x100)

        crop = prepare_classifier_input(Image.fromarray(rgba, mode="RGBA"), margin=0.0)

        assert crop is not None
        assert crop.width == pytest.approx(120, abs=2)


class TestPieceClassifierService:
    """Tests for checkpoint loading and scoring."""

    def test_missing_checkpoint_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No checkpoint (e.g. CI): service unavailable, score returns None."""
        service = make_service(None, monkeypatch)

        assert service.available is False
        assert service.score(make_rgba_with_region()) is None

    def test_corrupt_checkpoint_degrades_gracefully(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unloadable checkpoint leaves the service unavailable."""
        bad = tmp_path / "checkpoint_best.pt"
        bad.write_bytes(b"not a checkpoint")

        service = make_service(bad, monkeypatch)

        assert service.available is False

    def test_loads_checkpoint_and_scores(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid checkpoint loads and produces a probability in [0, 1]."""
        checkpoint = tmp_path / "checkpoint_best.pt"
        torch.save({"model_state_dict": PieceClassifier(pretrained=False).state_dict()}, checkpoint)

        service = make_service(checkpoint, monkeypatch)

        assert service.available is True
        score = service.score(make_rgba_with_region())
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_score_returns_none_without_usable_crop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fully transparent rembg result cannot be scored."""
        checkpoint = tmp_path / "checkpoint_best.pt"
        torch.save({"model_state_dict": PieceClassifier(pretrained=False).state_dict()}, checkpoint)
        service = make_service(checkpoint, monkeypatch)

        assert service.score(Image.new("RGBA", (100, 100), (0, 0, 0, 0))) is None


def test_get_piece_classifier_is_singleton() -> None:
    """get_piece_classifier returns the same instance on repeated calls."""
    assert get_piece_classifier() is get_piece_classifier()
