"""Tests for the piece detector service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.services.piece_detector import PieceDetector, crop_to_alpha_region, get_piece_detector, largest_alpha_component

# Opaque region drawn into test RGBA images: left, top, right, bottom
REGION = (60, 40, 260, 200)


def make_rgba_with_region(size: tuple[int, int] = (320, 240)) -> Image.Image:
    """Create a transparent RGBA image with one opaque rectangle at REGION."""
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    left, top, right, bottom = REGION
    rgba[top:bottom, left:right] = (200, 120, 80, 255)
    return Image.fromarray(rgba, mode="RGBA")


class TestCropToAlphaRegion:
    """Tests for crop_to_alpha_region."""

    def test_crops_to_opaque_region_with_margin(self) -> None:
        """The crop covers the opaque rectangle plus the requested margin."""
        left, top, right, bottom = REGION

        cropped = crop_to_alpha_region(make_rgba_with_region(), margin=0.1)

        expected_w = round((right - left) * 1.2)
        expected_h = round((bottom - top) * 1.2)
        assert abs(cropped.width - expected_w) <= 2
        assert abs(cropped.height - expected_h) <= 2

    def test_fully_transparent_image_is_unchanged(self) -> None:
        """An image with no opaque pixels is returned as-is."""
        rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))

        assert crop_to_alpha_region(rgba) is rgba

    def test_non_rgba_image_is_unchanged(self) -> None:
        """A non-RGBA image is returned as-is."""
        rgb = Image.new("RGB", (100, 100), (10, 20, 30))

        assert crop_to_alpha_region(rgb) is rgb

    def test_crop_is_clamped_to_image_bounds(self) -> None:
        """A region touching the edge does not produce an out-of-bounds crop."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[0:100, 0:100] = (10, 10, 10, 255)

        cropped = crop_to_alpha_region(Image.fromarray(rgba, mode="RGBA"), margin=0.2)

        assert cropped.size == (100, 100)


def test_largest_alpha_component_picks_biggest_blob() -> None:
    """With multiple opaque blobs, the largest one is selected."""
    import cv2

    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[10:30, 10:30, 3] = 255  # small blob
    rgba[80:180, 60:190, 3] = 255  # large blob

    contour = largest_alpha_component(Image.fromarray(rgba, mode="RGBA"))

    assert contour is not None
    x, y, w, h = cv2.boundingRect(contour)
    assert (x, y) == (60, 80)
    assert (w, h) == (130, 100)


class TestPieceDetector:
    """Tests for PieceDetector.detect_region with a mocked background remover."""

    @pytest.fixture
    def frame(self) -> Image.Image:
        """A camera frame; content is irrelevant since rembg is mocked."""
        return Image.new("RGB", (640, 480), (50, 60, 70))

    def test_detects_region_from_rembg_alpha(self, frame: Image.Image) -> None:
        """The polygon and bbox reflect the mocked rembg alpha mask."""
        remover = MagicMock()
        remover.remove_background.return_value = make_rgba_with_region()

        with patch("app.services.piece_detector.get_background_remover", return_value=remover):
            result = PieceDetector().detect_region(frame)

        assert result is not None
        polygon, bbox = result
        left, top, right, bottom = REGION
        assert bbox[0] == pytest.approx(left / 320, abs=0.02)
        assert bbox[1] == pytest.approx(top / 240, abs=0.02)
        assert bbox[2] == pytest.approx((right - left) / 320, abs=0.02)
        assert bbox[3] == pytest.approx((bottom - top) / 240, abs=0.02)
        assert len(polygon) >= 4
        for x, y in polygon:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_no_alpha_returns_none(self, frame: Image.Image) -> None:
        """A fully transparent rembg result yields no detection."""
        remover = MagicMock()
        remover.remove_background.return_value = Image.new("RGBA", (320, 240), (0, 0, 0, 0))

        with patch("app.services.piece_detector.get_background_remover", return_value=remover):
            assert PieceDetector().detect_region(frame) is None

    def test_tiny_region_returns_none(self, frame: Image.Image) -> None:
        """A region below the minimum area ratio is rejected."""
        rgba = np.zeros((240, 320, 4), dtype=np.uint8)
        rgba[100:104, 100:104, 3] = 255
        remover = MagicMock()
        remover.remove_background.return_value = Image.fromarray(rgba, mode="RGBA")

        with patch("app.services.piece_detector.get_background_remover", return_value=remover):
            assert PieceDetector().detect_region(frame) is None

    def test_rembg_failure_returns_none(self, frame: Image.Image) -> None:
        """A rembg exception degrades to no detection instead of raising."""
        remover = MagicMock()
        remover.remove_background.side_effect = RuntimeError("boom")

        with patch("app.services.piece_detector.get_background_remover", return_value=remover):
            assert PieceDetector().detect_region(frame) is None


def test_get_piece_detector_is_singleton() -> None:
    """get_piece_detector returns the same instance on repeated calls."""
    assert get_piece_detector() is get_piece_detector()
