"""Tests for the puzzle frame detector service."""

from typing import List, Tuple

import pytest
from PIL import Image, ImageDraw

from app.services.puzzle_detector import FULL_IMAGE_CORNERS, Point, PuzzleFrameDetector, get_puzzle_detector

# Canvas and rectangle used by the synthetic test image
CANVAS_SIZE = (800, 600)
RECT_BOUNDS = (100, 80, 700, 520)  # left, top, right, bottom in pixels

CORNER_TOLERANCE = 0.03


@pytest.fixture
def detector() -> PuzzleFrameDetector:
    """Provide a detector instance without the (slow) rembg salient-object fallback."""
    return PuzzleFrameDetector(salient_fallback=False)


def make_rectangle_image() -> Image.Image:
    """Create a dark canvas with a bright rectangle at RECT_BOUNDS."""
    image = Image.new("RGB", CANVAS_SIZE, (20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle(RECT_BOUNDS, fill=(220, 210, 190))
    return image


def expected_corners() -> List[Point]:
    """Expected normalized corners (TL, TR, BR, BL) of the synthetic rectangle."""
    width, height = CANVAS_SIZE
    left, top, right, bottom = RECT_BOUNDS
    return [
        (left / width, top / height),
        (right / width, top / height),
        (right / width, bottom / height),
        (left / width, bottom / height),
    ]


def assert_corners_close(actual: List[Point], expected: List[Point], tolerance: float = CORNER_TOLERANCE) -> None:
    """Assert each actual corner is within tolerance of the expected corner."""
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert abs(ax - ex) <= tolerance, f"x: {ax} vs {ex}"
        assert abs(ay - ey) <= tolerance, f"y: {ay} vs {ey}"


class TestDetectCorners:
    """Tests for PuzzleFrameDetector.detect_corners."""

    def test_detects_bright_rectangle(self, detector: PuzzleFrameDetector) -> None:
        """A clear rectangle on a dark background is detected with ordered corners."""
        corners, confidence = detector.detect_corners(make_rectangle_image())

        assert len(corners) == 4
        assert confidence > 0.5
        assert_corners_close(corners, expected_corners())

    def test_corners_are_normalized(self, detector: PuzzleFrameDetector) -> None:
        """All detected corners are within [0, 1]."""
        corners, _ = detector.detect_corners(make_rectangle_image())

        for x, y in corners:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_blank_image_falls_back_to_full_frame(self, detector: PuzzleFrameDetector) -> None:
        """A uniform image with no contours returns full-image corners and zero confidence."""
        blank = Image.new("RGB", (640, 480), (128, 128, 128))

        corners, confidence = detector.detect_corners(blank)

        assert corners == FULL_IMAGE_CORNERS
        assert confidence == 0.0

    def test_small_rectangle_is_rejected(self, detector: PuzzleFrameDetector) -> None:
        """A rectangle covering far less than MIN_AREA_RATIO is not accepted."""
        image = Image.new("RGB", CANVAS_SIZE, (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 90, 70), fill=(220, 210, 190))

        corners, confidence = detector.detect_corners(image)

        assert corners == FULL_IMAGE_CORNERS
        assert confidence == 0.0

    def test_tilted_photo_on_noisy_background_with_shadow(self, detector: PuzzleFrameDetector) -> None:
        """A tilted picture on a noisy table with a soft shadow is detected, shadow excluded.

        Regression test for the realistic-photo case: edge-based detection failed here
        because the picture's internal edges fragment its outline, and brightness-based
        segmentation included the shadow.
        """
        import cv2
        import numpy as np

        rng = np.random.default_rng(7)
        width, height = 1600, 1200
        background = np.full((height, width, 3), (92, 64, 48), dtype=np.uint8)
        noise = rng.normal(0, 8, (height, width, 1)).astype(np.int16)
        background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # A photo-like subject with internal structure (gradient + stripes), tilted
        subject = np.zeros((512, 512, 3), dtype=np.uint8)
        subject[:, :, 0] = np.linspace(30, 220, 512, dtype=np.uint8)[None, :]
        subject[:, :, 1] = np.linspace(200, 40, 512, dtype=np.uint8)[:, None]
        subject[:, :, 2] = 160
        subject[::32] = (240, 240, 240)

        dst_quad = np.array([[330, 240], [1290, 210], [1330, 950], [290, 990]], dtype=np.float32)
        src_quad = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
        warped = cv2.warpPerspective(subject, matrix, (width, height))
        mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), matrix, (width, height))

        # Soft shadow around the subject
        shadow = cv2.dilate(mask, np.ones((25, 25), np.uint8))
        shadow = cv2.GaussianBlur(shadow, (51, 51), 0)
        background = (background * (1 - 0.35 * (shadow[..., None] / 255.0))).astype(np.uint8)
        photo = Image.fromarray(np.where(mask[..., None] > 0, warped, background))

        corners, confidence = detector.detect_corners(photo)

        expected = [(float(x) / width, float(y) / height) for x, y in dst_quad.tolist()]
        assert confidence > 0.4
        assert_corners_close(corners, expected)

    def test_large_image_is_downscaled(self, detector: PuzzleFrameDetector) -> None:
        """Detection works on images larger than the working dimension."""
        image = Image.new("RGB", (2400, 1800), (20, 20, 20))
        draw = ImageDraw.Draw(image)
        draw.rectangle((300, 240, 2100, 1560), fill=(220, 210, 190))

        corners, confidence = detector.detect_corners(image)

        assert confidence > 0.5
        assert_corners_close(
            corners,
            [
                (300 / 2400, 240 / 1800),
                (2100 / 2400, 240 / 1800),
                (2100 / 2400, 1560 / 1800),
                (300 / 2400, 1560 / 1800),
            ],
        )


class TestWarp:
    """Tests for PuzzleFrameDetector.warp."""

    def test_warp_crops_to_rectangle(self, detector: PuzzleFrameDetector) -> None:
        """Warping with the rectangle's corners yields an image matching its size."""
        image = make_rectangle_image()
        left, top, right, bottom = RECT_BOUNDS
        expected_size: Tuple[int, int] = (right - left, bottom - top)

        warped = detector.warp(image, expected_corners())

        assert abs(warped.width - expected_size[0]) <= 2
        assert abs(warped.height - expected_size[1]) <= 2

    def test_warp_content_is_bright(self, detector: PuzzleFrameDetector) -> None:
        """The warped crop contains the bright rectangle, not the dark background."""
        warped = detector.warp(make_rectangle_image(), expected_corners())

        center = warped.getpixel((warped.width // 2, warped.height // 2))
        assert isinstance(center, tuple)
        assert center[0] > 150

    def test_warp_accepts_unordered_corners(self, detector: PuzzleFrameDetector) -> None:
        """Corners in a different order produce the same crop size."""
        tl, tr, br, bl = expected_corners()

        warped = detector.warp(make_rectangle_image(), [br, tl, bl, tr])

        left, top, right, bottom = RECT_BOUNDS
        assert abs(warped.width - (right - left)) <= 2
        assert abs(warped.height - (bottom - top)) <= 2

    def test_full_image_corners_return_whole_image(self, detector: PuzzleFrameDetector) -> None:
        """Warping with the fallback corners returns (approximately) the input image."""
        image = make_rectangle_image()

        warped = detector.warp(image, list(FULL_IMAGE_CORNERS))

        assert abs(warped.width - image.width) <= 2
        assert abs(warped.height - image.height) <= 2


class TestSalientFallback:
    """Tests for the rembg salient-object fallback path."""

    @staticmethod
    def make_cluttered_photo() -> Image.Image:
        """Create a photo whose border is too varied for background segmentation."""
        import numpy as np

        rng = np.random.default_rng(3)
        clutter = rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)
        image = Image.fromarray(clutter)
        draw = ImageDraw.Draw(image)
        draw.rectangle(RECT_BOUNDS, fill=(220, 210, 190))
        return image

    def test_salient_fallback_detects_subject(self) -> None:
        """When the border is cluttered, the rembg mask drives detection."""
        from unittest.mock import MagicMock, patch

        import numpy as np

        # Fake rembg output: alpha covering exactly the drawn rectangle
        alpha = np.zeros((600, 800), dtype=np.uint8)
        left, top, right, bottom = RECT_BOUNDS
        alpha[top:bottom, left:right] = 255
        rgba = np.dstack([np.zeros((600, 800, 3), dtype=np.uint8), alpha])
        remover = MagicMock()
        remover.remove_background.return_value = Image.fromarray(rgba, mode="RGBA")

        detector = PuzzleFrameDetector(salient_fallback=True)
        with patch("app.services.puzzle_detector.get_background_remover", return_value=remover):
            corners, confidence = detector.detect_corners(self.make_cluttered_photo())

        remover.remove_background.assert_called_once()
        assert confidence > 0.4
        assert_corners_close(corners, expected_corners())

    def test_salient_fallback_failure_returns_full_frame(self) -> None:
        """If rembg raises, detection degrades gracefully to the full-frame fallback."""
        from unittest.mock import MagicMock, patch

        remover = MagicMock()
        remover.remove_background.side_effect = RuntimeError("model unavailable")

        detector = PuzzleFrameDetector(salient_fallback=True)
        with patch("app.services.puzzle_detector.get_background_remover", return_value=remover):
            corners, confidence = detector.detect_corners(self.make_cluttered_photo())

        assert corners == FULL_IMAGE_CORNERS
        assert confidence == 0.0

    def test_no_fallback_when_disabled(self) -> None:
        """With salient_fallback=False a cluttered border goes straight to full frame."""
        from unittest.mock import patch

        detector = PuzzleFrameDetector(salient_fallback=False)
        with patch("app.services.puzzle_detector.get_background_remover") as mock_get:
            corners, confidence = detector.detect_corners(self.make_cluttered_photo())

        mock_get.assert_not_called()
        assert corners == FULL_IMAGE_CORNERS
        assert confidence == 0.0


def test_get_puzzle_detector_is_singleton() -> None:
    """get_puzzle_detector returns the same instance on repeated calls."""
    assert get_puzzle_detector() is get_puzzle_detector()
