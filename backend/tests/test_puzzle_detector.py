"""Tests for the puzzle frame detector service."""

from typing import List, Tuple

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.puzzle_detector import FULL_IMAGE_CORNERS, Point, PuzzleFrameDetector, get_puzzle_detector

# Canvas and rectangle used by the synthetic test image
CANVAS_SIZE = (800, 600)
RECT_BOUNDS = (100, 80, 700, 520)  # left, top, right, bottom in pixels

CORNER_TOLERANCE = 0.03


@pytest.fixture
def detector() -> PuzzleFrameDetector:
    """Provide a detector instance."""
    return PuzzleFrameDetector()


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


class TestEdgeFallback:
    """Tests for the edge-based rectangle fallback (no uniform background)."""

    @staticmethod
    def make_edge_to_edge_puzzle(with_subject: bool) -> Tuple[Image.Image, "np.ndarray"]:
        """Render a tilted, bordered puzzle on a non-uniform (gradient) surround.

        The surround's chroma varies across the frame, so the border ring is
        not a uniform background: the chroma fast path bails and the edge
        fallback runs. This reproduces the real handheld-overview failure mode
        from issue #110, where the chroma path bails on every real photo.

        Args:
            with_subject: When true, paint a large blob of a single colour in
                the middle of the artwork — a stand-in for the salient subject
                (Dumbo, Simba) that a salient-object model wrongly crops to.

        Returns:
            The photo and the destination quad (4x2, TL/TR/BR/BL) of the
            puzzle's outline in pixel coordinates.
        """
        width, height = 900, 700
        # Smooth left-to-right colour gradient: chroma varies across the border
        # ring (so the chroma path bails) but stays low-frequency (few stray edges).
        ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        photo = np.zeros((height, width, 3), dtype=np.uint8)
        photo[:, :, 0] = (200 - 150 * ramp).astype(np.uint8)  # red fades out
        photo[:, :, 1] = 120
        photo[:, :, 2] = (50 + 150 * ramp).astype(np.uint8)  # blue fades in

        # Busy artwork with a dark border, warped to a tilted quad that nearly fills the frame
        art = np.full((512, 512, 3), (235, 225, 205), dtype=np.uint8)
        art[:, :, 0] = np.linspace(60, 230, 512, dtype=np.uint8)[None, :]
        art[::40] = (60, 90, 160)
        art[:, ::40] = (170, 70, 60)
        cv2.rectangle(art, (0, 0), (511, 511), (25, 25, 25), 14)  # the puzzle's own border
        if with_subject:
            cv2.circle(art, (256, 256), 150, (240, 40, 40), -1)  # salient "subject"

        dst_quad = np.array([[70, 90], [840, 55], [860, 640], [55, 660]], dtype=np.float32)
        src_quad = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)
        warped = cv2.warpPerspective(art, matrix, (width, height))
        mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), matrix, (width, height))
        photo = np.where(mask[..., None] > 0, warped, photo)
        return Image.fromarray(photo), dst_quad

    def test_chroma_path_bails_on_gradient_surround(self, detector: PuzzleFrameDetector) -> None:
        """The fixture's non-uniform surround makes the chroma fast path bail.

        This is the precondition for the tests below: they only exercise the
        edge fallback if the chroma path declines first. Guards against a future
        threshold tweak silently routing these through the chroma path.
        """
        import cv2 as _cv2

        photo, _ = self.make_edge_to_edge_puzzle(with_subject=False)
        work = _cv2.GaussianBlur(np.asarray(photo), (7, 7), 0).astype(np.float32)

        assert detector._foreground_mask(work) is None

    def test_edge_fallback_traces_puzzle_rectangle(self, detector: PuzzleFrameDetector) -> None:
        """A bordered puzzle on a non-uniform surround is detected by its rectangle."""
        from unittest.mock import patch

        photo, dst_quad = self.make_edge_to_edge_puzzle(with_subject=False)
        width, height = photo.size

        with patch.object(detector, "_edge_quad", wraps=detector._edge_quad) as spy:
            corners, confidence = detector.detect_corners(photo)
        spy.assert_called_once()  # detection went through the edge path, not chroma

        expected = [(float(x) / width, float(y) / height) for x, y in dst_quad.tolist()]
        assert confidence > 0.4
        assert_corners_close(corners, expected, tolerance=0.05)

    def test_traces_rectangle_not_the_salient_subject(self, detector: PuzzleFrameDetector) -> None:
        """The puzzle outline wins over a large salient blob inside the artwork.

        Regression for issue #110: a salient-object model cropped overviews to
        the artwork's subject (an elephant, a lion). The edge detector must
        trace the puzzle's rectangular border instead, so the returned quad
        covers the whole puzzle rather than the central blob.
        """
        photo, dst_quad = self.make_edge_to_edge_puzzle(with_subject=True)
        width, height = photo.size

        corners, confidence = detector.detect_corners(photo)

        expected = [(float(x) / width, float(y) / height) for x, y in dst_quad.tolist()]
        assert confidence > 0.4
        assert_corners_close(corners, expected, tolerance=0.05)
        # The detected region covers the whole puzzle, not the ~0.09-area central blob
        area = cv2.contourArea(np.array([(x * width, y * height) for x, y in corners], dtype=np.float32))
        assert area / (width * height) > 0.5

    def test_ragged_contour_recovers_quad_via_convex_hull(self, detector: PuzzleFrameDetector) -> None:
        """A rectangle outline with a deep notch still reduces to its quad.

        Regression for the real photo that motivated it: a box lying on a
        slightly larger box welds a step into the edge contour, so no
        approximation tolerance yields exactly four convex points from the raw
        outline — but the convex hull recovers the outer rectangle.
        """
        notched = np.array(
            [(100, 100), (500, 100), (500, 250), (380, 260), (500, 270), (500, 500), (100, 500)],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        quad = detector._quad_from_contour(notched)

        assert quad is not None
        assert sorted(map(tuple, quad.tolist())) == [
            (100.0, 100.0),
            (100.0, 500.0),
            (500.0, 100.0),
            (500.0, 500.0),
        ]


def test_get_puzzle_detector_is_singleton() -> None:
    """get_puzzle_detector returns the same instance on repeated calls."""
    assert get_puzzle_detector() is get_puzzle_detector()
