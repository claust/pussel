"""Tests for the puzzle frame detector service."""

from typing import List, Tuple

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services import puzzle_detector
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


def winning_source(detector: PuzzleFrameDetector, photo: Image.Image) -> str:
    """The generator whose candidate won, for tests that pin which evidence decided.

    Every generator runs on every photo (see `detect_candidates`), so which one
    was *called* says nothing; which one won says what the image actually
    offered.
    """
    candidates = detector.detect_candidates(photo)
    return candidates[0].source if candidates else "none"


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
        photo, dst_quad = self.make_edge_to_edge_puzzle(with_subject=False)
        width, height = photo.size

        corners, confidence = detector.detect_corners(photo)

        # No uniform background to segment against, so the geometric generators
        # are the ones with anything to say here.
        assert winning_source(detector, photo) in {"edge", "lines"}
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


class TestCarpetBackground:
    """Regression tests for a small puzzle on speckled gray carpet.

    Both fixtures reproduce real handheld photos of a small puzzle on gray
    carpet (2026-07-24 GlareFreeDumps captures, not committed). The carpet's
    fiber speckle carries chromatic noise that survives the working blur, which
    corrupted the old chroma-distance segmentation into a confident wrong quad
    or a full-frame fallback.
    """

    @staticmethod
    def make_carpet(rng: "np.random.Generator", width: int, height: int) -> "np.ndarray":
        """Render a speckled gray carpet with chromatic noise in the speckle.

        The speckle is generated at a coarse "fiber" scale and upscaled so it
        survives the detector's working blur, like real carpet fibers do. The
        per-channel tint noise makes dark fibers chromatically unreliable —
        the property that broke the raw chroma-distance formulation.

        The fiber brightness range is set so the Canny edge map this produces
        lands at the density the real captures measure (0.30-0.37 on the
        2026-07-24 carpet dumps). It matters: edge support is a statistic
        against chance, so how busy the edge map is decides how much the term
        can say, and the wider uniform(40, 200) this fixture used originally
        saturated it at 0.79 — denser than any real photo, and dense enough to
        exercise a code path (`_edge_map_is_informative`) the scene it models
        never reaches.

        Args:
            rng: Seeded random generator.
            width: Output width in pixels.
            height: Output height in pixels.

        Returns:
            uint8 RGB array of shape (height, width, 3).
        """
        fiber = 6
        coarse_w, coarse_h = width // fiber, height // fiber
        fibers = rng.uniform(55, 180, (coarse_h, coarse_w, 1)).astype(np.float32)
        tint = rng.normal(0, 10, (coarse_h, coarse_w, 3)).astype(np.float32)
        coarse = np.clip(fibers + tint, 0, 255)
        return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    @staticmethod
    def compose(carpet: "np.ndarray", art: "np.ndarray", dst_quad: "np.ndarray") -> Image.Image:
        """Warp the artwork onto the carpet at the given destination quad."""
        height, width = carpet.shape[:2]
        size = art.shape[0]
        src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst_quad.astype(np.float32))
        warped = cv2.warpPerspective(art, matrix, (width, height))
        mask = cv2.warpPerspective(np.full((size, size), 255, np.uint8), matrix, (width, height))
        return Image.fromarray(np.where(mask[..., None] > 0, warped, carpet))

    def test_small_colorful_puzzle_on_speckled_carpet(self, detector: PuzzleFrameDetector) -> None:
        """A colorful puzzle covering ~18% of the frame is found on noisy carpet.

        Regression for the real capture where the border ring's noisy chroma
        made the old segmentation return a confident wrong quad: the color
        residual must treat the speckle as one uniform background.
        """
        width, height = 1200, 1600
        carpet = self.make_carpet(np.random.default_rng(3), width, height)

        art = np.zeros((512, 512, 3), dtype=np.uint8)
        art[:, :, 0] = np.linspace(60, 230, 512, dtype=np.uint8)[None, :]
        art[:, :, 1] = np.linspace(200, 40, 512, dtype=np.uint8)[:, None]
        art[:, :, 2] = 160
        art[::32] = (240, 120, 40)

        dst_quad = np.array([[370, 430], [850, 445], [840, 1160], [360, 1150]], dtype=np.float32)
        photo = self.compose(carpet, art, dst_quad)

        corners, confidence = detector.detect_corners(photo)

        expected = [(float(x) / width, float(y) / height) for x, y in dst_quad.tolist()]
        # Modest confidence is the honest reading here, and the corners are what the
        # test is really about: the puzzle covers a fifth of the frame and the carpet
        # texture leaves the edge map busy, so the boundary evidence is genuinely thin
        # even though the fitted quad is within a couple of pixels.
        assert confidence > 0.1
        assert_corners_close(corners, expected)

    def test_washed_out_puzzle_recovered_by_grabcut(self, detector: PuzzleFrameDetector) -> None:
        """A washed-out puzzle whose colors barely differ from the carpet is recovered.

        Regression for the real capture under warm light where thresholding
        only finds scattered colorful fragments, none passing the area gate:
        the GrabCut stage must grow the fragments and fit the quad on their
        combined convex hull instead of falling through to the edge path
        (which traces a garbage quad on the carpet texture).
        """
        width, height = 1200, 1600
        carpet = self.make_carpet(np.random.default_rng(5), width, height)

        # Mostly achromatic artwork with a wash tint too weak for the threshold
        # mask, plus colorful fragments along parts of the frame (a broken ring)
        rng = np.random.default_rng(11)
        coarse = rng.uniform(60, 200, (85, 85)).astype(np.float32)
        base = cv2.resize(np.clip(coarse, 0, 255), (512, 512))
        art_f = np.zeros((512, 512, 3), dtype=np.float32)
        art_f[:, :, 0] = base + 18
        art_f[:, :, 1] = base
        art_f[:, :, 2] = base + 18
        art = np.clip(art_f, 0, 255).astype(np.uint8)
        art[:45, :, :] = (210, 90, 40)
        art[:, -45:, :] = (60, 60, 200)
        art[-45:, :150, :] = (200, 40, 160)
        art[100:220, :45, :] = (240, 150, 30)
        art[220:300, 200:300] = (40, 160, 60)

        dst_quad = np.array([[330, 380], [890, 400], [880, 1210], [320, 1190]], dtype=np.float32)
        photo = self.compose(carpet, art, dst_quad)

        cv2.setRNGSeed(7)  # pin GrabCut's k-means initialization
        corners, confidence = detector.detect_corners(photo)

        # The point of the fixture: the colour mask fragments here, so GrabCut's
        # recovery is the candidate that has to win.
        assert winning_source(detector, photo) == "grabcut"
        expected = [(float(x) / width, float(y) / height) for x, y in dst_quad.tolist()]
        assert confidence > 0.05
        assert_corners_close(corners, expected)

    def test_edge_support_separates_a_real_border_from_dense_texture(self, detector: PuzzleFrameDetector) -> None:
        """A drawn border far outscores the same line over texture alone.

        On carpet the dilated Canny map is dense enough (0.3 in the real
        captures, reproduced here) that any outline "hits" edges by chance, so
        support counts only the excess. Chance correction is inherently noisy
        at this density, so what has to hold is the separation — an arbitrary
        absolute ceiling on the noise reading would just pin the seed.
        """
        rng = np.random.default_rng(2)
        texture = ((rng.random((600, 800)) < 0.3) * 255).astype(np.uint8)
        start, end = np.array([100.0, 80.0]), np.array([700.0, 90.0])

        density = float(np.count_nonzero(texture)) / texture.size
        on_texture = puzzle_detector._side_edge_support(start, end, texture, density)

        with_border = texture.copy()
        cv2.line(with_border, (100, 80), (700, 90), 255, 3)
        border_density = float(np.count_nonzero(with_border)) / with_border.size
        on_border = puzzle_detector._side_edge_support(start, end, with_border, border_density)

        assert on_border > 0.9
        assert on_border > 2 * on_texture

    def test_saturated_edge_map_abstains_rather_than_vetoing(self, detector: PuzzleFrameDetector) -> None:
        """Where every window contains an edge, the term abstains instead of scoring 0.

        The statistic has no discriminating power at that density — every
        candidate would score 0 and detection would return nothing at all,
        which is worse than leaving the ranking to the terms that still work.
        """
        rng = np.random.default_rng(2)
        edges = ((rng.random((600, 800)) < 0.6) * 255).astype(np.uint8)
        quad = np.array([[100, 80], [700, 90], [690, 520], [110, 510]], dtype=np.float32)

        assert not puzzle_detector._edge_map_is_informative(float(np.count_nonzero(edges)) / edges.size)
        _, components = detector._score_quad(quad, edges, 800, 600)
        assert components["edge_support"] == 1.0


class TestCandidateFusion:
    """Tests for the candidate generators competing on one score."""

    @staticmethod
    def box_with_hard_shadow() -> Tuple[Image.Image, "np.ndarray"]:
        """A bright box on textured carpet, with a hard-edged shadow beside it.

        Reproduces the real 2026-07-25 capture that motivated the fusion work:
        directional light throws a shadow whose boundary is as long and as
        straight as the box's own left edge, so a contour trace wraps both into
        one region and every edge-based score is happy with the result. Only
        the polarity of the step across that side tells them apart — into the
        shadow it goes darker, everywhere else it goes brighter.

        Returns:
            The photo and the box's true quad, shape (4, 2).
        """
        rng = np.random.default_rng(11)
        width, height = 1200, 1600
        carpet = np.clip(rng.normal(118, 14, (height, width, 1)) + rng.normal(0, 6, (height, width, 3)), 0, 255).astype(
            np.uint8
        )

        box_quad = np.array([[380, 430], [830, 440], [825, 1150], [375, 1140]], dtype=np.float32)
        shadow_quad = np.array([[210, 430], [380, 430], [375, 1140], [205, 1140]], dtype=np.float32)
        photo = carpet.copy()

        shadow_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(shadow_mask, [shadow_quad.astype(np.int32)], 255)
        photo = np.where(shadow_mask[..., None] > 0, (photo * 0.55).astype(np.uint8), photo)

        art = np.full((512, 512, 3), 225, np.uint8)
        art[40:472, 40:472] = np.linspace(40, 190, 432, dtype=np.uint8)[None, :, None]
        source = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(source, box_quad)
        warped = cv2.warpPerspective(art, matrix, (width, height))
        box_mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), matrix, (width, height))
        photo = np.where(box_mask[..., None] > 0, warped, photo)
        return Image.fromarray(photo), box_quad

    def test_straight_cast_shadow_is_not_swallowed_into_the_quad(self, detector: PuzzleFrameDetector) -> None:
        """The detected quad stops at the box, not at the shadow's outer edge."""
        photo, box_quad = self.box_with_hard_shadow()
        width, height = photo.size

        corners, confidence = detector.detect_corners(photo)

        expected = [(float(x) / width, float(y) / height) for x, y in box_quad.tolist()]
        assert_corners_close(corners, expected, tolerance=0.02)
        assert confidence > 0.3

    def test_the_shadow_quad_is_generated_but_scores_lower(self, detector: PuzzleFrameDetector) -> None:
        """The wrong quad is available and rejected — it isn't merely never proposed.

        The shadow-inclusive quad is bigger and sits on edges just as real, so
        every other term favours it. Scoring the two head to head pins *why* it
        loses — rather than checking it is absent from the results, which it is:
        it falls below `MIN_CANDIDATE_SCORE` and is dropped entirely.
        """
        photo, box_quad = self.box_with_hard_shadow()
        work = cv2.GaussianBlur(np.asarray(photo), (7, 7), 0).astype(np.float32)
        work_h, work_w = work.shape[:2]
        edges = detector._edge_map(work)
        # The same quad with its left side moved out to the shadow's outer boundary.
        shadow_quad = box_quad.copy()
        shadow_quad[0][0] = shadow_quad[3][0] = 210

        box_score, box_parts = detector._score_quad(box_quad, edges, work_w, work_h, work)
        shadow_score, shadow_parts = detector._score_quad(shadow_quad, edges, work_w, work_h, work)

        assert shadow_parts["coverage"] >= box_parts["coverage"]
        assert shadow_parts["edge_support"] > 0.3, "the shadow's boundary is a real edge, not a scoring artefact"
        assert shadow_parts["border_contrast"] < 0.5 < box_parts["border_contrast"]
        assert shadow_score < 0.5 * box_score

    def test_lines_recover_a_rectangle_whose_outline_is_broken(self, detector: PuzzleFrameDetector) -> None:
        """A border interrupted by gaps still yields a quad, via line intersection.

        Contour tracing needs an unbroken loop; four straight lines do not.
        """
        width, height = 1000, 800
        photo = np.full((height, width, 3), 30, np.uint8)
        cv2.rectangle(photo, (180, 150), (820, 650), (225, 220, 205), -1)
        # Punch gaps through the border so no single contour encloses the shape.
        for start, end in (((380, 140), (520, 160)), ((180, 380), (200, 470)), ((640, 640), (760, 660))):
            cv2.rectangle(photo, start, end, (30, 30, 30), -1)

        candidates = detector.detect_candidates(Image.fromarray(photo))

        assert any(candidate.source == "lines" for candidate in candidates)
        expected = [
            (180 / width, 150 / height),
            (820 / width, 150 / height),
            (820 / width, 650 / height),
            (180 / width, 650 / height),
        ]
        assert_corners_close(candidates[0].corners, expected, tolerance=0.02)


def test_get_puzzle_detector_is_singleton() -> None:
    """get_puzzle_detector returns the same instance on repeated calls."""
    assert get_puzzle_detector() is get_puzzle_detector()
