"""Service for detecting a puzzle picture's boundary in a photo and perspective-correcting it."""

import math
from typing import List, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image

Point = Tuple[float, float]

# Corners covering the whole image, used as fallback when no quadrilateral is found
FULL_IMAGE_CORNERS: List[Point] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

# Detection runs on a downscaled copy for speed; warping uses the full-resolution image
WORKING_MAX_DIM = 1000

# A candidate region must cover at least this fraction of the photo
MIN_AREA_RATIO = 0.15

# Area ratio at (or above) which the confidence heuristic saturates
FULL_CONFIDENCE_AREA_RATIO = 0.5

# Fraction of the shorter working dimension used as the border ring for background sampling
BORDER_FRACTION = 0.05

# If the border ring's 99th-percentile color residual (RGB units, see _foreground_mask)
# exceeds this, there is no uniform background around the subject (e.g. an edge-to-edge
# image) and detection bails out. Calibrated so a textured carpet with mild illumination
# tint (~20) passes while a colored surround (~100+) bails.
MAX_BORDER_COLOR_SPREAD = 30.0

# Minimum color residual (RGB units) for a pixel to count as foreground
COLOR_RESIDUAL_FLOOR = 12.0

# GrabCut refinement runs on a further-downscaled copy: cost grows quadratically and
# region shape, not pixel accuracy, is what the quad fit needs
GRABCUT_MAX_DIM = 750

# Seed components smaller than this fraction of the frame are dropped before GrabCut
# (stray specks — e.g. a photographer's shadow patch — would otherwise grow into the region)
GRABCUT_SEED_COMPONENT_RATIO = 0.005

# GrabCut is skipped when the surviving seeds cover less than this fraction of the frame:
# with almost no foreground evidence it would hallucinate a region from noise
GRABCUT_MIN_SEED_RATIO = 0.02

GRABCUT_ITERATIONS = 3

# Auto-Canny spread: thresholds are set to (1 ± sigma) × the image's median intensity,
# so the edge detector adapts to each photo's exposure rather than using fixed levels
EDGE_CANNY_SIGMA = 0.33

# Kernel size for dilating the edge map, closing small gaps in the puzzle's border
# so its outline forms a single traceable contour
EDGE_DILATE_KERNEL = 5

# Polygon-approximation tolerances (as a fraction of contour perimeter) tried in order
# when reducing an edge contour to a quadrilateral
EDGE_APPROX_EPSILONS = (0.02, 0.04, 0.06)

# Line thickness (working pixels) used when rasterizing a candidate quad's outline to
# measure how much of it coincides with real detected edges (the edge-support term)
EDGE_SUPPORT_THICKNESS = 2 * EDGE_DILATE_KERNEL + 1


def _order_corners(points: "np.ndarray") -> "np.ndarray":
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    Uses the standard sum/diff heuristic: top-left has the smallest x+y,
    bottom-right the largest x+y, top-right the smallest y-x, bottom-left
    the largest y-x.

    Args:
        points: Array of shape (4, 2) with (x, y) coordinates.

    Returns:
        Array of shape (4, 2) ordered TL, TR, BR, BL.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()  # y - x
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _chroma(pixels: "np.ndarray") -> "np.ndarray":
    """Compute brightness-invariant chromaticity (each channel divided by the channel sum).

    Shadows keep the background's chroma while differing in brightness, so
    chroma distance separates the subject from shadows on the background.

    Args:
        pixels: Float array of RGB values, last axis of size 3.

    Returns:
        Array of the same shape with channels normalized to sum to 1.
    """
    return pixels / (pixels.sum(axis=-1, keepdims=True) + 1e-6)


class PuzzleFrameDetector:
    """Detects the puzzle picture in a photo and warps it to a flat crop.

    Detection is layered. The fast path segments the subject from a roughly
    uniform background (sampled from the photo's border) using a color residual
    — which ignores shadows — plus a brighter-than-background luminance rule,
    then fits a quadrilateral to the convex hull of the largest region. When
    that mask has real foreground evidence but no single region large enough
    (a washed-out puzzle segments into fragments), a GrabCut pass seeded from
    those fragments recovers the full region. When the border is not a uniform
    background (e.g. a handheld photo where the puzzle nearly fills the frame),
    it falls back to direct edge-based detection of the puzzle's rectangular
    boundary.

    The puzzle picture is a strong quadrilateral with straight edges, so the
    fallback looks for that rectangle explicitly rather than asking a
    salient-object model — a salient-object model segments whatever the
    artwork depicts (an elephant, a lion) instead of the puzzle's outline.
    """

    def detect_corners(self, image: Image.Image) -> Tuple[List[Point], float]:
        """Detect the puzzle picture's quadrilateral in the image.

        The image should already be RGB and EXIF-corrected. Detection runs on a
        downscaled copy; returned corners are normalized to [0, 1] so they apply
        to the original image as well.

        Args:
            image: The photo to analyze.

        Returns:
            A tuple of (corners, confidence). Corners are ordered
            [top-left, top-right, bottom-right, bottom-left], normalized 0-1.
            Falls back to the full-image corners with confidence 0.0 when no
            suitable region is found.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        scale = min(1.0, WORKING_MAX_DIM / max(width, height))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
        work = cv2.GaussianBlur(rgb, (7, 7), 0).astype(np.float32)
        work_h, work_w = work.shape[:2]

        mask = self._foreground_mask(work)
        result = self._quad_from_mask(mask, work_w, work_h) if mask is not None else None

        if result is None and mask is not None:
            result = self._grabcut_quad(work, mask, work_w, work_h)

        if result is None:
            result = self._edge_quad(work, work_w, work_h)

        if result is None:
            return list(FULL_IMAGE_CORNERS), 0.0

        quad, confidence = result
        ordered = _order_corners(quad)
        corners: List[Point] = [
            (float(np.clip(x / work_w, 0.0, 1.0)), float(np.clip(y / work_h, 0.0, 1.0))) for x, y in ordered
        ]
        return corners, confidence

    def warp(self, image: Image.Image, corners: List[Point]) -> Image.Image:
        """Perspective-warp and crop the image using 4 normalized corners.

        Args:
            image: The full-resolution photo to warp.
            corners: Four (x, y) points normalized to [0, 1], in any consistent
                order; they are re-ordered TL, TR, BR, BL internally.

        Returns:
            A new image containing only the quadrilateral region, straightened
            to a rectangle sized from the corner-to-corner distances.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        pixels = np.array([(x * width, y * height) for x, y in corners], dtype=np.float32)
        src = _order_corners(pixels)
        tl, tr, br, bl = src

        dst_w = max(1, round(max(math.dist(tl, tr), math.dist(bl, br))))
        dst_h = max(1, round(max(math.dist(tl, bl), math.dist(tr, br))))
        dst = np.array([(0, 0), (dst_w - 1, 0), (dst_w - 1, dst_h - 1), (0, dst_h - 1)], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(rgb, matrix, (dst_w, dst_h))
        return Image.fromarray(warped)

    def _foreground_mask(self, work: "np.ndarray") -> Optional["np.ndarray"]:
        """Segment the subject from the background sampled along the photo's border.

        Deviation from the background color is measured as the residual
        ``||pixel - bg_chroma * luminance||`` in RGB units — the pixel compared
        against the background's chroma scaled to the pixel's own brightness.
        This equals chroma distance times luminance, which keeps the shadow
        invariance of chroma but stays robust where raw chroma is not: in dark
        pixels (a carpet's fiber shadows) the channel sum is small, so sensor
        and JPEG noise dominate the chroma ratio, while the residual stays
        near the noise amplitude.

        Args:
            work: Blurred float RGB working image.

        Returns:
            A uint8 mask (0/255) of foreground pixels, or None when the border
            is not uniform enough to represent a background.
        """
        work_h, work_w = work.shape[:2]
        margin = max(2, round(BORDER_FRACTION * min(work_w, work_h)))
        border_mask = np.zeros((work_h, work_w), dtype=bool)
        border_mask[:margin] = border_mask[-margin:] = True
        border_mask[:, :margin] = border_mask[:, -margin:] = True

        pix_chroma = _chroma(work)
        bg_chroma = np.median(pix_chroma[border_mask], axis=0)

        luminance = work.sum(axis=2)
        residual = np.linalg.norm(work - bg_chroma[None, None, :] * luminance[..., None], axis=2)
        border_spread = float(np.percentile(residual[border_mask], 99))
        if border_spread > MAX_BORDER_COLOR_SPREAD:
            # Border pixels vary too much: no uniform background (e.g. edge-to-edge picture)
            return None

        residual_thresh = max(COLOR_RESIDUAL_FLOOR, 1.5 * border_spread)

        # Brighter-than-background pixels are foreground too (a shadow is never brighter)
        border_lum = luminance[border_mask]
        bg_lum = float(np.median(border_lum))
        lum_spread = float(np.percentile(np.abs(border_lum - bg_lum), 99))
        lum_thresh = max(60.0, 1.5 * lum_spread)

        mask = ((residual > residual_thresh) | (luminance > bg_lum + lum_thresh)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), dtype=np.uint8))
        return cast("np.ndarray", mask)

    def _grabcut_quad(
        self, work: "np.ndarray", seed_mask: "np.ndarray", work_w: int, work_h: int
    ) -> Optional[Tuple["np.ndarray", float]]:
        """Recover the subject region with GrabCut when the color mask is fragmented.

        A washed-out puzzle under warm light can be so close to the background
        color that thresholding only picks up scattered colorful fragments,
        none large enough to pass the area gate. GrabCut — seeded with those
        fragments as probable foreground and the border ring as definite
        background — models both sides as color mixtures with spatial
        smoothness, growing the fragments out to the subject's true color
        boundary. The recovered foreground is often still fragmented — a ring
        with washed-out holes, or patches along the subject's frame — so the
        quad is fitted (and the area gate applied) on the convex hull of all
        significant recovered components, treating them as one convex subject.

        Args:
            work: Blurred float RGB working image.
            seed_mask: uint8 foreground mask (0/255) from _foreground_mask.
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            A (quad, confidence) tuple where quad has shape (4, 2), or None
            when the seeds are too sparse or no recovered region covers at
            least MIN_AREA_RATIO.
        """
        total_area = work_w * work_h
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
        seeds = np.zeros_like(seed_mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= GRABCUT_SEED_COMPONENT_RATIO * total_area:
                seeds[labels == label] = 255
        if np.count_nonzero(seeds) < GRABCUT_MIN_SEED_RATIO * total_area:
            return None

        scale = min(1.0, GRABCUT_MAX_DIM / max(work_w, work_h))
        grab_w = max(1, round(work_w * scale))
        grab_h = max(1, round(work_h * scale))
        image = cv2.resize(np.clip(work, 0, 255).astype(np.uint8), (grab_w, grab_h))
        seeds_small = cv2.resize(seeds, (grab_w, grab_h), interpolation=cv2.INTER_NEAREST)

        margin = max(2, round(BORDER_FRACTION * min(grab_w, grab_h)))
        grab = np.full((grab_h, grab_w), cv2.GC_PR_BGD, dtype=np.uint8)
        grab[seeds_small > 0] = cv2.GC_PR_FGD
        grab[:margin] = grab[-margin:] = cv2.GC_BGD
        grab[:, :margin] = grab[:, -margin:] = cv2.GC_BGD

        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        try:
            # The rect argument is unused in GC_INIT_WITH_MASK mode
            cv2.grabCut(
                image, grab, (0, 0, 0, 0), background_model, foreground_model, GRABCUT_ITERATIONS, cv2.GC_INIT_WITH_MASK
            )
        except cv2.error:
            return None

        foreground = ((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        foreground = cv2.resize(foreground, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((11, 11), dtype=np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        kept = np.zeros_like(foreground)
        kept_area = 0
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= GRABCUT_SEED_COMPONENT_RATIO * total_area:
                kept[labels == label] = 255
                kept_area += int(stats[label, cv2.CC_STAT_AREA])
        points = cv2.findNonZero(kept)
        if points is None:
            return None
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        hull_ratio = hull_area / total_area
        if hull_ratio < MIN_AREA_RATIO:
            return None

        quad = self._quad_from_hull(hull)
        confidence = self._confidence(quad, float(kept_area), hull_area, hull_ratio)
        return quad, confidence

    def _edge_quad(self, work: "np.ndarray", work_w: int, work_h: int) -> Optional[Tuple["np.ndarray", float]]:
        """Detect the puzzle's rectangular boundary directly from image edges.

        Used when there is no uniform background to segment against (e.g. a
        handheld photo where the puzzle nearly fills the frame). The puzzle
        picture is a large quadrilateral with straight edges, so this traces
        edge contours and keeps the best-scoring convex four-sided one. Unlike
        a salient-object model, it answers "where is the rectangle?" rather
        than "what does the artwork depict?".

        Args:
            work: Blurred float RGB working image.
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            A (quad, confidence) tuple where quad has shape (4, 2), or None
            when no sufficiently large convex quadrilateral is found.
        """
        gray = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        median = float(np.median(gray))
        lower = int(max(0, (1.0 - EDGE_CANNY_SIGMA) * median))
        upper = int(min(255, (1.0 + EDGE_CANNY_SIGMA) * median))
        edges = cv2.Canny(gray, lower, upper)
        edges = cv2.dilate(edges, np.ones((EDGE_DILATE_KERNEL, EDGE_DILATE_KERNEL), dtype=np.uint8))

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = work_w * work_h
        best: Optional[Tuple["np.ndarray", float]] = None
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA_RATIO * image_area:
                continue
            quad = self._quad_from_contour(contour)
            if quad is None:
                continue
            confidence = self._rect_confidence(quad, edges, work_w, work_h)
            if best is None or confidence > best[1]:
                best = (quad, confidence)
        return best

    def _quad_from_contour(self, contour: "np.ndarray") -> Optional["np.ndarray"]:
        """Reduce an edge contour to a convex quadrilateral, if it is one.

        Tries increasingly coarse polygon approximations; accepts the first
        that yields exactly four convex points. Requiring a clean four-sided
        approximation is what rejects the artwork's subject: a puzzle border
        traces a quadrilateral, an elephant does not.

        A real box's outline is often ragged rather than clean — a glare seam,
        or a slightly larger box stacked underneath, welds extra vertices onto
        the contour that no approximation tolerance removes. The convex hull
        strips those concavities while preserving the outer quadrilateral, so
        when the raw contour fails, its hull gets the same four-convex-points
        test (mirroring how the mask path quad-fits hulls).

        Args:
            contour: A contour from cv2.findContours.

        Returns:
            Array of shape (4, 2), float32, unordered, or None if neither the
            contour nor its convex hull approximates a convex quadrilateral.
        """
        for candidate in (contour, cv2.convexHull(contour)):
            perimeter = cv2.arcLength(candidate, closed=True)
            for epsilon in EDGE_APPROX_EPSILONS:
                approx = cv2.approxPolyDP(candidate, epsilon * perimeter, closed=True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    def _rect_confidence(self, quad: "np.ndarray", edges: "np.ndarray", work_w: int, work_h: int) -> float:
        """Score how puzzle-like a candidate quad is, in [0, 1].

        Combines three signals that together mean "this is the puzzle's
        rectangle" rather than merely "this is a tidy quad": how rectangular
        the quad is, how much of the photo it covers, and how much of its
        outline actually lies on detected edges (edge support). Edge support is
        the key discriminator — a quad that traces a real bordered rectangle
        sits on strong edges, an arbitrary quad does not. Support is measured
        as the excess over the edge map's overall density: on a densely
        textured background (a carpet) a random outline already hits many edge
        pixels, so only support above that chance level counts.

        Args:
            quad: The candidate quadrilateral, shape (4, 2).
            edges: The (dilated) binary edge map the quad was found in.
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            Confidence value clamped to [0, 1].
        """
        rect_w, rect_h = cv2.minAreaRect(quad.reshape(-1, 1, 2))[1]
        rect_area = rect_w * rect_h
        quad_area = cv2.contourArea(quad.reshape(-1, 1, 2))
        rectangularity = quad_area / rect_area if rect_area > 0 else 0.0
        area_ratio = quad_area / (work_w * work_h)

        outline = np.zeros((work_h, work_w), dtype=np.uint8)
        cv2.polylines(outline, [quad.astype(np.int32)], isClosed=True, color=255, thickness=EDGE_SUPPORT_THICKNESS)
        outline_pixels = int(np.count_nonzero(outline))
        edge_support = float(np.count_nonzero(edges & outline)) / outline_pixels if outline_pixels else 0.0

        edge_density = float(np.count_nonzero(edges)) / edges.size
        if edge_density >= 1.0:
            support_above_chance = 0.0
        else:
            support_above_chance = max(0.0, (edge_support - edge_density) / (1.0 - edge_density))

        confidence = rectangularity * support_above_chance * min(1.0, area_ratio / FULL_CONFIDENCE_AREA_RATIO)
        return float(np.clip(confidence, 0.0, 1.0))

    def _quad_from_mask(self, mask: "np.ndarray", work_w: int, work_h: int) -> Optional[Tuple["np.ndarray", float]]:
        """Fit a quadrilateral to the largest region of a foreground mask.

        Args:
            mask: uint8 foreground mask (0/255).
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            A (quad, confidence) tuple where quad has shape (4, 2), or None
            when the mask has no region covering at least MIN_AREA_RATIO.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area / (work_w * work_h)
        if area_ratio < MIN_AREA_RATIO:
            return None

        hull = cv2.convexHull(contour)
        quad = self._quad_from_hull(hull)
        confidence = self._confidence(quad, contour_area, cv2.contourArea(hull), area_ratio)
        return quad, confidence

    def _quad_from_hull(self, hull: "np.ndarray") -> "np.ndarray":
        """Approximate a convex hull with a quadrilateral.

        Tries increasingly coarse polygon approximations until one has exactly
        4 points; falls back to the hull's minimum-area bounding rectangle.

        Args:
            hull: Convex hull contour from cv2.convexHull.

        Returns:
            Array of shape (4, 2), float32, unordered.
        """
        perimeter = cv2.arcLength(hull, closed=True)
        for epsilon in (0.02, 0.04, 0.06, 0.08, 0.1):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, closed=True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        return cast("np.ndarray", cv2.boxPoints(cv2.minAreaRect(hull)))

    def _confidence(self, quad: "np.ndarray", contour_area: float, hull_area: float, area_ratio: float) -> float:
        """Compute a heuristic detection confidence in [0, 1].

        Combines how solid the segmented region is (contour vs hull area), how
        rectangular the fitted quad is, and how much of the photo it covers.
        This is a heuristic, not a calibrated probability.

        Args:
            quad: The fitted quadrilateral, shape (4, 2).
            contour_area: Area of the segmented contour.
            hull_area: Area of the contour's convex hull.
            area_ratio: Contour area as a fraction of the working image.

        Returns:
            Confidence value clamped to [0, 1].
        """
        hull_fill = contour_area / hull_area if hull_area > 0 else 0.0
        rect_w, rect_h = cv2.minAreaRect(quad.reshape(-1, 1, 2))[1]
        rect_area = rect_w * rect_h
        quad_area = cv2.contourArea(quad.reshape(-1, 1, 2))
        rectangularity = quad_area / rect_area if rect_area > 0 else 0.0
        confidence = hull_fill * rectangularity * min(1.0, area_ratio / FULL_CONFIDENCE_AREA_RATIO)
        return float(np.clip(confidence, 0.0, 1.0))


# Singleton instance
_puzzle_detector: Optional[PuzzleFrameDetector] = None


def get_puzzle_detector() -> PuzzleFrameDetector:
    """Get or create the singleton PuzzleFrameDetector instance.

    Returns:
        The shared PuzzleFrameDetector instance.
    """
    global _puzzle_detector
    if _puzzle_detector is None:
        _puzzle_detector = PuzzleFrameDetector()
    return _puzzle_detector
