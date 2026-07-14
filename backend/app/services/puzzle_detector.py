"""Service for detecting a puzzle picture's boundary in a photo and perspective-correcting it."""

import io
import math
from typing import List, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image

from app.services.background_remover import get_background_remover

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

# If the border ring's 99th-percentile chroma deviation exceeds this, there is no
# uniform background around the subject (e.g. an edge-to-edge image) and detection bails out
MAX_BORDER_CHROMA_SPREAD = 0.06


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
    uniform background (sampled from the photo's border) using chroma distance
    — which ignores shadows — plus a brighter-than-background luminance rule.
    When the border is not a uniform background (e.g. a webcam shot with a
    cluttered room behind the puzzle), it falls back to rembg salient-object
    segmentation. Either way a quadrilateral is fitted to the convex hull of
    the largest segmented region.
    """

    def __init__(self, salient_fallback: bool = True) -> None:
        """Initialize the detector.

        Args:
            salient_fallback: Whether to fall back to rembg salient-object
                segmentation when background-based segmentation is not
                applicable. Disable for fast, dependency-free detection only.
        """
        self._salient_fallback = salient_fallback

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

        if result is None and self._salient_fallback:
            mask = self._salient_mask(rgb)
            result = self._quad_from_mask(mask, work_w, work_h) if mask is not None else None

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
        border_chroma = pix_chroma[border_mask]
        bg_chroma = np.median(border_chroma, axis=0)
        border_spread = float(np.percentile(np.linalg.norm(border_chroma - bg_chroma, axis=1), 99))
        if border_spread > MAX_BORDER_CHROMA_SPREAD:
            # Border pixels vary too much: no uniform background (e.g. edge-to-edge picture)
            return None

        chroma_dist = np.linalg.norm(pix_chroma - bg_chroma, axis=2)
        chroma_thresh = max(0.03, 1.5 * border_spread)

        # Brighter-than-background pixels are foreground too (a shadow is never brighter)
        luminance = work.sum(axis=2)
        border_lum = luminance[border_mask]
        bg_lum = float(np.median(border_lum))
        lum_spread = float(np.percentile(np.abs(border_lum - bg_lum), 99))
        lum_thresh = max(60.0, 1.5 * lum_spread)

        mask = ((chroma_dist > chroma_thresh) | (luminance > bg_lum + lum_thresh)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), dtype=np.uint8))
        return cast("np.ndarray", mask)

    def _salient_mask(self, rgb: "np.ndarray") -> Optional["np.ndarray"]:
        """Segment the most salient object using rembg (u2net).

        Used when there is no uniform background to segment against, e.g. a
        webcam shot of a puzzle held up in front of a cluttered room.

        Args:
            rgb: Working-size uint8 RGB image.

        Returns:
            A uint8 mask (0/255) of the salient object, or None if
            segmentation fails.
        """
        try:
            buffer = io.BytesIO()
            Image.fromarray(rgb).save(buffer, format="PNG")
            rgba = get_background_remover().remove_background(buffer.getvalue())
        except Exception:
            return None
        if rgba.mode != "RGBA":
            return None
        alpha = np.asarray(rgba.split()[3])
        mask = (alpha > 128).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), dtype=np.uint8))
        return cast("np.ndarray", mask)

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
