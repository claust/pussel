"""Service for detecting a puzzle piece region in a live camera frame."""

import io
from typing import List, NamedTuple, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.background_remover import get_background_remover

Point = Tuple[float, float]
BBoxNorm = Tuple[float, float, float, float]  # x, y, width, height normalized to [0, 1]

# Frames are downscaled before segmentation to keep the preview loop fast
PREVIEW_MAX_DIM = 320

# The detected region must cover at least this fraction of the frame
MIN_PIECE_AREA_RATIO = 0.005

# ... and at most this fraction. A hand-held piece at capture distance covers a
# small part of the frame; a face or torso at webcam distance covers far more.
MAX_PIECE_AREA_RATIO = 0.35

# Area band (fraction of frame) that gets full confidence; outside it the
# confidence tapers linearly toward the hard limits above.
FULL_CONFIDENCE_AREA_RANGE = (0.01, 0.15)

# Bounding-box aspect ratio (long side / short side) limits. Pieces are roughly
# square-ish; long thin regions are arms, table edges, cables, etc.
MAX_ASPECT_RATIO = 3.5
FULL_CONFIDENCE_MAX_ASPECT = 2.0

# Fraction of skin-tone pixels inside the region above which it is rejected.
# The threshold is generous because fingers holding a piece legitimately add
# some skin pixels to the segmented region.
MAX_SKIN_FRACTION = 0.6
FULL_CONFIDENCE_MAX_SKIN = 0.15

# Cap on polygon outline points returned to the client
MAX_POLYGON_POINTS = 60


class DetectedRegion(NamedTuple):
    """A detected piece candidate with normalized coordinates and confidence."""

    polygon: List[Point]
    bbox: BBoxNorm
    confidence: float


def largest_alpha_component(rgba: Image.Image, alpha_threshold: int = 128) -> Optional["np.ndarray"]:
    """Find the contour of the largest opaque region in an RGBA image.

    Args:
        rgba: RGBA image, e.g. rembg output.
        alpha_threshold: Alpha value above which a pixel counts as opaque.

    Returns:
        The largest contour from cv2.findContours, or None when the image has
        no alpha channel or no opaque region.
    """
    if rgba.mode != "RGBA":
        return None
    alpha = np.asarray(rgba.split()[3])
    mask = (alpha > alpha_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def crop_to_alpha_region(rgba: Image.Image, margin: float = 0.08) -> Image.Image:
    """Crop an RGBA image to the bounding box of its largest opaque region.

    Centering the segmented subject this way puts camera captures much closer
    to the model's training distribution (pieces filling the frame) than a
    small cutout floating in a large canvas.

    Args:
        rgba: RGBA image, e.g. rembg output.
        margin: Extra margin around the bounding box as a fraction of its size.

    Returns:
        The cropped image, or the input unchanged when no opaque region exists.
    """
    contour = largest_alpha_component(rgba)
    if contour is None:
        return rgba
    x, y, w, h = cv2.boundingRect(contour)
    pad_x = round(w * margin)
    pad_y = round(h * margin)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(rgba.width, x + w + pad_x)
    bottom = min(rgba.height, y + h + pad_y)
    if right <= left or bottom <= top:
        return rgba
    return rgba.crop((left, top, right, bottom))


def skin_fraction(rgba: Image.Image, contour: Optional["np.ndarray"] = None, alpha_threshold: int = 128) -> float:
    """Estimate the fraction of skin-tone pixels in an RGBA image's opaque region.

    Uses the classic YCrCb skin-tone box (Cr in [133, 173], Cb in [77, 127]),
    which is intentionally broad: it matters far more that faces score high
    than that every piece scores exactly zero.

    Args:
        rgba: RGBA image, e.g. rembg output.
        contour: Optional contour restricting the measurement to one region;
            when omitted, all opaque pixels are measured.
        alpha_threshold: Alpha value above which a pixel counts as opaque.

    Returns:
        Skin-tone pixel fraction in [0, 1]; 0.0 when there are no opaque pixels.
    """
    arr = np.asarray(rgba.convert("RGBA"))
    mask = arr[..., 3] > alpha_threshold
    if contour is not None:
        region = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
        mask = mask & (region > 0)
    total = np.count_nonzero(mask)
    if total == 0:
        return 0.0
    ycrcb = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(np.count_nonzero(skin & mask) / total)


def _band_score(value: float, hard_low: float, full_low: float, full_high: float, hard_high: float) -> float:
    """Score a value against a confidence band.

    Args:
        value: The measured value.
        hard_low: Below this the score is 0.0.
        full_low: Lower edge of the full-confidence band.
        full_high: Upper edge of the full-confidence band.
        hard_high: Above this the score is 0.0.

    Returns:
        1.0 inside [full_low, full_high], tapering linearly to 0.0 at the hard
        limits.
    """
    if value < full_low:
        if value <= hard_low:
            return 0.0
        return (value - hard_low) / (full_low - hard_low)
    if value > full_high:
        if value >= hard_high:
            return 0.0
        return (hard_high - value) / (hard_high - full_high)
    return 1.0


class PieceDetector:
    """Detects the puzzle piece (most salient object) in a camera frame via rembg."""

    def detect_region(self, image: Image.Image) -> Optional[DetectedRegion]:
        """Detect the piece region in a frame.

        The most salient region found by background removal is only accepted
        when it plausibly is a puzzle piece: it must occupy a piece-like
        fraction of the frame, not be extremely elongated, and not be
        dominated by skin tones (faces and hands are what rembg most often
        latches onto when no piece is held up).

        Args:
            image: The camera frame (any mode; converted to RGB).

        Returns:
            A DetectedRegion with coordinates normalized to [0, 1] and a
            confidence in (0, 1], or None when background removal is disabled,
            segmentation fails, or the detected region does not look like a
            puzzle piece.
        """
        if not settings.ENABLE_BACKGROUND_REMOVAL:
            return None

        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = min(1.0, PREVIEW_MAX_DIM / max(width, height))
        if scale < 1.0:
            rgb = rgb.resize((max(1, round(width * scale)), max(1, round(height * scale))))
        work_w, work_h = rgb.size

        try:
            buffer = io.BytesIO()
            rgb.save(buffer, format="PNG")
            rgba = get_background_remover(settings.REMBG_MODEL).remove_background(buffer.getvalue())
        except Exception:
            return None

        contour = largest_alpha_component(rgba)
        if contour is None:
            return None

        area_ratio = cv2.contourArea(contour) / (work_w * work_h)
        bx, by, bw, bh = cast(Tuple[int, int, int, int], cv2.boundingRect(contour))
        aspect = max(bw, bh) / max(1, min(bw, bh))

        area_score = _band_score(area_ratio, MIN_PIECE_AREA_RATIO, *FULL_CONFIDENCE_AREA_RANGE, MAX_PIECE_AREA_RATIO)
        aspect_score = _band_score(aspect, 0.0, 1.0, FULL_CONFIDENCE_MAX_ASPECT, MAX_ASPECT_RATIO)
        if area_score == 0.0 or aspect_score == 0.0:
            return None

        skin_score = _band_score(skin_fraction(rgba, contour), 0.0, 0.0, FULL_CONFIDENCE_MAX_SKIN, MAX_SKIN_FRACTION)
        if skin_score == 0.0:
            return None

        # Gate on the raw product so rounding never acts as an extra hard limit
        # (a tiny-but-nonzero product must survive and report a low confidence).
        raw_confidence = area_score * aspect_score * skin_score
        if raw_confidence <= 0.0:
            return None
        confidence = round(raw_confidence, 3)

        perimeter = cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(contour, 0.005 * perimeter, closed=True).reshape(-1, 2)
        if len(polygon) > MAX_POLYGON_POINTS:
            step = len(polygon) / MAX_POLYGON_POINTS
            polygon = polygon[[int(i * step) for i in range(MAX_POLYGON_POINTS)]]

        points: List[Point] = [
            (float(np.clip(x / work_w, 0.0, 1.0)), float(np.clip(y / work_h, 0.0, 1.0))) for x, y in polygon
        ]
        bbox: BBoxNorm = (
            float(np.clip(bx / work_w, 0.0, 1.0)),
            float(np.clip(by / work_h, 0.0, 1.0)),
            float(np.clip(bw / work_w, 0.0, 1.0)),
            float(np.clip(bh / work_h, 0.0, 1.0)),
        )
        return DetectedRegion(polygon=points, bbox=bbox, confidence=confidence)


# Singleton instance
_piece_detector: Optional[PieceDetector] = None


def get_piece_detector() -> PieceDetector:
    """Get or create the singleton PieceDetector instance.

    Returns:
        The shared PieceDetector instance.
    """
    global _piece_detector
    if _piece_detector is None:
        _piece_detector = PieceDetector()
    return _piece_detector
