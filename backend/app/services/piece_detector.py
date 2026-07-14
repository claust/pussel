"""Service for detecting a puzzle piece region in a live camera frame."""

import io
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, cast
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.background_remover import get_background_remover
from app.services.piece_classifier import get_piece_classifier, prepare_classifier_input

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

# Floor for the confidence reported on an accepted region, so a detection
# (found=True) always carries a strictly positive confidence in (0, 1] even
# when the raw score rounds to 0.000.
MIN_REPORTED_CONFIDENCE = 0.001


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


def save_preview_crop(rgba: Image.Image, confidence: float) -> None:
    """Persist the classifier-input crop of an accepted preview region.

    Dev-only (gated by ``SAVE_PREVIEW_CROPS``): the saved crops let real-world
    false positives be harvested as hard negatives for classifier retraining.
    Never raises — the preview loop must not break over a disk hiccup.

    Args:
        rgba: The rembg RGBA output for the frame.
        confidence: The confidence reported for the region (encoded in the
            filename so false positives are easy to spot).
    """
    try:
        crop = prepare_classifier_input(rgba)
        if crop is None:
            return
        crops_dir = Path(settings.UPLOAD_DIR) / "preview_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        # uuid suffix: timestamps alone can collide under rapid preview polling
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crop.save(crops_dir / f"{timestamp}_{uuid4().hex[:8]}_c{confidence:.3f}.png", "PNG")
    except Exception as exc:
        # Dev-only feature: log so failures are diagnosable, but never break
        # the preview loop over a disk hiccup.
        print(f"Failed to save preview crop: {exc}")


class PieceDetector:
    """Detects the puzzle piece (most salient object) in a camera frame via rembg."""

    def _region_confidence(
        self,
        rgba: Image.Image,
        contour: "np.ndarray",
        area_score: float,
        aspect_score: float,
    ) -> Optional[float]:
        """Score how piece-like an accepted candidate region is.

        When the trained piece classifier is available, its probability is the
        confidence (the area/aspect gates have already filtered impossible
        regions). Without a classifier checkpoint (e.g. in CI) the original
        heuristic — area/aspect taper plus the skin-tone gate — is used.

        Args:
            rgba: The rembg RGBA output for the frame.
            contour: The detected region's contour.
            area_score: Area band score in [0, 1].
            aspect_score: Aspect band score in [0, 1].

        Returns:
            Confidence in (0, 1], or None when the region is rejected.
        """
        classifier = get_piece_classifier()
        if classifier.available:
            score = classifier.score(rgba)
            if score is not None:
                # Report the full probability (no rounding: it could flip values
                # across the client's 0.5 gate), floored so found=True always
                # carries a confidence in (0, 1].
                return max(score, MIN_REPORTED_CONFIDENCE)

        # Heuristic fallback: the skin gate is what rejects faces and hands
        # when no trained classifier is around to do it better.
        skin_score = _band_score(skin_fraction(rgba, contour), 0.0, 0.0, FULL_CONFIDENCE_MAX_SKIN, MAX_SKIN_FRACTION)
        if skin_score == 0.0:
            return None

        # Gate on the raw product so rounding never acts as an extra hard limit
        # (a tiny-but-nonzero product must survive and report a low confidence).
        raw_confidence = area_score * aspect_score * skin_score
        if raw_confidence <= 0.0:
            return None
        # Floor the reported value so an accepted region keeps confidence in
        # (0, 1] even when the raw score would round to 0.000.
        return max(round(raw_confidence, 3), MIN_REPORTED_CONFIDENCE)

    def detect_region(self, image: Image.Image) -> Optional[DetectedRegion]:
        """Detect the piece region in a frame.

        The most salient region found by background removal is only accepted
        when it plausibly is a puzzle piece: it must occupy a piece-like
        fraction of the frame and not be extremely elongated. The confidence
        comes from the trained piece/not-piece classifier when available,
        otherwise from the heuristic scores (including the skin-tone gate,
        which rejects the faces and hands rembg most often latches onto).

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

        confidence = self._region_confidence(rgba, contour, area_score, aspect_score)
        if confidence is None:
            return None

        if settings.SAVE_PREVIEW_CROPS:
            save_preview_crop(rgba, confidence)

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
