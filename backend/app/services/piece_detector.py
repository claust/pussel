"""Service for detecting a puzzle piece region in a live camera frame."""

import io
from typing import List, Optional, Tuple, cast

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

# Cap on polygon outline points returned to the client
MAX_POLYGON_POINTS = 60


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


class PieceDetector:
    """Detects the puzzle piece (most salient object) in a camera frame via rembg."""

    def detect_region(self, image: Image.Image) -> Optional[Tuple[List[Point], BBoxNorm]]:
        """Detect the piece region in a frame.

        Args:
            image: The camera frame (any mode; converted to RGB).

        Returns:
            A (polygon, bbox) tuple with coordinates normalized to [0, 1], or
            None when background removal is disabled, segmentation fails, or
            the detected region is too small.
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
        if cv2.contourArea(contour) / (work_w * work_h) < MIN_PIECE_AREA_RATIO:
            return None

        perimeter = cv2.arcLength(contour, closed=True)
        polygon = cv2.approxPolyDP(contour, 0.005 * perimeter, closed=True).reshape(-1, 2)
        if len(polygon) > MAX_POLYGON_POINTS:
            step = len(polygon) / MAX_POLYGON_POINTS
            polygon = polygon[[int(i * step) for i in range(MAX_POLYGON_POINTS)]]

        points: List[Point] = [
            (float(np.clip(x / work_w, 0.0, 1.0)), float(np.clip(y / work_h, 0.0, 1.0))) for x, y in polygon
        ]
        x, y, w, h = cast(Tuple[int, int, int, int], cv2.boundingRect(contour))
        bbox: BBoxNorm = (
            float(np.clip(x / work_w, 0.0, 1.0)),
            float(np.clip(y / work_h, 0.0, 1.0)),
            float(np.clip(w / work_w, 0.0, 1.0)),
            float(np.clip(h / work_h, 0.0, 1.0)),
        )
        return points, bbox


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
