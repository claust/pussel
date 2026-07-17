"""Contour extraction, smoothing, resampling, and quality metrics.

Ported from ``network/experiments/exp28_piece_geometry/common.py`` — keep
algorithm changes in sync. Adapted for production: operates on a single
in-memory alpha mask (from rembg) rather than a north_star dataset crop, and
replaces ``scipy.ndimage.gaussian_filter1d`` (the backend has no scipy) with
a small numpy circular-Gaussian helper that is numerically equivalent to
scipy's ``mode="wrap"`` filtering (same truncated kernel, same wraparound).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Alpha value above which a rembg pixel counts as opaque (matches
# app/services/piece_detector.py's harden_alpha/largest_alpha_component).
ALPHA_THRESHOLD = 128

# Morphological open+close kernel used to clean up masks before contour extraction.
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# A connected component must be at least this fraction of the largest component's
# area to count as a distinct "large" component for quality scoring.
LARGE_COMPONENT_AREA_FRAC = 0.02

# A contour point within this many pixels of the crop edge counts as border-touching.
BORDER_TOUCH_MARGIN_PX = 2.0

# Kernel half-width in standard deviations, matching scipy.ndimage.gaussian_filter1d's default.
GAUSSIAN_TRUNCATE = 4.0


def alpha_to_mask(alpha: np.ndarray, alpha_threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
    """Harden a soft alpha matte into a binary mask.

    Args:
        alpha: Single-channel alpha array (H, W), values in [0, 255].
        alpha_threshold: Alpha value above which a pixel counts as opaque.

    Returns:
        Binary mask (H, W) with values in {0, 255}.
    """
    return np.where(alpha > alpha_threshold, np.uint8(255), np.uint8(0))


def largest_component_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the largest connected component in a binary mask.

    Mirrors the intent of `app.services.piece_detector.largest_alpha_component`
    (pick the biggest opaque blob as the piece candidate), but returns the
    component's bbox directly: the piece-geometry pipeline needs the crop
    window, not the contour, at this stage.

    Args:
        mask: Binary mask (H, W) with values in {0, 255}.

    Returns:
        Inclusive pixel bbox (x1, y1, x2, y2) of the largest component, or
        None when the mask has no foreground pixels.
    """
    n_components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_components <= 1:
        return None
    areas = stats[1:n_components, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x = int(stats[largest, cv2.CC_STAT_LEFT])
    y = int(stats[largest, cv2.CC_STAT_TOP])
    w = int(stats[largest, cv2.CC_STAT_WIDTH])
    h = int(stats[largest, cv2.CC_STAT_HEIGHT])
    return x, y, x + w - 1, y + h - 1


def crop_with_margin(
    image: np.ndarray, bbox: Tuple[int, int, int, int], margin_frac: float = 0.15
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop an image to a bounding box expanded by a margin, clamped to image bounds.

    This reproduces exp28's frame of reference: the calibrated pipeline
    (quality gates, corner windows, spatial color grid) was tuned on
    piece-bbox crops with a 15% margin, not on full camera frames.

    Args:
        image: Source image as a numpy array (H, W, C) or (H, W).
        bbox: Bounding box (x1, y1, x2, y2), inclusive pixel coordinates.
        margin_frac: Extra margin added on each side, as a fraction of the
            bbox width/height.

    Returns:
        Tuple of (crop, (offset_x, offset_y)) where offset is the crop's
        top-left corner in the original image's pixel coordinates.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = round(box_w * margin_frac)
    pad_y = round(box_h * margin_frac)

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x + 1)
    bottom = min(height, y2 + pad_y + 1)

    crop = image[top:bottom, left:right]
    return crop, (left, top)


def mask_to_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Clean up a binary mask and extract its largest external contour.

    Applies a morphological open then close (5px ellipse) to remove speckle
    noise and close small gaps, then returns the largest external contour at
    full point density.

    Args:
        mask: Binary mask (H, W) with values in {0, 255}.

    Returns:
        Nx2 float array of contour points in the mask's own pixel coordinates,
        or None when no contour is found.
    """
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, MORPH_KERNEL)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    return largest.reshape(-1, 2).astype(np.float64)


def _gaussian_kernel1d(sigma: float, truncate: float = GAUSSIAN_TRUNCATE) -> np.ndarray:
    """Build a normalized 1D Gaussian kernel, matching scipy's gaussian_filter1d.

    Args:
        sigma: Gaussian standard deviation in samples.
        truncate: Kernel half-width in standard deviations.

    Returns:
        1D array of length ``2 * radius + 1`` (``radius = int(truncate * sigma + 0.5)``),
        summing to 1.
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_filter1d_wrap(values: np.ndarray, sigma: float) -> np.ndarray:
    """Circularly filter a 1D signal with a Gaussian kernel, numpy-only.

    Numerically equivalent to
    ``scipy.ndimage.gaussian_filter1d(values, sigma=sigma, mode="wrap")``:
    same truncated kernel (see `_gaussian_kernel1d`) and the same circular
    (wrap) boundary handling. Because the kernel is symmetric, convolution
    and correlation coincide, so a plain `numpy.convolve` on a
    wrap-padded signal reproduces scipy's `correlate1d`-based result exactly.

    Args:
        values: 1D signal, treated as one period of a periodic (closed-loop) signal.
        sigma: Gaussian standard deviation in samples.

    Returns:
        The filtered signal, same length as `values`.
    """
    kernel = _gaussian_kernel1d(sigma)
    radius = (len(kernel) - 1) // 2
    padded = np.concatenate([values[-radius:], values, values[:radius]])
    return np.convolve(padded, kernel, mode="valid")


def smooth_contour(contour: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth a closed contour with circular Gaussian filtering.

    Args:
        contour: Nx2 array of contour points, implicitly closed (first point
            follows the last).
        sigma: Gaussian standard deviation in samples.

    Returns:
        Nx2 smoothed contour, same length as the input.
    """
    x = gaussian_filter1d_wrap(contour[:, 0], sigma=sigma)
    y = gaussian_filter1d_wrap(contour[:, 1], sigma=sigma)
    return np.column_stack([x, y])


def resample_contour(contour: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed contour to `n` arc-length-equidistant points.

    Args:
        contour: Nx2 array of contour points, implicitly closed.
        n: Number of output points.

    Returns:
        Nx2 resampled contour.
    """
    closed = np.vstack([contour, contour[:1]])
    diffs = np.diff(closed, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        idx = np.linspace(0, len(contour) - 1, n).astype(int)
        return contour[idx]

    target_lengths = np.linspace(0, total_length, n, endpoint=False)
    resampled_x = np.interp(target_lengths, cumulative_length, closed[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_length, closed[:, 1])
    return np.column_stack([resampled_x, resampled_y])


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an OPEN polyline to `n` arc-length-equidistant points.

    Unlike `resample_contour`, the polyline is not closed: the first and last
    output points coincide with the input endpoints.

    Args:
        points: Mx2 array of polyline points.
        n: Number of output points (>= 2).

    Returns:
        Nx2 resampled polyline.
    """
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        return np.repeat(points[:1], n, axis=0)

    target_lengths = np.linspace(0, total_length, n)
    resampled_x = np.interp(target_lengths, cumulative_length, points[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_length, points[:, 1])
    return np.column_stack([resampled_x, resampled_y])


@dataclass(frozen=True)
class QualityMetrics:
    """Quality signals for a contour extracted from a piece photo.

    Attributes:
        n_large_components: Number of connected components in the source mask
            with area at least `LARGE_COMPONENT_AREA_FRAC` of the largest one.
        border_touching: Whether any contour point lies within
            `BORDER_TOUCH_MARGIN_PX` of the crop edge (segmentation likely
            clipped the piece).
        area_ratio: Contour area divided by the crop's pixel area.
        solidity: Contour area divided by its convex hull area (low solidity
            indicates a ragged or multi-lobed contour).
        is_clean: Overall pass/fail: exactly one large component, not
            border-touching, plausible area ratio, and plausible solidity.
    """

    n_large_components: int
    border_touching: bool
    area_ratio: float
    solidity: float
    is_clean: bool


def contour_quality(contour: np.ndarray, mask: np.ndarray, crop_shape: Tuple[int, int]) -> QualityMetrics:
    """Score how trustworthy an extracted contour is.

    Args:
        contour: Nx2 contour points in the mask's own pixel coordinates.
        mask: The (uncleaned) binary mask the contour was derived from,
            used to count connected components.
        crop_shape: (height, width) of the crop the mask was computed on.

    Returns:
        The computed `QualityMetrics`.
    """
    height, width = crop_shape
    crop_area = float(height * width)

    n_components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = stats[1:n_components, cv2.CC_STAT_AREA] if n_components > 1 else np.array([])
    if len(component_areas) == 0:
        n_large_components = 0
    else:
        largest_area = float(component_areas.max())
        n_large_components = int(np.sum(component_areas >= LARGE_COMPONENT_AREA_FRAC * largest_area))

    border_touching = bool(
        np.any(contour[:, 0] <= BORDER_TOUCH_MARGIN_PX)
        or np.any(contour[:, 1] <= BORDER_TOUCH_MARGIN_PX)
        or np.any(contour[:, 0] >= width - 1 - BORDER_TOUCH_MARGIN_PX)
        or np.any(contour[:, 1] >= height - 1 - BORDER_TOUCH_MARGIN_PX)
    )

    contour_area = float(cv2.contourArea(contour.astype(np.float32)))
    area_ratio = contour_area / crop_area if crop_area > 0 else 0.0

    hull = cv2.convexHull(contour.astype(np.float32))
    hull_area = float(cv2.contourArea(hull))
    solidity = contour_area / hull_area if hull_area > 0 else 0.0

    is_clean = (
        n_large_components == 1 and not border_touching and 0.05 <= area_ratio <= 0.9 and 0.6 <= solidity <= 0.995
    )

    return QualityMetrics(
        n_large_components=n_large_components,
        border_touching=border_touching,
        area_ratio=area_ratio,
        solidity=solidity,
        is_clean=is_clean,
    )
