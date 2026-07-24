"""Shared utilities: glare-free stitch-quality scoring and offline stitching.

Both `score_stitch.py` (metrics) and `stitch.py` (offline reimplementation)
consume the same DEBUG-build capture-dump directory produced by the iOS
app's glare-free capture flow: `reference.jpg` (the centered shot),
`corner_1.jpg`..`corner_4.jpg` (the 4 corner-offset shots -- 0 to 4 of which
may be present), `composite.jpg` (the app's min-composited glare-free
result), and an optional `metadata.json` (timestamp, expectedShifts,
alignedFrameCount). The tools must also run on 6 loose images with no
metadata.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REFERENCE_FILENAME = "reference.jpg"
COMPOSITE_FILENAME = "composite.jpg"
CORNER_FILENAME_TEMPLATE = "corner_{index}.jpg"
METADATA_FILENAME = "metadata.json"
N_CORNERS = 4

# Long-side size (pixels) all scoring/stitching runs at, so results are comparable
# across dumps captured at different device resolutions.
WORKING_LONG_SIDE = 2048

SIFT_RATIO_TEST_THRESH = 0.75
RANSAC_REPROJ_THRESH_PX = 4.0
MIN_MATCHES_FOR_HOMOGRAPHY = 4

PATCH_SIZE_PX = 64
# Reference-side patch std below this is treated as near-uniform (blank cardboard, sky,
# glare wash) -- phase correlation is unreliable there, so these patches are excluded.
PATCH_MIN_STD = 4.0

# Blur applied to both images before differencing for the darkening map, to suppress
# resample/JPEG noise rather than reacting to it pixel-for-pixel.
DARKENING_BLUR_SIGMA = 1.0
# Per-pixel darkening (0-255 gray levels) above which a pixel counts as "healed" for the
# glare-healing fraction/mean-over-darkened metrics.
DARKENED_PIXEL_THRESHOLD = 8.0
# Per-patch MEAN darkening above which a local-ghosting grid patch is excluded from shift
# statistics as "healed" rather than "misaligned". Empirically tuned against a real capture
# with a broad glare sheen (scripts/stitch_quality/README.md): comfortably
# above the ~18-27 gray-level baseline seen even on unaffected/background patches in that
# capture, but below the ~40-76 range of the confirmed healed-sheen cluster.
HEALED_PATCH_DARKENING_THRESHOLD = 25.0

# Ellipse kernel side length (pixels) for the white top-hat filter used to detect small
# bright specks (e.g. stars on a dark background) -- see `detect_bright_specks`.
SPECK_TOPHAT_KERNEL_PX = 7
# Top-hat response (0-255 gray levels) above which a pixel is considered part of a bright
# speck. Empirically tuned against a real starry-artwork capture
# (scripts/stitch_quality/README.md): low enough to catch genuine small
# stars, high enough that JPEG/resample texture noise (which explodes the component count
# past this point) isn't mistaken for one.
SPECK_BRIGHTNESS_THRESHOLD = 12.0
# Connected-component area range (pixels, working size) counted as a "speck" rather than
# noise (too small) or a larger bright feature that isn't fine detail (too large).
SPECK_MIN_AREA_PX = 1
SPECK_MAX_AREA_PX = 40


@dataclass(frozen=True)
class CaptureDump:
    """One glare-free capture dump, loaded and normalized to a common working size.

    Attributes:
        dump_dir: Source directory.
        reference: Centered reference shot, BGR, resized to the working long side.
        composite: App-produced min-composite, BGR, same pixel size as `reference`.
        corners: One BGR image per present `corner_N.jpg`, keyed by 1-based index;
            missing corner shots are simply absent from the dict.
        metadata: Parsed `metadata.json` contents, or None if the file is absent.
        scale: Factor applied to `reference`'s original pixels to reach the working
            long side (`working_px = original_px * scale`).
    """

    dump_dir: Path
    reference: np.ndarray
    composite: np.ndarray
    corners: Dict[int, np.ndarray]
    metadata: Optional[dict]
    scale: float


def _imread(path: Path) -> np.ndarray:
    """Read a color image, raising a clear error instead of returning None.

    Args:
        path: Image file path.

    Returns:
        BGR image array.

    Raises:
        FileNotFoundError: If the file is missing or unreadable by OpenCV.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_to_long_side(image: np.ndarray, long_side: int = WORKING_LONG_SIDE) -> Tuple[np.ndarray, float]:
    """Resize an image so its longer side equals `long_side`, preserving aspect ratio.

    Args:
        image: Source image (H, W, C) or (H, W).
        long_side: Target length of the longer dimension in pixels.

    Returns:
        Tuple of (resized image, scale factor applied).
    """
    height, width = image.shape[:2]
    scale = long_side / max(height, width)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, new_size, interpolation=interpolation)
    return resized, scale


def load_dump(dump_dir: Path, long_side: int = WORKING_LONG_SIDE) -> CaptureDump:
    """Load a capture-dump directory, resizing everything to a common working size.

    Args:
        dump_dir: Directory containing `reference.jpg`, `composite.jpg`, 0-4
            `corner_N.jpg` files, and an optional `metadata.json`.
        long_side: Target long-side size in pixels for all loaded images (the
            source files may differ slightly in native size due to re-encoding).

    Returns:
        The loaded, normalized `CaptureDump`.

    Raises:
        FileNotFoundError: If `reference.jpg` or `composite.jpg` is missing.
    """
    reference_path = dump_dir / REFERENCE_FILENAME
    composite_path = dump_dir / COMPOSITE_FILENAME
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing required {REFERENCE_FILENAME} in {dump_dir}")
    if not composite_path.exists():
        raise FileNotFoundError(f"Missing required {COMPOSITE_FILENAME} in {dump_dir}")

    reference, scale = resize_to_long_side(_imread(reference_path), long_side)
    working_size = (reference.shape[1], reference.shape[0])

    composite, _ = resize_to_long_side(_imread(composite_path), long_side)
    if composite.shape[:2] != reference.shape[:2]:
        # Composite should already live in the reference frame; only happens if the
        # two source JPEGs differ slightly in native aspect ratio.
        composite = cv2.resize(composite, working_size, interpolation=cv2.INTER_AREA)

    corners: Dict[int, np.ndarray] = {}
    for index in range(1, N_CORNERS + 1):
        corner_path = dump_dir / CORNER_FILENAME_TEMPLATE.format(index=index)
        if corner_path.exists():
            corners[index], _ = resize_to_long_side(_imread(corner_path), long_side)

    metadata: Optional[dict] = None
    metadata_path = dump_dir / METADATA_FILENAME
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)

    return CaptureDump(
        dump_dir=dump_dir,
        reference=reference,
        composite=composite,
        corners=corners,
        metadata=metadata,
        scale=scale,
    )


@dataclass(frozen=True)
class FeatureMatchResult:
    """Result of SIFT + Lowe's-ratio-test + RANSAC-homography matching between two images.

    Attributes:
        n_keypoints_src: Number of SIFT keypoints detected in the source image.
        n_keypoints_dst: Number of SIFT keypoints detected in the destination image.
        n_ratio_matches: Number of matches surviving the ratio test.
        src_pts: (n_ratio_matches, 2) matched keypoint coordinates in the source image.
        dst_pts: (n_ratio_matches, 2) corresponding coordinates in the destination image.
        homography: 3x3 homography mapping src -> dst, or None when there were too
            few ratio-test matches to attempt RANSAC.
        inlier_mask: Boolean array (n_ratio_matches,) marking RANSAC inliers, or None.
    """

    n_keypoints_src: int
    n_keypoints_dst: int
    n_ratio_matches: int
    src_pts: np.ndarray
    dst_pts: np.ndarray
    homography: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]

    @property
    def n_inliers(self) -> int:
        """Number of RANSAC inlier matches (0 when no homography was found)."""
        return int(self.inlier_mask.sum()) if self.inlier_mask is not None else 0

    @property
    def inlier_ratio(self) -> float:
        """Fraction of ratio-test matches that are RANSAC inliers (0 when there are none)."""
        return self.n_inliers / self.n_ratio_matches if self.n_ratio_matches else 0.0


_sift = None


def _get_sift():  # type: ignore[no-untyped-def]
    """Get or create the shared `cv2.SIFT` detector (created once, reused across calls)."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    return _sift


def match_sift_ransac(
    gray_src: np.ndarray,
    gray_dst: np.ndarray,
    ratio: float = SIFT_RATIO_TEST_THRESH,
    ransac_reproj_thresh: float = RANSAC_REPROJ_THRESH_PX,
) -> FeatureMatchResult:
    """Match two grayscale images with SIFT, Lowe's ratio test, and a RANSAC homography.

    Args:
        gray_src: Source single-channel image.
        gray_dst: Destination single-channel image. For a correctly registered pair
            (e.g. composite vs reference), `gray_src` and `gray_dst` share a coordinate
            frame and the fitted homography should be close to identity.
        ratio: Lowe's ratio-test threshold (lower = stricter).
        ransac_reproj_thresh: RANSAC reprojection threshold in pixels for `cv2.findHomography`.

    Returns:
        The `FeatureMatchResult`. `homography`/`inlier_mask` are None when fewer than
        `MIN_MATCHES_FOR_HOMOGRAPHY` ratio-test matches were found.
    """
    sift = _get_sift()
    kp_src, desc_src = sift.detectAndCompute(gray_src, None)
    kp_dst, desc_dst = sift.detectAndCompute(gray_dst, None)
    n_kp_src, n_kp_dst = len(kp_src), len(kp_dst)
    empty = np.empty((0, 2), dtype=np.float64)

    if desc_src is None or desc_dst is None or len(kp_src) < 2 or len(kp_dst) < 2:
        return FeatureMatchResult(n_kp_src, n_kp_dst, 0, empty, empty, None, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc_src, desc_dst, k=2)
    good = [pair[0] for pair in raw_matches if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]

    src_pts = np.array([kp_src[m.queryIdx].pt for m in good], dtype=np.float64) if good else empty
    dst_pts = np.array([kp_dst[m.trainIdx].pt for m in good], dtype=np.float64) if good else empty

    if len(good) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return FeatureMatchResult(n_kp_src, n_kp_dst, len(good), src_pts, dst_pts, None, None)

    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else None
    return FeatureMatchResult(n_kp_src, n_kp_dst, len(good), src_pts, dst_pts, homography, inlier_mask)


def _masked_values(values: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Flatten an array, optionally restricted to where `mask` is nonzero.

    Args:
        values: Array to select from.
        mask: Optional uint8/bool mask, same (height, width) as `values`; nonzero = keep.

    Returns:
        1-D array of the selected values (all of `values` when `mask` is None).
    """
    return values[mask > 0] if mask is not None else values.ravel()


def variance_of_laplacian(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Focus/sharpness proxy: variance of the Laplacian (higher = sharper).

    Args:
        gray: Single-channel image.
        mask: Optional region mask (see `_masked_values`); restricts the variance to
            that region instead of the whole image.

    Returns:
        The Laplacian variance. 0.0 if `mask` selects no pixels.
    """
    selected = _masked_values(cv2.Laplacian(gray, cv2.CV_64F), mask)
    return float(selected.var()) if selected.size else 0.0


def canny_edge_count(gray: np.ndarray, low: int = 50, high: int = 150, mask: Optional[np.ndarray] = None) -> int:
    """Count Canny edge pixels.

    Canny always runs on the full image first, so edge detection has full context; `mask`
    only restricts which resulting edge pixels get counted, avoiding false edges that
    masking-then-detecting would introduce at the region boundary.

    Args:
        gray: Single-channel image.
        low: Canny lower hysteresis threshold.
        high: Canny upper hysteresis threshold.
        mask: Optional region mask (see `_masked_values`); restricts the count to that region.

    Returns:
        Number of nonzero pixels in the (optionally masked) edge map.
    """
    edges = cv2.Canny(gray, low, high)
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    return int(np.count_nonzero(edges))


def mean_gradient_magnitude(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean Sobel gradient magnitude, as a second edge-strength signal alongside Canny counts.

    Args:
        gray: Single-channel image.
        mask: Optional region mask (see `_masked_values`); restricts the mean to that region.

    Returns:
        Mean of the per-pixel Sobel gradient magnitude. 0.0 if `mask` selects no pixels.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    selected = _masked_values(np.sqrt(grad_x**2 + grad_y**2), mask)
    return float(selected.mean()) if selected.size else 0.0


def near_saturated_fraction(
    gray: np.ndarray, threshold: int = 250, blur_sigma: float = 1.0, mask: Optional[np.ndarray] = None
) -> float:
    """Fraction of pixels at/above `threshold` after a slight blur (near-saturated / glare).

    The blur merges single hot pixels (sensor noise) into the surrounding value so only
    genuinely bright glare regions count, not isolated speckle.

    Args:
        gray: Single-channel image.
        threshold: Gray level (0-255) at/above which a pixel counts as near-saturated.
        blur_sigma: Gaussian blur sigma applied before thresholding.
        mask: Optional region mask (see `_masked_values`); restricts the fraction to that region.

    Returns:
        Fraction of pixels counted as near-saturated, in [0, 1]. 0.0 if `mask` selects no pixels.
    """
    blurred = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=blur_sigma)
    selected = _masked_values(blurred, mask)
    return float(np.count_nonzero(selected >= threshold)) / selected.size if selected.size else 0.0


def compute_darkening_map(
    gray_composite: np.ndarray, gray_reference: np.ndarray, blur_sigma: float = DARKENING_BLUR_SIGMA
) -> np.ndarray:
    """Per-pixel darkening the min-composite applied, relative to the reference.

    The app's min-composite can only ever pick a darker (or equal) pixel than the
    reference at any given location -- it never brightens. So `max(0, reference - composite)`
    (after a slight blur to suppress resample/JPEG noise) is the actual, unambiguous benefit
    signal: real glare healing, a broad matte-print glare sheen, or any other place a corner
    shot's darker pixel won, all show up here directly, without relying on either image ever
    reaching outright saturation (`near_saturated_fraction` misses matte glare that never
    reaches gray 250).

    Args:
        gray_composite: Composite image, single channel, working size.
        gray_reference: Reference image, single channel, same size as `gray_composite`.
        blur_sigma: Gaussian blur sigma applied to both images before differencing.

    Returns:
        float32 array (same shape as the inputs), 0.0 where the composite is equal to or
        brighter than the reference, positive where it's darker.
    """
    ref_blurred = cv2.GaussianBlur(gray_reference, ksize=(0, 0), sigmaX=blur_sigma).astype(np.float32)
    comp_blurred = cv2.GaussianBlur(gray_composite, ksize=(0, 0), sigmaX=blur_sigma).astype(np.float32)
    return np.maximum(0.0, ref_blurred - comp_blurred)


def parse_quad(value: str) -> np.ndarray:
    """Parse a `--quad` CLI value into 4 unit-coordinate points.

    Args:
        value: `"x1,y1 x2,y2 x3,y3 x4,y4"`, unit coordinates (0-1), clockwise from top-left.

    Returns:
        (4, 2) float array of unit coordinates.

    Raises:
        ValueError: If `value` doesn't parse to exactly 4 comma-separated coordinate pairs.
    """
    points = []
    for token in value.split():
        x_str, y_str = token.split(",")
        points.append((float(x_str), float(y_str)))
    if len(points) != 4:
        raise ValueError(f"--quad must have exactly 4 points, got {len(points)}: {value!r}")
    return np.array(points, dtype=np.float64)


def quad_to_mask(quad_unit: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize a unit-coordinate quad into a working-size binary mask.

    Args:
        quad_unit: (4, 2) unit coordinates (0-1), clockwise from top-left.
        width: Working-size image width.
        height: Working-size image height.

    Returns:
        uint8 mask (height, width), 255 inside the quad and 0 outside.
    """
    pixel_points = (quad_unit * np.array([width, height])).round().astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixel_points], (255,))
    return mask


@dataclass(frozen=True)
class PatchShift:
    """Local phase-correlation result for one grid patch.

    Attributes:
        row: Patch row index (0-based, top to bottom).
        col: Patch column index (0-based, left to right).
        x: Patch's left edge, in working-size pixels.
        y: Patch's top edge, in working-size pixels.
        shift_px: Estimated composite-vs-reference shift magnitude, in pixels.
            0.0 and `valid=False` when the patch was skipped as near-uniform.
        valid: Whether the patch had enough reference-side contrast for phase
            correlation to be trustworthy.
        healed: Whether this patch's mean darkening (see `compute_darkening_map`) exceeds
            `HEALED_PATCH_DARKENING_THRESHOLD` -- phase correlation reacting to a legitimate
            appearance change (glare healed away) rather than real misregistration. Always
            False as returned by `phase_correlation_grid` itself; set by callers that have
            a darkening map (see `score_stitch.compute_local_ghosting`).
    """

    row: int
    col: int
    x: int
    y: int
    shift_px: float
    valid: bool
    healed: bool = False


def phase_correlation_grid(
    gray_composite: np.ndarray, gray_reference: np.ndarray, patch_size: int = PATCH_SIZE_PX
) -> List[PatchShift]:
    """Estimate local misregistration via per-patch phase correlation.

    Divides the (common-size) images into a `patch_size` grid and computes the
    sub-pixel translational shift between each composite/reference patch pair with
    `cv2.phaseCorrelate`. Patches with low reference-side contrast (near-uniform:
    blank cardboard, sky, a glare wash) are marked invalid and excluded from shift
    statistics -- phase correlation is unreliable there and would just add noise.

    Args:
        gray_composite: Composite image, single channel, working size.
        gray_reference: Reference image, single channel, same size as `gray_composite`.
        patch_size: Patch side length in pixels.

    Returns:
        One `PatchShift` per grid cell, in row-major order. Trailing rows/columns
        that don't fill a whole patch are dropped.
    """
    height, width = gray_reference.shape[:2]
    n_rows, n_cols = height // patch_size, width // patch_size
    window: Optional[np.ndarray] = None
    results: List[PatchShift] = []

    for row in range(n_rows):
        for col in range(n_cols):
            y0, x0 = row * patch_size, col * patch_size
            ref_patch = gray_reference[y0 : y0 + patch_size, x0 : x0 + patch_size]
            comp_patch = gray_composite[y0 : y0 + patch_size, x0 : x0 + patch_size]

            if float(ref_patch.std()) < PATCH_MIN_STD:
                results.append(PatchShift(row, col, x0, y0, 0.0, False))
                continue

            if window is None:
                window = cv2.createHanningWindow((ref_patch.shape[1], ref_patch.shape[0]), cv2.CV_32F)
            shift, _response = cv2.phaseCorrelate(ref_patch.astype(np.float32), comp_patch.astype(np.float32), window)
            results.append(PatchShift(row, col, x0, y0, float(np.hypot(*shift)), True))

    return results


@dataclass(frozen=True)
class SpeckDetection:
    """Small bright specks detected in one grayscale image via a white top-hat filter.

    Attributes:
        mask: uint8 mask, same size as the source image, 255 on pixels belonging to a
            detected speck (after the area filter), 0 elsewhere.
        centroids: (N, 2) float array of (x, y) speck centroids, working-size pixel
            coordinates, one row per detected speck.
    """

    mask: np.ndarray
    centroids: np.ndarray

    @property
    def count(self) -> int:
        """Number of detected specks."""
        return int(self.centroids.shape[0])


def detect_bright_specks(
    gray: np.ndarray,
    kernel_px: int = SPECK_TOPHAT_KERNEL_PX,
    brightness_threshold: float = SPECK_BRIGHTNESS_THRESHOLD,
    min_area_px: int = SPECK_MIN_AREA_PX,
    max_area_px: int = SPECK_MAX_AREA_PX,
) -> SpeckDetection:
    """Detect small bright specks (e.g. stars on a dark background) via a white top-hat filter.

    A white top-hat (`src - opening(src, kernel)`) isolates small bright structures that are
    narrower than `kernel_px` from their locally darker surroundings -- exactly the profile of
    a fine bright detail like a star, and unlike a large, gradually-varying bright region
    (which the opening reconstructs and the subtraction cancels out). The result is
    thresholded and filtered by connected-component area so only genuinely small blobs count,
    not JPEG/resample texture noise (isolated single/double pixels) or larger bright features.

    Args:
        gray: Single-channel image, working size.
        kernel_px: Ellipse structuring-element side length in pixels.
        brightness_threshold: Top-hat response above which a pixel is speck material.
        min_area_px: Minimum connected-component area (pixels) to count as a speck.
        max_area_px: Maximum connected-component area (pixels) to count as a speck.

    Returns:
        The detected specks, as a `SpeckDetection`.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, brightness_threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(gray, dtype=np.uint8)
    kept_centroids = []
    for label in range(1, n_labels):  # label 0 is the background component
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area_px <= area <= max_area_px:
            mask[labels == label] = 255
            kept_centroids.append(centroids[label])

    centroids_array = np.array(kept_centroids, dtype=np.float64) if kept_centroids else np.empty((0, 2))
    return SpeckDetection(mask=mask, centroids=centroids_array)
