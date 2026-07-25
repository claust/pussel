"""Geometry primitives for measuring how well a puzzle photo gets rectified.

The shipped path (see `docs/puzzle-image-rectification.html`) detects the
puzzle's quadrilateral with `app.services.puzzle_detector.PuzzleFrameDetector`
and warps it onto an axis-aligned rectangle whose size comes from the quad's
own edge lengths. This module supplies the measurements that say how close
that output is to the physical rectangle the camera was pointed at:

* `detect_with_path` — runs the real detector and additionally reports which
  of its three layers produced the answer.
* `rectangle_aspect` — the true width/height of a photographed rectangle,
  recovered from its image quad alone (Zhang & He's whiteboard-rectification
  formulation), together with the focal length it implies.
* `plane_tilt_degrees` — how oblique the shot was, so an aspect error can be
  read against the tilt that caused it.
* `residual_skew_degrees` — a ground-truth-free proxy: how far the straight
  lines *inside* a warped crop still sit off the horizontal/vertical axes.
* `quad_iou` / `corner_errors` — agreement with a hand-labelled quad.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

# Corners covering the whole image — what the detector falls back to, mirrored here so
# `detect_with_path` can report the same thing without importing the backend.
FULL_IMAGE_CORNERS: List[Point] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

# Long-side size (pixels) corner errors are reported at, so they are comparable across
# dumps captured at different resolutions. Matches scripts/stitch_quality's convention.
REPORTING_LONG_SIDE = 2048

# Horizontal field of view (degrees) assumed for the fallback focal length when the quad
# is too close to a parallelogram for `rectangle_aspect` to solve for one. iPhone main
# camera, 26 mm full-frame equivalent: 2 * atan(36 / (2 * 26)).
DEFAULT_HORIZONTAL_FOV_DEG = 69.4

# |k - 1| below which the projection is treated as parallel (a parallelogram image), where
# the focal length drops out of the aspect solution entirely.
PARALLEL_PROJECTION_EPS = 1e-3

# Plausible focal lengths, as multiples of the image width. The orthogonality constraint
# that solves for `f` is ill-conditioned as a shot approaches square-on — the vanishing
# points run off toward infinity — and it degrades by returning an enormous focal rather
# than by failing, which then reads back as a large, entirely fictional tilt. Real phone
# cameras span roughly 25°-70° horizontal FOV, i.e. 0.7x to 2.4x the image width; an
# estimate outside that came from the ill-conditioned regime and is discarded in favor of
# the assumed FOV. (Observed on real captures: 4170, 5102 and 61823 px on 1536-wide
# frames whose true focal is ~1100.)
MIN_PLAUSIBLE_FOCAL_FACTOR = 0.7
MAX_PLAUSIBLE_FOCAL_FACTOR = 2.4

# Hough parameters for `residual_skew_degrees`. The structure of interest is the puzzle's
# own long straight edges, so lines must span a fair fraction of the crop. The vote
# threshold is derived from that same length rather than fixed, so small crops (a tight
# quad on a 2048px composite) don't silently return no lines at all.
SKEW_MIN_LINE_FRACTION = 0.15
SKEW_MAX_LINE_GAP_FRACTION = 0.01
SKEW_HOUGH_THRESHOLD_FRACTION = 0.5
# Lines farther than this from the nearest axis are scene content (artwork diagonals),
# not a mis-rectified border, and are excluded from the skew estimate.
SKEW_AXIS_WINDOW_DEG = 20.0
# Segments this close to the crop's own boundary are excluded: a warp fills the output
# rectangle edge to edge, so Canny fires along the image border itself, and those
# segments are exactly axis-aligned by construction. Counting them would drag every
# measurement toward 0 and make a badly rectified crop look perfect.
SKEW_BORDER_MARGIN_FRACTION = 0.02


def order_corners(points: Sequence[Point]) -> List[Point]:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Mirrors `puzzle_detector._order_corners` so measurements and the shipped
    warp agree on which corner is which.

    Args:
        points: Four (x, y) points in any order.

    Returns:
        The same four points ordered TL, TR, BR, BL.
    """
    array = np.asarray(points, dtype=np.float64)
    sums = array.sum(axis=1)
    diffs = array[:, 1] - array[:, 0]
    return [
        (float(array[int(np.argmin(sums))][0]), float(array[int(np.argmin(sums))][1])),
        (float(array[int(np.argmin(diffs))][0]), float(array[int(np.argmin(diffs))][1])),
        (float(array[int(np.argmax(sums))][0]), float(array[int(np.argmax(sums))][1])),
        (float(array[int(np.argmax(diffs))][0]), float(array[int(np.argmax(diffs))][1])),
    ]


def detect_with_path(detector: object, image: "np.ndarray") -> Tuple[List[Point], float, str, List[object]]:
    """Run the real detector on an RGB image and report which generator won.

    Which of the four generators produced the winning quad is the single most
    useful diagnostic when a detection goes wrong — they fail in completely
    different ways — and the runners-up say how close the call was.

    Args:
        detector: A `PuzzleFrameDetector` instance.
        image: RGB uint8 image (already EXIF-corrected).

    Returns:
        A tuple of (corners, confidence, source, candidates), where corners
        are normalized [0, 1] TL/TR/BR/BL, source is one of "mask",
        "grabcut", "edge", "lines" or "none", and candidates is the full
        scored list, best first.
    """
    from PIL import Image as PILImage

    candidates = detector.detect_candidates(PILImage.fromarray(image))  # type: ignore[attr-defined]
    if not candidates:
        return list(FULL_IMAGE_CORNERS), 0.0, "none", []
    best = candidates[0]
    return list(best.corners), float(best.confidence), str(best.source), list(candidates)


def _cross(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    return np.cross(a, b)


def rectangle_aspect(
    corners: Sequence[Point],
    image_size: Tuple[int, int],
    assumed_focal_px: Optional[float] = None,
    known_focal_px: Optional[float] = None,
) -> Dict[str, object]:
    """Recover a photographed rectangle's true width/height from its image quad.

    Implements Zhang & He's whiteboard-rectification solution. A rectangle's
    four image corners constrain both the camera's focal length and the
    rectangle's aspect ratio: the two pairs of opposite edges give two
    vanishing points, whose directions must be orthogonal in space, which is
    one equation in the focal length; with the focal known the aspect follows
    from the two edge directions' lengths under the recovered calibration.

    This is what makes "the crop's aspect ratio is wrong" measurable at all.
    The shipped `warp` sizes its output from the quad's own edge lengths,
    which under perspective are foreshortened by different amounts — the
    error this function quantifies.

    Args:
        corners: Four (x, y) pixel corners, ordered TL, TR, BR, BL.
        image_size: (width, height) of the image the corners live in, used for
            the principal point (assumed centered).
        assumed_focal_px: Focal length to fall back on when the quad cannot
            solve for one — a tilt about a single axis leaves the horizontal
            edges parallel, so there is only one finite vanishing point and
            nothing to pin `f` with. Defaults to `DEFAULT_HORIZONTAL_FOV_DEG`
            on this image's width, which on a real device is off by enough to
            move the aspect several percent (see the self-tests) — the reason
            capturing the camera's real intrinsics is worth doing.
        known_focal_px: A focal length that is actually known (from the
            capture's intrinsics). Skips estimation entirely.

    Returns:
        A dict with "aspect" (width/height, None when the quad is degenerate),
        "focal_px", and "focal_source" ("known", "estimated", "assumed", "implausible",
        "parallel" or "degenerate").
    """
    width, height = image_size
    u0, v0 = width / 2.0, height / 2.0
    fallback = (
        assumed_focal_px
        if assumed_focal_px
        else (width / 2.0) / math.tan(math.radians(DEFAULT_HORIZONTAL_FOV_DEG) / 2.0)
    )

    top_left, top_right, bottom_right, bottom_left = (np.array([x, y, 1.0]) for x, y in corners)
    # Zhang's naming: m1 = TL, m2 = TR, m3 = BL, m4 = BR.
    m1, m2, m3, m4 = top_left, top_right, bottom_left, bottom_right

    denominator_2 = float(np.dot(_cross(m2, m4), m3))
    denominator_3 = float(np.dot(_cross(m3, m4), m2))
    if abs(denominator_2) < 1e-12 or abs(denominator_3) < 1e-12:
        return {"aspect": None, "focal_px": fallback, "focal_source": "degenerate"}
    k2 = float(np.dot(_cross(m1, m4), m3)) / denominator_2
    k3 = float(np.dot(_cross(m1, m4), m2)) / denominator_3

    if abs(k2 - 1.0) < PARALLEL_PROJECTION_EPS and abs(k3 - 1.0) < PARALLEL_PROJECTION_EPS:
        # Parallel projection: the quad is a parallelogram and the focal length drops out.
        across = np.linalg.norm(m2[:2] - m1[:2])
        down = np.linalg.norm(m3[:2] - m1[:2])
        aspect = float(across / down) if down > 0 else None
        return {"aspect": aspect, "focal_px": fallback, "focal_source": "parallel"}

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1
    n21, n22, n23 = (float(value) for value in n2)
    n31, n32, n33 = (float(value) for value in n3)

    focal_source = "estimated"
    focal_squared = None
    if known_focal_px is not None:
        focal_squared = known_focal_px * known_focal_px
        focal_source = "known"
    elif abs(n23 * n33) > 1e-12:
        focal_squared = -(
            (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0)
            + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0)
        ) / (n23 * n33)
    if focal_squared is None or focal_squared <= 0:
        focal = fallback
        focal_source = "assumed"
    else:
        focal = math.sqrt(focal_squared)
    if focal_source == "estimated" and not (
        MIN_PLAUSIBLE_FOCAL_FACTOR * width <= focal <= MAX_PLAUSIBLE_FOCAL_FACTOR * width
    ):
        focal = fallback
        focal_source = "implausible"

    calibration_inverse = np.array([[1.0 / focal, 0.0, -u0 / focal], [0.0, 1.0 / focal, -v0 / focal], [0.0, 0.0, 1.0]])
    across = calibration_inverse @ n2
    down = calibration_inverse @ n3
    down_norm = float(np.dot(down, down))
    if down_norm <= 0:
        return {"aspect": None, "focal_px": focal, "focal_source": focal_source}
    aspect = math.sqrt(float(np.dot(across, across)) / down_norm)
    return {"aspect": float(aspect), "focal_px": float(focal), "focal_source": focal_source}


def plane_tilt_degrees(corners: Sequence[Point], aspect: float, focal_px: float, image_size: Tuple[int, int]) -> float:
    """Angle between the camera's optical axis and the puzzle plane's normal.

    Decomposes the plane-to-image homography with the given calibration: the
    first two columns of `K⁻¹H` are the plane's in-image axes, and their cross
    product is its normal in camera coordinates. 0° is a perfectly square-on
    shot; the tilt is the context an aspect or skew error should be read
    against.

    Args:
        corners: Four (x, y) pixel corners, ordered TL, TR, BR, BL.
        aspect: The rectangle's true width/height (from `rectangle_aspect`).
        focal_px: Focal length in pixels.
        image_size: (width, height), for the principal point.

    Returns:
        Tilt in degrees, in [0, 90].
    """
    width, height = image_size
    source = np.array([(0.0, 0.0), (aspect, 0.0), (aspect, 1.0), (0.0, 1.0)], dtype=np.float32)
    destination = np.array(corners, dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)
    calibration_inverse = np.array(
        [
            [1.0 / focal_px, 0.0, -(width / 2.0) / focal_px],
            [0.0, 1.0 / focal_px, -(height / 2.0) / focal_px],
            [0.0, 0.0, 1.0],
        ]
    )
    normalized = calibration_inverse @ homography
    first = normalized[:, 0]
    second = normalized[:, 1]
    scale = np.linalg.norm(first)
    if scale <= 0:
        return float("nan")
    first = first / scale
    second = second / np.linalg.norm(second) if np.linalg.norm(second) > 0 else second
    normal = np.cross(first, second)
    norm = np.linalg.norm(normal)
    if norm <= 0:
        return float("nan")
    cosine = abs(float(normal[2] / norm))
    return float(math.degrees(math.acos(min(1.0, cosine))))


def residual_skew_degrees(image: "np.ndarray") -> Dict[str, object]:
    """Measure how far a rectified crop's straight lines still lie off-axis.

    The ground-truth-free counterpart to the aspect metrics, and the one that
    corresponds to what the user sees on the confirm screen: when the detector
    picks the wrong quad, the puzzle's real borders end up *inside* the crop at
    an angle, and this is that angle. On a correctly rectified crop the box's
    edges coincide with the crop's own borders and read ~0°.

    Long straight segments are found with a probabilistic Hough transform,
    each is measured against the nearer axis, and two classes are dropped:
    segments more than `SKEW_AXIS_WINDOW_DEG` off both axes (artwork content,
    not structure), and segments hugging the crop's own boundary (which a warp
    makes axis-aligned by construction — see `SKEW_BORDER_MARGIN_FRACTION`).
    The remaining deviations are combined length-weighted, so a puzzle's long
    border outweighs incidental short edges.

    Args:
        image: The warped crop, RGB or grayscale uint8.

    Returns:
        A dict with "skew_deg" (None when too little straight structure was
        found), "n_lines" and "total_line_length_px".
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    short_side = min(width, height)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    edges = cv2.Canny(blurred, int(max(0, 0.67 * median)), int(min(255, 1.33 * median)))
    minimum_length = int(SKEW_MIN_LINE_FRACTION * short_side)
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(20, int(SKEW_HOUGH_THRESHOLD_FRACTION * minimum_length)),
        minLineLength=minimum_length,
        maxLineGap=int(max(1, SKEW_MAX_LINE_GAP_FRACTION * short_side)),
    )
    if segments is None:
        return {"skew_deg": None, "n_lines": 0, "total_line_length_px": 0.0}

    margin = SKEW_BORDER_MARGIN_FRACTION * short_side
    deviations: List[float] = []
    weights: List[float] = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        length = float(math.hypot(x2 - x1, y2 - y1))
        angle = math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))) % 180.0
        deviation = min(abs(angle), abs(angle - 90.0), abs(angle - 180.0))
        if deviation > SKEW_AXIS_WINDOW_DEG:
            continue
        on_vertical_border = max(x1, x2) < margin or min(x1, x2) > width - margin
        on_horizontal_border = max(y1, y2) < margin or min(y1, y2) > height - margin
        if on_vertical_border or on_horizontal_border:
            continue
        deviations.append(deviation)
        weights.append(length)
    if not deviations:
        return {"skew_deg": None, "n_lines": 0, "total_line_length_px": 0.0}

    order = np.argsort(np.asarray(deviations))
    sorted_deviations = np.asarray(deviations)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    half = cumulative[-1] / 2.0
    skew = float(sorted_deviations[int(np.searchsorted(cumulative, half))])
    return {"skew_deg": skew, "n_lines": len(deviations), "total_line_length_px": float(cumulative[-1])}


def snap_quad_to_edges(
    image: "np.ndarray", corners: Sequence[Point], band_fraction: float = 0.012, samples: int = 60
) -> List[Point]:
    """Refine a roughly-placed quad onto the strong image edges near it.

    Hand-clicked (or eyeballed) corners land within a percent or so of the
    truth, which is not accurate enough to measure a detector that is itself
    wrong by a few percent. This snaps each of the four sides onto the real
    boundary: it samples points along the side, walks perpendicular to it
    looking for the strongest intensity gradient within a narrow band, fits a
    line through the hits by least squares, and intersects adjacent fitted
    lines for the refined corners.

    Seeded by a human and confined to a narrow band, so it sharpens a label
    rather than inventing one — it cannot wander off to a different structure
    the way a detector can.

    Args:
        image: RGB or grayscale uint8 image.
        corners: Four rough (x, y) pixel corners ordered TL, TR, BR, BL.
        band_fraction: Half-width of the perpendicular search band, as a
            fraction of the image's long side.
        samples: Points sampled along each side.

    Returns:
        The four refined (x, y) pixel corners, in the same order. Sides with
        too few gradient hits keep their seeded position.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = gray.shape[:2]
    band = max(3, int(band_fraction * max(width, height)))

    lines: List[Optional[Tuple[float, float, float]]] = []
    for index in range(4):
        start = np.array(corners[index], dtype=np.float64)
        end = np.array(corners[(index + 1) % 4], dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1:
            lines.append(None)
            continue
        direction /= length
        normal = np.array([-direction[1], direction[0]])
        hits: List[Tuple[float, float]] = []
        for step in np.linspace(0.08, 0.92, samples):
            base = start + direction * (length * step)
            offsets = np.arange(-band, band + 1, dtype=np.float64)
            points = base[None, :] + normal[None, :] * offsets[:, None]
            xs = np.clip(points[:, 0], 0, width - 1).astype(np.int32)
            ys = np.clip(points[:, 1], 0, height - 1).astype(np.int32)
            profile = gray[ys, xs].astype(np.float64)
            if profile.size < 3:
                continue
            gradient = np.abs(np.gradient(profile))
            best = int(np.argmax(gradient))
            if gradient[best] < 4.0:
                continue
            hits.append((float(xs[best]), float(ys[best])))
        if len(hits) < samples // 4:
            lines.append(None)
            continue
        lines.append(_fit_line(np.array(hits)))

    refined: List[Point] = []
    for index in range(4):
        previous = lines[(index - 1) % 4]
        current = lines[index]
        intersection = _intersect(previous, current) if previous and current else None
        if intersection is None or not (0 <= intersection[0] < width and 0 <= intersection[1] < height):
            refined.append((float(corners[index][0]), float(corners[index][1])))
        else:
            refined.append(intersection)
    return refined


def _fit_line(points: "np.ndarray") -> Tuple[float, float, float]:
    """Total-least-squares line through points, as (a, b, c) with ax + by = c.

    Args:
        points: (N, 2) array of points.

    Returns:
        The line coefficients, normalized so (a, b) is a unit vector.
    """
    centroid = points.mean(axis=0)
    _, _, right = np.linalg.svd(points - centroid)
    normal = right[1]
    return float(normal[0]), float(normal[1]), float(np.dot(normal, centroid))


def _intersect(
    first: Optional[Tuple[float, float, float]], second: Optional[Tuple[float, float, float]]
) -> Optional[Point]:
    """Intersection of two lines in (a, b, c) form, or None when near-parallel.

    Args:
        first: The first line.
        second: The second line.

    Returns:
        The (x, y) intersection, or None.
    """
    if first is None or second is None:
        return None
    matrix = np.array([first[:2], second[:2]])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        return None
    solution = np.linalg.solve(matrix, np.array([first[2], second[2]]))
    return float(solution[0]), float(solution[1])


def quad_iou(first: Sequence[Point], second: Sequence[Point], size: Tuple[int, int] = (1000, 1000)) -> float:
    """Intersection-over-union of two quadrilaterals given in unit coordinates.

    Rasterized rather than computed analytically: the quads are small, the
    rasterization is exact enough at 1000², and it cannot trip over the
    degenerate self-intersecting quads a bad detection sometimes produces.

    Args:
        first: Four (x, y) points normalized to [0, 1].
        second: Four (x, y) points normalized to [0, 1].
        size: Rasterization resolution.

    Returns:
        IoU in [0, 1].
    """
    width, height = size
    masks = []
    for quad in (first, second):
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.array([(x * width, y * height) for x, y in quad], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        masks.append(mask > 0)
    union = int(np.count_nonzero(masks[0] | masks[1]))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(masks[0] & masks[1])) / union


def corner_errors(detected: Sequence[Point], truth: Sequence[Point], long_side: int = REPORTING_LONG_SIDE) -> Dict:
    """Per-corner distance between a detected quad and a hand-labelled one.

    Both quads are ordered TL/TR/BR/BL first, so the comparison is corner to
    corner rather than order-dependent.

    Args:
        detected: Four (x, y) points normalized to [0, 1].
        truth: Four (x, y) points normalized to [0, 1].
        long_side: Pixel scale the errors are reported at.

    Returns:
        A dict with "rms_px", "max_px" and "per_corner_px".
    """
    detected_ordered = np.array(order_corners(detected)) * long_side
    truth_ordered = np.array(order_corners(truth)) * long_side
    distances = np.linalg.norm(detected_ordered - truth_ordered, axis=1)
    return {
        "rms_px": float(np.sqrt(np.mean(distances**2))),
        "max_px": float(np.max(distances)),
        "per_corner_px": [float(value) for value in distances],
    }


def warp_like_backend(image: "np.ndarray", corners: Sequence[Point]) -> "np.ndarray":
    """Reproduce the shipped `PuzzleFrameDetector.warp` on an RGB array.

    Kept here (rather than calling the backend through PIL) so the scorer can
    warp with a *different* destination aspect ratio for comparison without
    touching production code.

    Args:
        image: RGB uint8 image.
        corners: Four (x, y) points normalized to [0, 1].

    Returns:
        The warped RGB crop.
    """
    height, width = image.shape[:2]
    pixels = [(x * width, y * height) for x, y in corners]
    ordered = order_corners(pixels)
    return warp_to_aspect(image, ordered, aspect=None)


def warp_to_aspect(image: "np.ndarray", pixel_corners: Sequence[Point], aspect: Optional[float]) -> "np.ndarray":
    """Warp a quad onto a rectangle, either backend-style or at a given aspect.

    Args:
        image: RGB uint8 image.
        pixel_corners: Four (x, y) pixel corners ordered TL, TR, BR, BL.
        aspect: Desired output width/height. None reproduces the backend's
            max-edge-length sizing.

    Returns:
        The warped RGB crop.
    """
    top_left, top_right, bottom_right, bottom_left = pixel_corners
    edge_width = max(math.dist(top_left, top_right), math.dist(bottom_left, bottom_right))
    edge_height = max(math.dist(top_left, bottom_left), math.dist(top_right, bottom_right))
    if aspect is None:
        destination_width = max(1, round(edge_width))
        destination_height = max(1, round(edge_height))
    else:
        # Preserve the quad's pixel scale: keep the longer measured edge and derive
        # the other from the true aspect, so no detail is resampled away.
        if aspect >= edge_width / max(edge_height, 1e-6):
            destination_width = max(1, round(edge_width))
            destination_height = max(1, round(edge_width / aspect))
        else:
            destination_height = max(1, round(edge_height))
            destination_width = max(1, round(edge_height * aspect))
    source = np.array(pixel_corners, dtype=np.float32)
    destination = np.array(
        [
            (0, 0),
            (destination_width - 1, 0),
            (destination_width - 1, destination_height - 1),
            (0, destination_height - 1),
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (destination_width, destination_height))
