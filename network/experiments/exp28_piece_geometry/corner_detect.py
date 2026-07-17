"""M2: corner detection bake-off — three pure, rotation-agnostic detectors.

Each detector takes a full-density contour (Nx2, original image coordinates)
and returns a `CornerResult`: 4 corner points ordered clockwise starting near
the top-left, one confidence per corner, and the method name.

All three detectors share the same selection pipeline, built around puzzle
structure: a true piece corner has locally STRAIGHT contour on both sides
(the edge shoulder regions), while a tab bulb tip curves away immediately.
Each candidate gets a "cornerness" score (shoulder straightness on both
sides x local angle near 90 degrees), the best 4-subset is chosen by
mean-cornerness x quadrilateral angle-score x spacing-balance x normalized
area, and the winning corners are refined by intersecting the two fitted
shoulder lines (which recovers the un-rounded corner that `corner_radius`
rounds off the actual contour).
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from common import resample_contour
from scipy.ndimage import gaussian_filter1d

# --- Shared config -----------------------------------------------------

# All detectors snap candidates to a contour resampled to this many points.
RESAMPLE_POINTS = 512

CURVATURE_K = 12
CURVATURE_SMOOTH_SIGMA = 2.0
CURVATURE_MAX_CANDIDATES = 16

POLYDP_EPSILON_FRACS = np.linspace(0.005, 0.05, 10)

SHITOMASI_MAX_CORNERS = 20
SHITOMASI_QUALITY_LEVEL = 0.02

# Cap on the candidate pool entering the brute-force 4-subset search; when a
# detector proposes more, the top candidates by cornerness are kept.
MAX_SUBSET_CANDIDATES = 16

# Cornerness windows, in resampled-contour samples: skip the first W1 samples
# on each side of a candidate (~0.8% of perimeter, the corner-rounding zone),
# then fit a line to the following samples out to W2 (~4% of perimeter, the
# edge shoulder region).
CORNERNESS_W1 = 4
CORNERNESS_W2 = 24

# Straightness falloff: shoulder-fit RMS residual (as a fraction of the piece
# bbox diagonal) at which the straightness score drops to 1/e. Straight
# shoulders sit well under this; curved tab necks sit several times above it.
STRAIGHTNESS_FALLOFF = 0.004

# Corner refinement: the shoulder-line intersection replaces the raw contour
# point only when the lines are not near-parallel and the intersection stays
# close to the candidate. Refinement uses its own, much tighter window pair
# than the cornerness score: right next to the corner the contour tangents
# point at the un-rounded corner, whereas a wide fit picks up the edges'
# slight `corner_slope` curvature and lands off-corner.
REFINE_W1 = 2
REFINE_W2 = 10
REFINE_MIN_ANGLE_DEG = 20.0
REFINE_MAX_SHIFT_FRAC = 0.03


class InsufficientCornersError(RuntimeError):
    """Raised when a detector cannot find enough candidate points to form a quadrilateral."""


@dataclass(frozen=True)
class CornerResult:
    """Four detected piece corners with per-corner confidence.

    Attributes:
        corners: 4x2 float array, ordered clockwise starting near the
            top-left corner.
        confidences: One confidence value per corner, in [0, 1], aligned with
            `corners` (candidate cornerness x the winning quadrilateral's
            per-corner angle score).
        method: Name of the detector that produced this result.
    """

    corners: np.ndarray
    confidences: List[float]
    method: str


@dataclass(frozen=True)
class CandidateGeometry:
    """Local shoulder geometry at one corner candidate.

    Attributes:
        cornerness: Straightness-before x straightness-after x local angle
            score, in [0, 1]. Near 1 for true piece corners (straight
            shoulders meeting near 90 degrees), near 0 for tab bulb tips
            (curved neighborhoods on both sides).
        line_before: (centroid, away-pointing unit direction) of the line
            fitted to the shoulder window preceding the candidate
            (`CORNERNESS_W1..W2`, used for the cornerness score).
        line_after: (centroid, away-pointing unit direction) of the line
            fitted to the shoulder window following the candidate.
        refine_line_before: Like `line_before` but fitted on the tight
            `REFINE_W1..W2` window, used for corner refinement.
        refine_line_after: Like `line_after` but on the tight window.
    """

    cornerness: float
    line_before: Tuple[np.ndarray, np.ndarray]
    line_after: Tuple[np.ndarray, np.ndarray]
    refine_line_before: Tuple[np.ndarray, np.ndarray]
    refine_line_after: Tuple[np.ndarray, np.ndarray]


# --- Geometry helpers ----------------------------------------------------


def _cumulative_arc_length(contour: np.ndarray) -> np.ndarray:
    """Compute cumulative closed-loop arc length at each contour point.

    Args:
        contour: Nx2 contour points, implicitly closed.

    Returns:
        Length-N array; element i is the arc length from point 0 to point i.
    """
    closed = np.vstack([contour, contour[:1]])
    diffs = np.diff(closed, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])[:-1]


def _polygon_interior_angles(pts: np.ndarray) -> np.ndarray:
    """Compute the interior angle (degrees) at each vertex of a quadrilateral.

    Args:
        pts: 4x2 array of vertices in order around the polygon.

    Returns:
        Length-4 array of interior angles in degrees.
    """
    angles = np.zeros(4)
    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % 4]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles[i] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angles


def _quad_area(pts: np.ndarray) -> float:
    """Shoelace-formula area of a quadrilateral.

    Args:
        pts: 4x2 array of vertices in order around the polygon.

    Returns:
        Absolute area.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _fit_line_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit a line to 2D points via PCA.

    Args:
        points: Mx2 array of points.

    Returns:
        Tuple of (centroid, unit direction of the first principal component,
        RMS perpendicular residual).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    residuals = centered @ normal
    rms = float(np.sqrt(np.mean(residuals**2)))
    return centroid, direction, rms


def _intersect_lines(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """Intersect two lines given in point + direction form.

    Args:
        c1: A point on the first line.
        d1: Unit direction of the first line.
        c2: A point on the second line.
        d2: Unit direction of the second line.

    Returns:
        The intersection point, or None when the lines are near-parallel.
    """
    matrix = np.column_stack([d1, -d2])
    det = float(np.linalg.det(matrix))
    if abs(det) < 1e-9:
        return None
    params = np.linalg.solve(matrix, c2 - c1)
    return c1 + params[0] * d1


def _bbox_diagonal(contour: np.ndarray) -> float:
    """Diagonal length of a contour's axis-aligned bounding box.

    Args:
        contour: Nx2 contour points.

    Returns:
        Euclidean length of the bbox diagonal.
    """
    extent = contour.max(axis=0) - contour.min(axis=0)
    return float(np.linalg.norm(extent))


def _nearest_contour_index(contour: np.ndarray, point: Tuple[float, float]) -> int:
    """Find the index of the contour point nearest to `point`.

    Args:
        contour: Nx2 contour points.
        point: (x, y) query point.

    Returns:
        Index into `contour` of the nearest point.
    """
    dists = np.sum((contour - np.array(point)) ** 2, axis=1)
    return int(np.argmin(dists))


# --- Cornerness ------------------------------------------------------------


def candidate_geometry(resampled: np.ndarray, index: int, diagonal: float) -> CandidateGeometry:
    """Score one candidate's local shoulder geometry.

    Fits a line to the contour window on each side of the candidate (skipping
    the first `CORNERNESS_W1` samples for corner rounding, using the samples
    out to `CORNERNESS_W2`). Straightness of each fit and the angle between
    the two fitted lines combine into the cornerness score: a true piece
    corner has two straight shoulders meeting near 90 degrees, while a tab
    bulb tip curves away immediately on both sides.

    Args:
        resampled: Arc-length-resampled contour (`RESAMPLE_POINTS` x 2).
        index: Candidate index into `resampled`.
        diagonal: Piece bbox diagonal, for scale-normalizing fit residuals.

    Returns:
        The computed `CandidateGeometry`.
    """
    n = len(resampled)
    idx_before = [(index - j) % n for j in range(CORNERNESS_W1, CORNERNESS_W2 + 1)]
    idx_after = [(index + j) % n for j in range(CORNERNESS_W1, CORNERNESS_W2 + 1)]
    pts_before = resampled[idx_before]
    pts_after = resampled[idx_after]

    centroid_b, dir_b, rms_b = _fit_line_pca(pts_before)
    centroid_a, dir_a, rms_a = _fit_line_pca(pts_after)

    scale = max(diagonal, 1e-8)
    straightness_before = math.exp(-(rms_b / scale) / STRAIGHTNESS_FALLOFF)
    straightness_after = math.exp(-(rms_a / scale) / STRAIGHTNESS_FALLOFF)

    # Orient each direction away from the corner (from the window's near end
    # toward its far end) so their angle approximates the interior angle.
    if np.dot(dir_b, pts_before[-1] - pts_before[0]) < 0:
        dir_b = -dir_b
    if np.dot(dir_a, pts_after[-1] - pts_after[0]) < 0:
        dir_a = -dir_a

    cos_angle = float(np.clip(np.dot(dir_b, dir_a), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_angle))
    local_angle_score = max(0.0, 1.0 - abs(angle - 90.0) / 45.0)

    cornerness = straightness_before * straightness_after * local_angle_score

    # Tight-window fits for refinement: right next to the corner the contour
    # tangents point at the un-rounded corner.
    ref_before = [(index - j) % n for j in range(REFINE_W1, REFINE_W2 + 1)]
    ref_after = [(index + j) % n for j in range(REFINE_W1, REFINE_W2 + 1)]
    ref_centroid_b, ref_dir_b, _ = _fit_line_pca(resampled[ref_before])
    ref_centroid_a, ref_dir_a, _ = _fit_line_pca(resampled[ref_after])

    return CandidateGeometry(
        cornerness=cornerness,
        line_before=(centroid_b, dir_b),
        line_after=(centroid_a, dir_a),
        refine_line_before=(ref_centroid_b, ref_dir_b),
        refine_line_after=(ref_centroid_a, ref_dir_a),
    )


# --- Shared 4-subset scorer and refinement --------------------------------


def best_four_subset(
    contour: np.ndarray,
    candidate_indices: Sequence[int],
    arc_length: np.ndarray,
    cornerness: Dict[int, float],
    hull_area: float,
) -> Tuple[Tuple[int, int, int, int], float, np.ndarray]:
    """Brute-force the best 4-corner subset of a candidate index set.

    Scores each combination by mean-cornerness x quadrilateral angle-score x
    spacing-balance x normalized area: cornerness rejects tab bulb tips
    (which have curved shoulders), angle-score rewards interior angles near
    90 degrees, spacing-balance (ratio of the smallest to largest arc-length
    gap between consecutive corners) penalizes lopsided corner placement, and
    the normalized area (quad area / contour convex hull area, in [0, 1])
    suppresses degenerate small quads without dominating the other 0-1 terms.

    Args:
        contour: Nx2 contour points the candidate indices refer into.
        candidate_indices: Indices into `contour` to consider as corners.
        arc_length: Cumulative closed-loop arc length at each contour point
            (see `_cumulative_arc_length`), used for spacing-balance.
        cornerness: Per-candidate cornerness score (see `candidate_geometry`).
        hull_area: Convex hull area of the full contour, for area normalization.

    Returns:
        Tuple of (best 4-tuple of indices, its score, its per-corner angle
        scores).

    Raises:
        InsufficientCornersError: When fewer than 4 unique candidates are given.
    """
    unique = sorted(set(candidate_indices))
    if len(unique) < 4:
        raise InsufficientCornersError(f"Need >=4 candidate corners, got {len(unique)}")

    perimeter = arc_length[-1] + np.linalg.norm(contour[-1] - contour[0])
    hull_area = max(hull_area, 1e-8)
    best_combo: Tuple[int, int, int, int] = tuple(unique[:4])  # type: ignore[assignment]
    best_score = -1.0
    best_corner_scores = np.zeros(4)

    for combo in itertools.combinations(unique, 4):
        pts = contour[list(combo)]
        area_normalized = _quad_area(pts) / hull_area
        if area_normalized <= 0:
            continue
        angles = _polygon_interior_angles(pts)
        corner_scores = np.clip(1.0 - np.abs(angles - 90.0) / 45.0, 0.0, None)
        angle_score = float(corner_scores.mean())

        positions = np.sort(arc_length[list(combo)])
        gaps = np.diff(np.concatenate([positions, positions[:1] + perimeter]))
        spacing_balance = float(gaps.min() / gaps.max()) if gaps.max() > 0 else 0.0

        mean_cornerness = float(np.mean([cornerness[i] for i in combo]))
        score = mean_cornerness * angle_score * spacing_balance * area_normalized
        if score > best_score:
            best_score = score
            best_combo = combo  # type: ignore[assignment]
            best_corner_scores = corner_scores

    return best_combo, best_score, best_corner_scores


def _order_clockwise(corners: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """Order 4 corners clockwise (image coordinates, y-down) starting near the top-left.

    Args:
        corners: 4x2 array of corner points.
        confidences: Length-4 array of per-corner confidences, aligned with `corners`.

    Returns:
        Tuple of (reordered 4x2 corners, reordered confidence list).
    """
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    order = np.argsort(angles)
    ordered = corners[order]
    ordered_conf = confidences[order]

    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    ordered_conf = np.roll(ordered_conf, -start, axis=0)
    return ordered, ordered_conf.tolist()


def _refine_corner(candidate_point: np.ndarray, geometry: CandidateGeometry, diagonal: float) -> np.ndarray:
    """Refine a corner by intersecting its two fitted shoulder lines.

    Ground truth is the un-rounded base-square corner while `corner_radius`
    rounds the actual contour, so the intersection of the two straight
    shoulder lines lands closer to the true corner than any contour point.

    Args:
        candidate_point: The candidate's raw contour point.
        geometry: The candidate's shoulder geometry.
        diagonal: Piece bbox diagonal, for the max-shift sanity check.

    Returns:
        The refined corner, or `candidate_point` unchanged when the shoulder
        lines are near-parallel (angle < `REFINE_MIN_ANGLE_DEG`) or the
        intersection lands more than `REFINE_MAX_SHIFT_FRAC` of the diagonal
        away from the candidate.
    """
    centroid_b, dir_b = geometry.refine_line_before
    centroid_a, dir_a = geometry.refine_line_after

    cos_angle = float(np.clip(abs(np.dot(dir_b, dir_a)), 0.0, 1.0))
    angle_between = math.degrees(math.acos(cos_angle))
    if angle_between < REFINE_MIN_ANGLE_DEG:
        return candidate_point

    intersection = _intersect_lines(centroid_b, dir_b, centroid_a, dir_a)
    if intersection is None:
        return candidate_point
    if float(np.linalg.norm(intersection - candidate_point)) > REFINE_MAX_SHIFT_FRAC * diagonal:
        return candidate_point
    return intersection


def _select_and_refine(resampled: np.ndarray, candidate_indices: Sequence[int], method: str) -> CornerResult:
    """Shared back half of every detector: score candidates, pick 4, refine, order.

    When the detector proposes more than `MAX_SUBSET_CANDIDATES` candidates,
    the pool is capped to the top candidates by cornerness — the corner prior
    itself, rather than any detector-specific persistence ranking (which
    tends to favor tab bulb tips over shallow true corners).

    Args:
        resampled: Arc-length-resampled contour (`RESAMPLE_POINTS` x 2).
        candidate_indices: Candidate indices into `resampled`.
        method: Detector name for the result.

    Returns:
        The final `CornerResult`.

    Raises:
        InsufficientCornersError: When fewer than 4 unique candidates are given.
    """
    unique = sorted(set(candidate_indices))
    if len(unique) < 4:
        raise InsufficientCornersError(f"Need >=4 candidate corners, got {len(unique)}")

    diagonal = _bbox_diagonal(resampled)
    geometries = {i: candidate_geometry(resampled, i, diagonal) for i in unique}
    cornerness = {i: g.cornerness for i, g in geometries.items()}
    if len(unique) > MAX_SUBSET_CANDIDATES:
        unique = sorted(sorted(unique, key=lambda i: -cornerness[i])[:MAX_SUBSET_CANDIDATES])
    arc_length = _cumulative_arc_length(resampled)
    hull_area = float(cv2.contourArea(cv2.convexHull(resampled.astype(np.float32))))

    combo, _, quad_corner_scores = best_four_subset(resampled, unique, arc_length, cornerness, hull_area)

    refined = np.zeros((4, 2))
    confidences = np.zeros(4)
    for j, idx in enumerate(combo):
        geometry = geometries[idx]
        refined[j] = _refine_corner(resampled[idx], geometry, diagonal)
        confidences[j] = float(np.clip(geometry.cornerness * quad_corner_scores[j], 0.0, 1.0))

    ordered, ordered_conf = _order_clockwise(refined, confidences)
    return CornerResult(corners=ordered, confidences=ordered_conf, method=method)


# --- Detectors ------------------------------------------------------------


def detect_corners_curvature(contour: np.ndarray) -> CornerResult:
    """Detect corners from local curvature (turn angle) along the resampled contour.

    Resamples the contour to a fixed point count, scores each point by how
    much the path turns over a +-k window, smooths that response, picks
    well-separated local maxima via non-max suppression, then runs the shared
    cornerness-based selection and shoulder-line refinement.

    Args:
        contour: Nx2 contour points in original image coordinates.

    Returns:
        The detected `CornerResult`.

    Raises:
        InsufficientCornersError: When fewer than 4 corner candidates survive.
    """
    resampled = resample_contour(contour, RESAMPLE_POINTS)
    n = len(resampled)
    k = CURVATURE_K

    response = np.zeros(n)
    for i in range(n):
        p_prev = resampled[(i - k) % n]
        p_curr = resampled[i]
        p_next = resampled[(i + k) % n]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        response[i] = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    response = gaussian_filter1d(response, sigma=CURVATURE_SMOOTH_SIGMA, mode="wrap")

    # ~3% of the perimeter: tight enough that a true corner is not suppressed
    # by a nearby higher-curvature tab armpit, wide enough to dedupe peaks.
    min_separation = n // 32
    order = np.argsort(-response)
    selected: List[int] = []
    for idx in order:
        if all(min(abs(idx - s), n - abs(idx - s)) >= min_separation for s in selected):
            selected.append(int(idx))
        if len(selected) >= CURVATURE_MAX_CANDIDATES:
            break

    if len(selected) < 4:
        selected = [int(i) for i in order[:CURVATURE_MAX_CANDIDATES]]

    return _select_and_refine(resampled, selected, "curvature")


def detect_corners_polydp(contour: np.ndarray) -> CornerResult:
    """Detect corners via an `approxPolyDP` epsilon sweep.

    Sweeps the Douglas-Peucker epsilon from 0.5% to 5% of the perimeter,
    collecting every vertex candidate seen across the sweep (snapped to the
    resampled contour and deduplicated), then runs the shared
    cornerness-based selection (which caps the pool by cornerness) and
    shoulder-line refinement.

    Args:
        contour: Nx2 contour points in original image coordinates.

    Returns:
        The detected `CornerResult`.

    Raises:
        InsufficientCornersError: When fewer than 4 corner candidates survive.
    """
    resampled = resample_contour(contour, RESAMPLE_POINTS)
    contour_cv = contour.astype(np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour_cv, closed=True)

    candidates: set[int] = set()
    for eps_frac in POLYDP_EPSILON_FRACS:
        epsilon = eps_frac * perimeter
        approx = cv2.approxPolyDP(contour_cv, epsilon, closed=True).reshape(-1, 2)
        for pt in approx:
            candidates.add(_nearest_contour_index(resampled, (float(pt[0]), float(pt[1]))))

    return _select_and_refine(resampled, sorted(candidates), "polydp")


def detect_corners_shitomasi(contour: np.ndarray, crop_shape: Tuple[int, int]) -> CornerResult:
    """Detect corners via Shi-Tomasi ("good features to track") on the filled piece mask.

    Args:
        contour: Nx2 contour points in original image coordinates.
        crop_shape: (height, width) of the crop the contour lives in, used to
            render the filled mask for `cv2.goodFeaturesToTrack`.

    Returns:
        The detected `CornerResult`.

    Raises:
        InsufficientCornersError: When fewer than 4 corner candidates survive.
    """
    resampled = resample_contour(contour, RESAMPLE_POINTS)
    height, width = crop_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], (255,))

    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    piece_size = max(w, h)
    # /16 (not /8): with a wider suppression radius, sharp tab armpits knock
    # out the rounded (weaker-response) true corners next to them.
    min_distance = max(1.0, piece_size / 16)

    detected = cv2.goodFeaturesToTrack(
        mask,
        maxCorners=SHITOMASI_MAX_CORNERS,
        qualityLevel=SHITOMASI_QUALITY_LEVEL,
        minDistance=min_distance,
    )

    candidates: List[int] = []
    if detected is not None:
        for pt in detected.reshape(-1, 2):
            idx = _nearest_contour_index(resampled, (float(pt[0]), float(pt[1])))
            if idx not in candidates:
                candidates.append(idx)

    return _select_and_refine(resampled, candidates, "shitomasi")
