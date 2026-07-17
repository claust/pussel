"""Corner detection: polydp (primary) + curvature (cross-check).

Ported from ``network/experiments/exp28_piece_geometry/corner_detect.py`` —
keep algorithm changes in sync. Two adaptations for the backend:

1. The ``shitomasi`` detector is dropped (exp28's M2 bake-off found it fails
   structurally on tab bulbs; only ``polydp``/``curvature`` are used
   downstream).
2. ``scipy.ndimage.gaussian_filter1d`` and ``scipy.optimize.linear_sum_assignment``
   are unavailable in the backend. Curvature smoothing uses
   ``app.services.piece_geometry.contour.gaussian_filter1d_wrap`` (numerically
   equivalent numpy replacement). The corner-disagreement metric's 4x4
   optimal assignment is brute-forced over the 24 permutations of 4 elements
   instead of using the Hungarian algorithm — for n=4 this is an exact,
   not approximate, equivalent (`_optimal_assignment_max_distance`).

Both detectors share a candidate-pool -> best-4-subset -> refinement
pipeline built around puzzle structure: a true piece corner has locally
STRAIGHT contour on both sides (the edge shoulder regions), while a tab bulb
tip curves away immediately. Each candidate gets a "cornerness" score
(shoulder straightness on both sides x local angle near 90 degrees), the
best 4-subset is chosen by mean-cornerness x quadrilateral angle-score x
spacing-balance x normalized area, and the winning corners are refined by
intersecting the two fitted shoulder lines (which recovers the un-rounded
corner that `corner_radius` rounds off the actual contour).
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.services.piece_geometry.contour import gaussian_filter1d_wrap, resample_contour

# --- Shared config -----------------------------------------------------

# All detectors snap candidates to a contour resampled to this many points.
RESAMPLE_POINTS = 512

CURVATURE_K = 12
CURVATURE_SMOOTH_SIGMA = 2.0
CURVATURE_MAX_CANDIDATES = 16

POLYDP_EPSILON_FRACS = np.linspace(0.005, 0.05, 10)

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
# close to the candidate.
REFINE_W1 = 2
REFINE_W2 = 10
REFINE_MIN_ANGLE_DEG = 20.0
REFINE_MAX_SHIFT_FRAC = 0.03

# Corner cross-check: polydp vs curvature max matched-corner distance (as a
# fraction of the bbox diagonal) above which the piece is flagged.
CORNER_DISAGREEMENT_FRAC = 0.03

# All 24 permutations of 4 elements, used to brute-force the optimal
# (minimum total distance) assignment between two 4-corner sets.
_PERMUTATIONS_4 = list(itertools.permutations(range(4)))


class InsufficientCornersError(RuntimeError):
    """Raised when a detector cannot find enough candidate points to form a quadrilateral."""


@dataclass(frozen=True)
class CornerResult:
    """Four detected piece corners with per-corner confidence.

    Attributes:
        corners: 4x2 float array, ordered clockwise starting near the
            top-left corner.
        confidences: One confidence value per corner, in [0, 1], aligned with
            `corners`.
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
            score, in [0, 1]. Near 1 for true piece corners, near 0 for tab
            bulb tips.
        line_before: (centroid, away-pointing unit direction) of the line
            fitted to the shoulder window preceding the candidate.
        line_after: (centroid, away-pointing unit direction) of the line
            fitted to the shoulder window following the candidate.
        refine_line_before: Like `line_before` but fitted on the tight
            refinement window.
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


def bbox_diagonal(contour: np.ndarray) -> float:
    """Diagonal length of a contour's axis-aligned bounding box.

    Args:
        contour: Nx2 contour points.

    Returns:
        Euclidean length of the bbox diagonal.
    """
    extent = contour.max(axis=0) - contour.min(axis=0)
    return float(np.linalg.norm(extent))


def nearest_contour_index(contour: np.ndarray, point: Tuple[float, float]) -> int:
    """Find the index of the contour point nearest to `point`.

    Args:
        contour: Nx2 contour points.
        point: (x, y) query point.

    Returns:
        Index into `contour` of the nearest point.
    """
    dists = np.sum((contour - np.array(point)) ** 2, axis=1)
    return int(np.argmin(dists))


def _optimal_assignment_max_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Max matched-point distance under the minimum-total-distance assignment of two 4-point sets.

    Brute-forces the 24 permutations of 4 elements to find the assignment
    minimizing total distance, then returns the maximum distance under that
    (winning) assignment. For n=4 this is an exact equivalent of
    ``scipy.optimize.linear_sum_assignment(cdist(a, b)).max()`` — the
    Hungarian algorithm's guaranteed-optimal result coincides with the
    brute-force optimum, and 24 permutations is cheap enough to enumerate.

    Args:
        a: 4x2 point set.
        b: 4x2 point set.

    Returns:
        The maximum matched-pair distance under the optimal assignment.
    """
    best_sum = float("inf")
    best_max = 0.0
    for perm in _PERMUTATIONS_4:
        dists = np.linalg.norm(a - b[list(perm)], axis=1)
        total = float(dists.sum())
        if total < best_sum:
            best_sum = total
            best_max = float(dists.max())
    return best_max


def corner_max_distance_frac(a: np.ndarray, b: np.ndarray, diagonal: float) -> float:
    """Max matched-corner distance between two 4-corner sets, as a fraction of the diagonal.

    Args:
        a: 4x2 corner set.
        b: 4x2 corner set.
        diagonal: Piece bbox diagonal for normalization.

    Returns:
        `_optimal_assignment_max_distance(a, b)` divided by `diagonal`.
    """
    return _optimal_assignment_max_distance(a, b) / max(diagonal, 1e-8)


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
    spacing-balance x normalized area.

    Args:
        contour: Nx2 contour points the candidate indices refer into.
        candidate_indices: Indices into `contour` to consider as corners.
        arc_length: Cumulative closed-loop arc length at each contour point.
        cornerness: Per-candidate cornerness score.
        hull_area: Convex hull area of the full contour, for area normalization.

    Returns:
        Tuple of (best 4-tuple of indices, its score, its per-corner angle scores).

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

    Args:
        candidate_point: The candidate's raw contour point.
        geometry: The candidate's shoulder geometry.
        diagonal: Piece bbox diagonal, for the max-shift sanity check.

    Returns:
        The refined corner, or `candidate_point` unchanged when the shoulder
        lines are near-parallel or the intersection lands too far away.
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

    diagonal = bbox_diagonal(resampled)
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

    response = gaussian_filter1d_wrap(response, sigma=CURVATURE_SMOOTH_SIGMA)

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
    collecting every vertex candidate seen across the sweep, then runs the
    shared cornerness-based selection and shoulder-line refinement.

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
            candidates.add(nearest_contour_index(resampled, (float(pt[0]), float(pt[1]))))

    return _select_and_refine(resampled, sorted(candidates), "polydp")


def detect_corners_with_cross_check(contour: np.ndarray) -> Tuple[CornerResult, bool]:
    """Detect corners with polydp (primary) and curvature (cross-check).

    Args:
        contour: Nx2 contour points in original image coordinates.

    Returns:
        Tuple of (primary polydp `CornerResult`, corner_disagreement flag).
        The flag is True when the curvature cross-check disagrees with
        polydp by more than `CORNER_DISAGREEMENT_FRAC` of the bbox diagonal,
        or when the cross-check itself fails to find 4 corners.

    Raises:
        InsufficientCornersError: When the primary (polydp) detector fails.
    """
    primary = detect_corners_polydp(contour)

    disagreement = True
    try:
        cross_check = detect_corners_curvature(contour)
        diagonal = bbox_diagonal(contour)
        frac = corner_max_distance_frac(primary.corners, cross_check.corners, diagonal)
        disagreement = frac > CORNER_DISAGREEMENT_FRAC
    except InsufficientCornersError:
        pass  # keep disagreement=True: no cross-check available

    return primary, disagreement
