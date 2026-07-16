"""Edge splitting, tab/blank/flat classification, and canonical edge frames.

Ported from ``network/experiments/exp28_piece_geometry/edge_split.py`` (split
+ classify) and ``edge_match.py`` (``canonicalize_edge``) — keep algorithm
changes in sync.

Deviation from exp28: production pieces are photographed at an UNKNOWN
orientation (exp28's north_star photos were all captured upright at
rotation 0, so ``edge_split.py`` could map arcs to a N/E/S/W grid frame).
This module keeps edges in CONTOUR TRAVERSAL ORDER instead — that dataset-
specific grid mapping does not apply to freely-oriented production captures.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.services.piece_geometry.contour import resample_contour, resample_polyline
from app.services.piece_geometry.corners import (
    CornerResult,
    InsufficientCornersError,
    detect_corners_with_cross_check,
    nearest_contour_index,
)

# Dense resampling used for edge splitting (finer than the detectors' 512).
SPLIT_RESAMPLE_POINTS = 1024

# Each edge arc is stored resampled to this many equidistant points.
EDGE_POLYLINE_POINTS = 100

# Tab/blank/flat threshold on |dominant deviation| / chord length. The
# distribution measured across all 3,664 clean exp28 edges is strongly
# bimodal: grid-truth flat edges cluster in [0, 0.04] (median 0.009) and
# tab/blank features in [0.11, 0.42] (median 0.292). 0.07 sits mid-gap.
FLAT_THRESHOLD = 0.07


@dataclass(frozen=True)
class Edge:
    """One classified edge arc, in contour traversal order.

    Attributes:
        index: The edge's position (0-3) in contour traversal order, starting
            at the corner nearest the top-left of the piece's bbox. Since
            production photos have unknown orientation, this is NOT a grid
            direction (N/E/S/W) — just a stable per-piece ordering.
        edge_type: "tab", "blank", or "flat".
        dominant_dev: Signed dominant deviation from the corner-to-corner
            chord, normalized by chord length (positive = tab, negative =
            blank).
        max_dev: Max signed deviation (normalized).
        min_dev: Min signed deviation (normalized).
        chord_length_px: The edge's chord length in image pixels.
        polyline: (100, 2) raw image-coordinate polyline, corner A -> corner B.
        canonical_polyline: (100, 2) chord-normalized canonical polyline (see
            `canonicalize_edge`).
    """

    index: int
    edge_type: str
    dominant_dev: float
    max_dev: float
    min_dev: float
    chord_length_px: float
    polyline: np.ndarray
    canonical_polyline: np.ndarray


def canonicalize_edge(polyline: np.ndarray) -> np.ndarray:
    """Map an edge polyline to the canonical frame.

    Translates the first point to the origin, rotates so the last point lies
    on the +x axis, and scales so the chord length is 1.

    Args:
        polyline: Nx2 edge points, corner A first, corner B last.

    Returns:
        Nx2 canonical polyline (A at (0,0), B at (1,0)).
    """
    a = polyline[0]
    chord = polyline[-1] - a
    length = float(np.linalg.norm(chord))
    if length < 1e-9:
        return np.zeros_like(polyline)
    cos_t = chord[0] / length
    sin_t = chord[1] / length
    rotation = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    return (polyline - a) @ rotation.T / length


def _signed_area(contour: np.ndarray) -> float:
    """Shoelace signed area of a closed contour in image coordinates.

    In image coordinates (y down), a visually-clockwise polygon has POSITIVE
    signed area.

    Args:
        contour: Nx2 contour points.

    Returns:
        The signed area.
    """
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _arc_slice(resampled: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract the cyclic slice start..end (inclusive) from a closed contour.

    Args:
        resampled: Nx2 closed contour points.
        start: Start index.
        end: End index (may be cyclically before `start`).

    Returns:
        The arc points from `start` to `end` inclusive, in contour order.
    """
    if end >= start:
        return resampled[start : end + 1]
    return np.vstack([resampled[start:], resampled[: end + 1]])


def classify_arc(arc: np.ndarray, centroid: np.ndarray) -> Tuple[str, float, float, float, float]:
    """Classify one edge arc as tab/blank/flat from its chord deviation profile.

    The chord runs between the arc's two endpoint corners. Every arc point
    gets a signed perpendicular deviation from the chord, with positive
    meaning away from the piece centroid. The dominant deviation (larger
    magnitude of max/min) decides the type.

    Args:
        arc: Mx2 arc points, endpoints being the two corners.
        centroid: The piece centroid (for the away-from-piece sign convention).

    Returns:
        Tuple of (edge type, dominant_dev, max_dev, min_dev, chord_length),
        deviations normalized by chord length.
    """
    corner_a = arc[0]
    corner_b = arc[-1]
    chord = corner_b - corner_a
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 1e-8:
        return "flat", 0.0, 0.0, 0.0, chord_length
    u = chord / chord_length

    def cross_with_u(points: np.ndarray) -> np.ndarray:
        rel = points - corner_a
        return u[0] * rel[:, 1] - u[1] * rel[:, 0]

    centroid_side = float(cross_with_u(centroid.reshape(1, 2))[0])
    away_sign = -1.0 if centroid_side > 0 else 1.0

    devs = cross_with_u(arc) * away_sign / chord_length
    max_dev = float(devs.max())
    min_dev = float(devs.min())
    dominant_dev = max_dev if abs(max_dev) >= abs(min_dev) else min_dev

    if abs(dominant_dev) < FLAT_THRESHOLD:
        edge_type = "flat"
    elif dominant_dev > 0:
        edge_type = "tab"
    else:
        edge_type = "blank"
    return edge_type, dominant_dev, max_dev, min_dev, chord_length


def split_edges(contour: np.ndarray) -> Optional[Tuple[List[Edge], CornerResult, bool]]:
    """Detect corners and split a contour into 4 classified, canonicalized edges.

    Args:
        contour: Nx2 contour in original image coordinates.

    Returns:
        Tuple of (4 edges in contour traversal order, the polydp
        `CornerResult`, the corner_disagreement flag), or None when corner
        detection fails or the detected corners collapse onto each other on
        the resampled contour.
    """
    try:
        primary, disagreement = detect_corners_with_cross_check(contour)
    except InsufficientCornersError:
        return None

    resampled = resample_contour(contour, SPLIT_RESAMPLE_POINTS)
    # Enforce visually-clockwise winding so traversal order matches the
    # clockwise corner order.
    if _signed_area(resampled) < 0:
        resampled = resampled[::-1]

    corner_idx = [nearest_contour_index(resampled, (float(c[0]), float(c[1]))) for c in primary.corners]
    if len(set(corner_idx)) < 4:
        return None

    forward = [(corner_idx[(j + 1) % 4] - corner_idx[j]) % SPLIT_RESAMPLE_POINTS for j in range(4)]
    if sum(forward) != SPLIT_RESAMPLE_POINTS:
        return None

    centroid = resampled.mean(axis=0)
    edges: List[Edge] = []
    for j in range(4):
        arc = _arc_slice(resampled, corner_idx[j], corner_idx[(j + 1) % 4])
        edge_type, dominant_dev, max_dev, min_dev, chord_length = classify_arc(arc, centroid)
        polyline = resample_polyline(arc, EDGE_POLYLINE_POINTS)
        edges.append(
            Edge(
                index=j,
                edge_type=edge_type,
                dominant_dev=dominant_dev,
                max_dev=max_dev,
                min_dev=min_dev,
                chord_length_px=chord_length,
                polyline=polyline,
                canonical_polyline=canonicalize_edge(polyline),
            )
        )

    return edges, primary, disagreement
