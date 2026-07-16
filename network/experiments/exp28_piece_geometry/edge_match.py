#!/usr/bin/env python3
"""M4: canonical edge frames, mate flipping, and pairwise edge match distances.

Canonical frame: an M3 edge polyline (100 pts, corner A -> corner B in
clockwise contour order, image coordinates) is translated so A is at the
origin, rotated so B lies on the +x axis, and scaled so the chord |AB| = 1.
Because M3 contours are traversed clockwise in image coordinates (y down),
OUTWARD deviation maps to NEGATIVE canonical y: a tab dips below the x axis
at its peak, a blank rises above it.

Mate flip: a physically mating edge is the same curve traversed in the
opposite direction by the neighboring piece. In canonical frames the mate
transform is: reverse point order, then re-canonicalize. Re-canonicalizing
the reversed polyline rotates by 180 degrees, which maps (x, y) ->
(1 - x, -y) - the y negation the mate needs is inherent in that rotation,
so `flip_edge` is exactly reverse + re-canonicalize, and for a perfect mate
``flip_edge(mate) == query`` pointwise.

Distances (query Q vs flipped candidate C', both canonical 100-pt):
- ``l2``: mean pointwise Euclidean distance (same index alignment).
- ``chamfer``: symmetric mean nearest-neighbor distance.
- ``scalar6``: Euclidean distance between 6-feature vectors; the candidate
  contributes its OWN features mirrored (see `mirror_features`) instead of a
  flipped polyline.
- l2/chamfer chord-penalty variants: dist + LAMBDA * |log(chordQ/chordC)|.

The CLI runs a synthetic self-check: it builds a `puzzle_shapes` edge grid,
extracts the shared edge between two neighboring pieces from both sides, and
verifies distance(edge, flip(mate)) is near zero while
distance(edge, flip(non-mate)) is large.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/edge_match.py   # self-check
"""

import argparse
from typing import List, Tuple

import numpy as np
from common import resample_polyline
from puzzle_shapes import BezierCurve, generate_edge_grid, reverse_curves, transform_curves
from scipy.spatial.distance import cdist

CANONICAL_POINTS = 100

# Chord-length penalty weight for the *_chord distance variants.
CHORD_PENALTY_LAMBDA = 0.5

# Only tab<->blank pairs can mate; flat edges never mate. A query whose own
# type prediction is "flat" (necessarily a misclassification when the grid
# says the edge is interior) is scored against both feature types.
COMPATIBLE_TYPES = {"tab": ("blank",), "blank": ("tab",), "flat": ("tab", "blank")}


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


def flip_edge(canonical: np.ndarray) -> np.ndarray:
    """Transform a canonical edge into the frame of its would-be mate.

    Reverses the point order and re-canonicalizes; the re-canonicalization
    rotates by 180 degrees, mapping (x, y) -> (1 - x, -y). For a true mate
    the result coincides with the query's canonical polyline pointwise.

    Args:
        canonical: Nx2 canonical edge polyline.

    Returns:
        Nx2 flipped canonical polyline.
    """
    return canonicalize_edge(canonical[::-1])


def dist_l2(query: np.ndarray, candidate_flipped: np.ndarray) -> float:
    """Mean pointwise Euclidean distance between two canonical polylines.

    Args:
        query: Nx2 canonical query edge.
        candidate_flipped: Nx2 flipped canonical candidate edge.

    Returns:
        The mean pointwise distance.
    """
    return float(np.mean(np.linalg.norm(query - candidate_flipped, axis=1)))


def dist_chamfer(query: np.ndarray, candidate_flipped: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor (chamfer) distance.

    Args:
        query: Nx2 canonical query edge.
        candidate_flipped: Nx2 flipped canonical candidate edge.

    Returns:
        The chamfer distance.
    """
    dists = cdist(query, candidate_flipped)
    return float((dists.min(axis=1).mean() + dists.min(axis=0).mean()) / 2.0)


def edge_features(canonical: np.ndarray, chord_length_px: float) -> np.ndarray:
    """Compute the 6-feature scalar descriptor of a canonical edge.

    Features (outward-positive convention, i.e. dev = -canonical_y):
    0. signed peak deviation (positive for tabs, negative for blanks),
    1. arc-position of the peak along the chord, clipped to [0, 1],
    2. width at half peak height (x-extent of the contiguous half-peak run),
    3. total |area| between polyline and chord,
    4. asymmetry: signed outward area of the first half minus the second half,
    5. log(chord length in px) - pairwise differences of this feature give
       the log chord-ratio term.

    Args:
        canonical: Nx2 canonical edge polyline.
        chord_length_px: The edge's chord length in original-image pixels.

    Returns:
        The 6-element feature vector.
    """
    x = canonical[:, 0]
    dev = -canonical[:, 1]

    peak_idx = int(np.argmax(np.abs(dev)))
    peak = float(dev[peak_idx])
    pos = float(np.clip(x[peak_idx], 0.0, 1.0))

    width = 0.0
    if abs(peak) > 1e-9:
        above = dev * np.sign(peak) >= abs(peak) / 2.0
        lo = peak_idx
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = peak_idx
        while hi < len(dev) - 1 and above[hi + 1]:
            hi += 1
        width = float(abs(x[lo : hi + 1].max() - x[lo : hi + 1].min()))

    # Signed outward area via the line integral -integral(y dx) along the
    # polyline (the chord return path lies on y=0 and contributes nothing).
    seg_areas = 0.5 * (dev[:-1] + dev[1:]) * np.diff(x)
    half = len(canonical) // 2
    area_first = float(seg_areas[:half].sum())
    area_second = float(seg_areas[half:].sum())
    total_area = abs(area_first + area_second)
    asymmetry = area_first - area_second

    return np.array([peak, pos, width, total_area, asymmetry, np.log(max(chord_length_px, 1e-9))])


def mirror_features(features: np.ndarray) -> np.ndarray:
    """Mirror a candidate edge's features into the query's frame.

    Mirror rules for viewing the same physical junction from the mating
    side: the peak sign flips (tab <-> blank), the peak position reflects
    (pos -> 1 - pos), width and total |area| are invariant, and asymmetry is
    invariant (reversing traversal swaps the halves AND flips the outward
    sign, which cancel). The log-chord feature is unchanged; the query -
    candidate difference of it is the log chord-ratio term.

    Args:
        features: 6-element feature vector from `edge_features`.

    Returns:
        The mirrored 6-element feature vector.
    """
    peak, pos, width, total_area, asymmetry, log_chord = features
    return np.array([-peak, 1.0 - pos, width, total_area, asymmetry, log_chord])


def dist_scalar6(query_features: np.ndarray, candidate_features_mirrored: np.ndarray) -> float:
    """Euclidean distance between a query's features and a mirrored candidate's.

    Args:
        query_features: The query edge's 6-feature vector.
        candidate_features_mirrored: The candidate's feature vector after
            `mirror_features`.

    Returns:
        The Euclidean feature distance.
    """
    return float(np.linalg.norm(query_features - candidate_features_mirrored))


def chord_penalty(chord_q_px: float, chord_c_px: float, lam: float = CHORD_PENALTY_LAMBDA) -> float:
    """Chord-length penalty term: lam * |log(chordQ / chordC)|.

    Args:
        chord_q_px: Query chord length in pixels.
        chord_c_px: Candidate chord length in pixels.
        lam: Penalty weight (0 disables).

    Returns:
        The penalty value.
    """
    return lam * abs(float(np.log(max(chord_q_px, 1e-9) / max(chord_c_px, 1e-9))))


# --- Synthetic self-check ---------------------------------------------------


def _curves_to_polyline(curves: List[BezierCurve], n: int = CANONICAL_POINTS) -> np.ndarray:
    """Sample a curve chain densely and resample to n equidistant points.

    Args:
        curves: Bezier curves in traversal order.
        n: Output point count.

    Returns:
        Nx2 polyline.
    """
    points: List[Tuple[float, float]] = []
    for curve in curves:
        samples = curve.get_points(50)
        points.extend((float(p[0]), float(p[1])) for p in samples[:-1])
    last = curves[-1].get_points(2)[-1]
    points.append((float(last[0]), float(last[1])))
    return resample_polyline(np.array(points), n)


def self_check(seed: int = 7) -> None:
    """Verify flip/distance semantics on a synthetic shared edge from puzzle_shapes.

    Builds a 2x2 edge grid, extracts the interior vertical edge between
    pieces (0,0) and (0,1) from BOTH sides (as each piece traverses it,
    mirroring `get_piece_curves`), plus a non-mating interior edge, and
    prints distance(edge, flip(mate)) vs distance(edge, flip(non-mate)).

    Args:
        seed: Random seed for the edge grid.
    """
    grid = generate_edge_grid(2, 2, seed=seed)

    shared = grid.vertical_edges[0][1]
    # Piece (0,0) traverses this edge as its RIGHT edge: reversed, rotated 90 CCW.
    right_curves = transform_curves(
        reverse_curves(shared.curves), translate=(1.0, 0.0), scale=(1.0, 1.0), rotate_90_ccw=1
    )
    # Piece (0,1) traverses the same edge as its LEFT edge: forward, rotated 90 CCW.
    left_curves = transform_curves(shared.curves, translate=(0.0, 0.0), scale=(1.0, 1.0), rotate_90_ccw=1)

    other = grid.horizontal_edges[1][0]  # interior edge between (0,0) and (1,0)
    other_curves = transform_curves(
        reverse_curves(other.curves), translate=(0.0, 0.0), scale=(1.0, 1.0), rotate_90_ccw=0
    )

    query = canonicalize_edge(_curves_to_polyline(right_curves))
    mate = canonicalize_edge(_curves_to_polyline(left_curves))
    non_mate = canonicalize_edge(_curves_to_polyline(other_curves))

    for name, dist_fn in (("l2", dist_l2), ("chamfer", dist_chamfer)):
        d_mate = dist_fn(query, flip_edge(mate))
        d_other = dist_fn(query, flip_edge(non_mate))
        status = "OK" if d_mate < 0.01 and d_other > 5 * max(d_mate, 1e-9) else "FAIL"
        print(
            f"  {name:8s} dist(edge, flip(mate)) = {d_mate:.5f}   dist(edge, flip(non-mate)) = {d_other:.5f}  {status}"
        )

    fq = edge_features(query, 100.0)
    d_mate6 = dist_scalar6(fq, mirror_features(edge_features(mate, 100.0)))
    d_other6 = dist_scalar6(fq, mirror_features(edge_features(non_mate, 100.0)))
    status = "OK" if d_mate6 < 0.05 and d_other6 > 5 * max(d_mate6, 1e-9) else "FAIL"
    print(
        f"  {'scalar6':8s} dist(edge, mirror(mate)) = {d_mate6:.5f}   "
        f"dist(edge, mirror(non-mate)) = {d_other6:.5f}  {status}"
    )


def main() -> None:
    """CLI entry point: run the synthetic self-check."""
    parser = argparse.ArgumentParser(description="Edge match distances - synthetic self-check.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print("Synthetic self-check (puzzle_shapes shared edge, 2x2 grid):")
    self_check(seed=args.seed)


if __name__ == "__main__":
    main()
