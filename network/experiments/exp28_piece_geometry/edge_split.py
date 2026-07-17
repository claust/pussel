#!/usr/bin/env python3
"""M3: split clean contours into 4 edges and classify each as tab/blank/flat.

For every piece x background with a clean rembg contour (produced by
`extract_contours.py`):

1. Detect corners with polydp (primary); run curvature as a cross-check and
   flag `corner_disagreement` when the two disagree by more than 3% of the
   bbox diagonal (optimal Hungarian matching).
2. Split a densely resampled (1024-pt) contour at the 4 corners into 4 arcs.
3. Map the arcs to grid directions N/E/S/W: corners are ordered clockwise
   from top-left, so with a clockwise-wound contour the arc corner0->corner1
   is North, then East, South, West. All 944 photos have rotation 0
   (verified from metadata); a nonzero rotation would shift the mapping.
4. Classify each arc by its dominant signed perpendicular deviation from the
   corner-to-corner chord (positive = away from the piece centroid):
   |dominant| < FLAT_THRESHOLD -> flat, positive -> tab, negative -> blank.

Writes one JSON per piece x background to
`outputs/piece_records/{puzzle_id}/{piece_stem}.json`, appends
`outputs/edge_summary.csv`, and prints a histogram of |dominant deviation|
across all edges (used to place FLAT_THRESHOLD in the bimodal gap).

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/edge_split.py
    uv run python experiments/exp28_piece_geometry/edge_split.py --puzzle bambi
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from common import PieceRecord, load_metadata, resample_contour, resample_polyline
from corner_detect import (
    CornerResult,
    InsufficientCornersError,
    _bbox_diagonal,
    _nearest_contour_index,
    detect_corners_curvature,
    detect_corners_polydp,
)
from scipy.optimize import linear_sum_assignment

# Repo-relative default (this script lives in network/experiments/exp28_piece_geometry/,
# so parents[2] is the network/ dir); override with --dataset-root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "north_star" / "v1"

# Dense resampling used for edge splitting (finer than the detectors' 512).
SPLIT_RESAMPLE_POINTS = 1024

# Each edge arc is stored resampled to this many equidistant points.
EDGE_POLYLINE_POINTS = 100

# Corner cross-check: polydp vs curvature max matched-corner distance (as a
# fraction of the bbox diagonal) above which the piece is flagged.
CORNER_DISAGREEMENT_FRAC = 0.03

# Tab/blank/flat threshold on |dominant deviation| / chord length. The
# measured distribution across all 3,664 clean edges is strongly bimodal:
# grid-truth flat edges cluster in [0, 0.04] (median 0.009, p90 0.021) and
# tab/blank features in [0.11, 0.42] (median 0.292, p1 0.109; the shallow
# tail is the big-piece toddler puzzles, where tab height is small relative
# to the chord). 0.07 sits mid-gap and maximizes measured flat-vs-nonflat
# accuracy; the residual errors are outliers (corner/segmentation faults),
# not threshold placement.
FLAT_THRESHOLD = 0.07

DIRECTIONS = ("N", "E", "S", "W")


def _signed_area(contour: np.ndarray) -> float:
    """Shoelace signed area of a closed contour in image coordinates.

    In image coordinates (y down), a visually-clockwise polygon (the order
    TL -> TR -> BR -> BL) has POSITIVE signed area.

    Args:
        contour: Nx2 contour points.

    Returns:
        The signed area.
    """
    x = contour[:, 0]
    y = contour[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _corner_max_distance_frac(a: np.ndarray, b: np.ndarray, diagonal: float) -> float:
    """Max matched-corner distance between two 4-corner sets, as a fraction of the diagonal.

    Args:
        a: 4x2 corner set.
        b: 4x2 corner set.
        diagonal: Piece bbox diagonal for normalization.

    Returns:
        Max distance over the optimal (Hungarian) corner assignment, divided
        by `diagonal`.
    """
    dist_matrix = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    return float(dist_matrix[row_ind, col_ind].max()) / max(diagonal, 1e-8)


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


def split_piece(contour: np.ndarray, rotation: int) -> Optional[Tuple[Dict[str, Dict[str, Any]], CornerResult, bool]]:
    """Split one clean contour into 4 classified, direction-mapped edges.

    Args:
        contour: Nx2 contour in original image coordinates.
        rotation: The piece's photographed rotation label in degrees
            (0/90/180/270); nonzero values shift the arc -> direction mapping.

    Returns:
        Tuple of (per-direction edge dicts, the polydp corner result, the
        corner_disagreement flag), or None when corner detection fails or
        corners collapse onto each other on the resampled contour.
    """
    try:
        primary = detect_corners_polydp(contour)
    except InsufficientCornersError:
        return None

    disagreement = True
    try:
        cross_check = detect_corners_curvature(contour)
        diagonal = _bbox_diagonal(contour)
        frac = _corner_max_distance_frac(primary.corners, cross_check.corners, diagonal)
        disagreement = frac > CORNER_DISAGREEMENT_FRAC
    except InsufficientCornersError:
        pass  # keep disagreement=True: no cross-check available

    resampled = resample_contour(contour, SPLIT_RESAMPLE_POINTS)
    # Enforce visually-clockwise winding (positive shoelace area in image
    # coordinates) so traversal order matches the clockwise corner order.
    if _signed_area(resampled) < 0:
        resampled = resampled[::-1]

    corner_idx = [_nearest_contour_index(resampled, (float(c[0]), float(c[1]))) for c in primary.corners]
    if len(set(corner_idx)) < 4:
        return None

    # Corners are clockwise from top-left; verify the snapped indices advance
    # cyclically in contour order (they must, given the winding fix).
    forward = [(corner_idx[(j + 1) % 4] - corner_idx[j]) % SPLIT_RESAMPLE_POINTS for j in range(4)]
    if sum(forward) != SPLIT_RESAMPLE_POINTS:
        return None

    centroid = resampled.mean(axis=0)
    rot_steps = (rotation // 90) % 4

    edges: Dict[str, Dict[str, Any]] = {}
    for j in range(4):
        arc = _arc_slice(resampled, corner_idx[j], corner_idx[(j + 1) % 4])
        # Image-frame arc j (N,E,S,W for an upright piece); a piece
        # photographed rotated clockwise by rot_steps has its grid-frame
        # directions shifted back by the same amount.
        direction = DIRECTIONS[(j - rot_steps) % 4]
        edge_type, dominant_dev, max_dev, min_dev, chord_length = classify_arc(arc, centroid)
        polyline = resample_polyline(arc, EDGE_POLYLINE_POINTS)
        edges[direction] = {
            "type": edge_type,
            "dominant_dev": dominant_dev,
            "max_dev": max_dev,
            "min_dev": min_dev,
            "chord_length_px": chord_length,
            "polyline": polyline.tolist(),
        }

    return edges, primary, disagreement


def _print_histogram(values: List[float], bin_width: float = 0.02) -> None:
    """Print an ASCII histogram of |dominant deviation| values.

    Args:
        values: The |dominant_dev| values across all edges.
        bin_width: Histogram bin width.
    """
    arr = np.array(values)
    top = float(arr.max()) if len(arr) else 0.0
    n_bins = int(np.ceil(top / bin_width)) + 1
    counts, _ = np.histogram(arr, bins=n_bins, range=(0.0, n_bins * bin_width))
    peak = counts.max() if counts.max() > 0 else 1
    print(f"\n|dominant_dev| distribution across {len(arr)} edges (bin width {bin_width}):")
    for b, count in enumerate(counts):
        lo = b * bin_width
        bar = "#" * int(round(50 * count / peak))
        print(f"  {lo:5.2f}-{lo + bin_width:4.2f}  {count:5d}  {bar}")


def _load_clean_contour(contours_dir: Path, puzzle_id: str, piece_stem: str) -> Optional[np.ndarray]:
    """Load a piece's clean rembg contour from the M1 outputs, if it exists.

    Args:
        contours_dir: The `outputs/contours` directory.
        puzzle_id: The piece's puzzle id.
        piece_stem: The piece's filename stem.

    Returns:
        Nx2 contour array in original image coordinates, or None when the
        contour JSON is missing, has no rembg contour, or is not clean.
    """
    contour_path = contours_dir / puzzle_id / f"{piece_stem}.json"
    if not contour_path.exists():
        return None
    with open(contour_path, encoding="utf-8") as handle:
        data = json.load(handle)
    rembg = data["methods"].get("rembg")
    if not rembg or not rembg.get("contour") or not rembg["quality"]["is_clean"]:
        return None
    return np.array(rembg["contour"], dtype=np.float64)


def _write_piece_record(
    records_dir: Path,
    record: PieceRecord,
    edges: Dict[str, Dict[str, Any]],
    corner_result: CornerResult,
    disagreement: bool,
) -> None:
    """Write one piece-record JSON.

    Args:
        records_dir: The `outputs/piece_records` directory.
        record: The piece's metadata row (`PieceRecord`).
        edges: Per-direction edge dicts from `split_piece`.
        corner_result: The polydp corner result.
        disagreement: The corner_disagreement flag.
    """
    piece_record = {
        "puzzle_id": record.puzzle_id,
        "piece_file": record.piece_file,
        "row": record.row,
        "col": record.col,
        "rows": record.rows,
        "cols": record.cols,
        "background": record.background,
        "rotation": record.rotation,
        "corners": corner_result.corners.tolist(),
        "corner_confidences": corner_result.confidences,
        "corner_disagreement": disagreement,
        "edges": edges,
    }
    puzzle_dir = records_dir / record.puzzle_id
    puzzle_dir.mkdir(parents=True, exist_ok=True)
    with open(puzzle_dir / f"{record.piece_stem}.json", "w", encoding="utf-8") as handle:
        json.dump(piece_record, handle)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Split clean contours into classified N/E/S/W edges.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--contours-dir", type=Path, default=Path(__file__).parent / "outputs" / "contours")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    parser.add_argument("--background", type=str, default=None, help="Exact filter on background")
    args = parser.parse_args()

    records = load_metadata(args.dataset_root)
    if args.puzzle:
        records = [r for r in records if args.puzzle in r.puzzle_id]
    if args.background:
        records = [r for r in records if r.background == args.background]

    nonzero_rotations = [r for r in records if r.rotation != 0]
    if nonzero_rotations:
        print(f"NOTE: {len(nonzero_rotations)} pieces have rotation != 0; direction mapping is shifted for them.")

    records_dir = args.output_dir / "piece_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "edge_summary.csv"

    n_processed = 0
    n_skipped_dirty = 0
    n_failed = 0
    n_disagreement = 0
    dominant_values: List[float] = []

    with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=["puzzle", "piece", "background", "direction", "type", "dominant_dev", "corner_disagreement"],
        )
        writer.writeheader()

        for record in records:
            contour = _load_clean_contour(args.contours_dir, record.puzzle_id, record.piece_stem)
            if contour is None:
                n_skipped_dirty += 1
                continue

            result = split_piece(contour, record.rotation)
            if result is None:
                n_failed += 1
                continue
            edges, corner_result, disagreement = result
            n_disagreement += int(disagreement)

            _write_piece_record(records_dir, record, edges, corner_result, disagreement)
            for direction in DIRECTIONS:
                edge = edges[direction]
                dominant_values.append(abs(edge["dominant_dev"]))
                writer.writerow(
                    {
                        "puzzle": record.puzzle_id,
                        "piece": record.piece_stem,
                        "background": record.background,
                        "direction": direction,
                        "type": edge["type"],
                        "dominant_dev": f"{edge['dominant_dev']:.4f}",
                        "corner_disagreement": disagreement,
                    }
                )

            n_processed += 1
            if n_processed % 100 == 0:
                print(f"  ...{n_processed} pieces split")

    print(
        f"\nDone. {n_processed} pieces split, {n_skipped_dirty} skipped (no clean contour), "
        f"{n_failed} failed (corner detection/splitting), {n_disagreement} flagged corner_disagreement."
    )
    print(f"Records in {records_dir}, summary in {summary_path}")
    _print_histogram(dominant_values)
    print(f"\nFLAT_THRESHOLD = {FLAT_THRESHOLD}")


if __name__ == "__main__":
    main()
