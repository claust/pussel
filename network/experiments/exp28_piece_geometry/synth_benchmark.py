#!/usr/bin/env python3
"""M2 quantitative: corner-detector bake-off on synthetic pieces with known ground truth.

Generates `--n` random pieces via `puzzle_shapes.PieceConfig.random()`,
rasterizes each to a filled mask (~500px piece size), applies a random
rotation (expanding the canvas so nothing clips), Gaussian blur, light noise,
and binarization to mimic a segmented real-photo mask, then runs all three
`corner_detect` detectors and scores them against the known corner positions
(the base square's corners, carried through the same rotation).

Metric per piece per method: max over the 4 corners of (distance from
predicted to its optimally-assigned true corner) / piece diagonal size, as a
percentage. A method "succeeds" on a piece when that value is <= 3%.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/synth_benchmark.py --n 10
    uv run python experiments/exp28_piece_geometry/synth_benchmark.py --n 200 --seed 42
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from common import mask_to_contour
from corner_detect import (
    CornerResult,
    InsufficientCornersError,
    detect_corners_curvature,
    detect_corners_polydp,
    detect_corners_shitomasi,
)
from puzzle_shapes import PieceConfig, generate_piece_path
from scipy.optimize import linear_sum_assignment

PIECE_SIZE_PX = 500.0
CANVAS_PAD_FRAC = 0.15  # extra canvas padding, as a fraction of piece size, for tab bulges
BLUR_SIGMA = 1.0
NOISE_STD = 6.0
SUCCESS_THRESHOLD_PCT = 3.0

METHODS = ("curvature", "polydp", "shitomasi")


@dataclass(frozen=True)
class SyntheticPiece:
    """A rasterized synthetic piece with known ground-truth corners.

    Attributes:
        mask: Binary mask (H, W) with values in {0, 255}, post rotation/blur/noise.
        gt_corners: 4x2 array of ground-truth corner positions in mask pixel
            coordinates, in the same order as the base square
            (0,0)-(size,0)-(size,size)-(0,size) after rotation.
    """

    mask: np.ndarray
    gt_corners: np.ndarray


def rasterize_piece(config: PieceConfig, piece_size_px: float = PIECE_SIZE_PX) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize a piece config's path to a filled mask, with matching ground-truth corners.

    The ground-truth corners are the 4 corners of the piece's base square
    (before corner rounding); `corner_radius` rounds them slightly in the
    rendered path, so the true corners are a small underestimate of the
    rendered mask's extremal points.

    Args:
        config: The piece configuration to rasterize.
        piece_size_px: Pixel size of the base square's side.

    Returns:
        Tuple of (mask (H, W) uint8 in {0, 255}, ground-truth corners (4x2)),
        both in the same unrotated pixel coordinate frame.
    """
    x, y = generate_piece_path(config)
    path = np.column_stack([x, y])
    gt_local = np.array(
        [[0.0, 0.0], [config.size, 0.0], [config.size, config.size], [0.0, config.size]],
    )

    scale = piece_size_px / config.size
    pad = CANVAS_PAD_FRAC * config.size

    min_xy = path.min(axis=0)
    max_y = path[:, 1].max()

    def to_pixels(pts: np.ndarray) -> np.ndarray:
        px = (pts[:, 0] - min_xy[0] + pad) * scale
        py = (max_y - pts[:, 1] + pad) * scale  # flip y: path is math-convention (y up)
        return np.column_stack([px, py])

    path_px = to_pixels(path)
    gt_px = to_pixels(gt_local)

    extent = path.max(axis=0) - min_xy
    canvas_w = int(np.ceil((extent[0] + 2 * pad) * scale))
    canvas_h = int(np.ceil((extent[1] + 2 * pad) * scale))

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    cv2.fillPoly(mask, [path_px.astype(np.int32)], (255,))
    return mask, gt_px


def apply_random_rotation(mask: np.ndarray, gt_corners: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a mask and its ground-truth corners by the same transform, expanding the canvas.

    Args:
        mask: Binary mask (H, W).
        gt_corners: 4x2 ground-truth corners in `mask`'s pixel coordinates.
        angle_deg: Rotation angle in degrees.

    Returns:
        Tuple of (rotated mask on an expanded canvas, transformed corners).
    """
    h, w = mask.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]

    rotated_mask = cv2.warpAffine(mask, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

    ones = np.ones((gt_corners.shape[0], 1))
    homogeneous = np.hstack([gt_corners, ones])
    rotated_corners = (matrix @ homogeneous.T).T
    return rotated_mask, rotated_corners


def degrade_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blur, add light noise to, and re-binarize a mask to mimic a segmented real photo.

    Args:
        mask: Binary mask (H, W) with values in {0, 255}.
        rng: Numpy random generator for reproducible noise.

    Returns:
        Re-binarized mask (H, W) with values in {0, 255}.
    """
    blurred = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=BLUR_SIGMA)
    noise = rng.normal(0.0, NOISE_STD, size=blurred.shape)
    noisy = np.clip(blurred.astype(np.float64) + noise, 0, 255)
    return np.where(noisy > 127, np.uint8(255), np.uint8(0))


def generate_synthetic_piece(rng: np.random.Generator) -> Optional[SyntheticPiece]:
    """Generate one synthetic piece: rasterize, rotate, degrade.

    Args:
        rng: Numpy random generator for rotation angle and noise.

    Returns:
        The `SyntheticPiece`, or None if rasterization failed (degenerate config).
    """
    config = PieceConfig.random()
    mask, gt_corners = rasterize_piece(config)
    if mask.sum() == 0:
        return None

    angle = float(rng.uniform(0.0, 360.0))
    rotated_mask, rotated_corners = apply_random_rotation(mask, gt_corners, angle)
    final_mask = degrade_mask(rotated_mask, rng)
    if final_mask.sum() == 0:
        return None

    return SyntheticPiece(mask=final_mask, gt_corners=rotated_corners)


def _piece_diagonal(gt_corners: np.ndarray) -> float:
    """Diagonal size of the ground-truth square (distance between opposite corners).

    Args:
        gt_corners: 4x2 ground-truth corners, ordered around the square.

    Returns:
        Euclidean distance between corners 0 and 2.
    """
    return float(np.linalg.norm(gt_corners[0] - gt_corners[2]))


def score_corners(predicted: np.ndarray, gt_corners: np.ndarray) -> float:
    """Score predicted corners against ground truth via optimal assignment.

    Args:
        predicted: 4x2 predicted corner points.
        gt_corners: 4x2 ground-truth corner points.

    Returns:
        Max matched-corner distance, as a percentage of the piece diagonal.
    """
    dist_matrix = np.linalg.norm(predicted[:, None, :] - gt_corners[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    max_dist = float(dist_matrix[row_ind, col_ind].max())
    diagonal = _piece_diagonal(gt_corners)
    return 100.0 * max_dist / diagonal if diagonal > 0 else float("inf")


def run_detector(method: str, contour: np.ndarray, mask_shape: Tuple[int, int]) -> Optional[CornerResult]:
    """Run one named detector, returning None if it fails to find enough corners.

    Args:
        method: One of "curvature", "polydp", "shitomasi".
        contour: Nx2 contour points.
        mask_shape: (height, width) of the source mask, needed by "shitomasi".

    Returns:
        The `CornerResult`, or None on `InsufficientCornersError`.
    """
    try:
        if method == "curvature":
            return detect_corners_curvature(contour)
        if method == "polydp":
            return detect_corners_polydp(contour)
        if method == "shitomasi":
            return detect_corners_shitomasi(contour, mask_shape)
    except InsufficientCornersError:
        return None
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Synthetic corner-detector benchmark against known ground truth.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "synth_benchmark.csv"

    per_method_errors: Dict[str, List[float]] = {m: [] for m in METHODS}
    per_method_success: Dict[str, int] = dict.fromkeys(METHODS, 0)
    per_method_detected: Dict[str, int] = dict.fromkeys(METHODS, 0)
    n_generated = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["piece_index", "method", "detected", "max_error_pct", "success"])
        writer.writeheader()

        for i in range(args.n):
            piece = generate_synthetic_piece(rng)
            if piece is None:
                print(f"  [warn] piece {i}: rasterization/degradation produced an empty mask, skipping")
                continue
            n_generated += 1

            contour = mask_to_contour(piece.mask)
            if contour is None:
                print(f"  [warn] piece {i}: no contour found in degraded mask, skipping")
                continue

            for method in METHODS:
                result = run_detector(method, contour, piece.mask.shape)
                if result is None:
                    writer.writerow(
                        {"piece_index": i, "method": method, "detected": False, "max_error_pct": "", "success": False}
                    )
                    continue
                per_method_detected[method] += 1
                error_pct = score_corners(result.corners, piece.gt_corners)
                success = error_pct <= SUCCESS_THRESHOLD_PCT
                per_method_errors[method].append(error_pct)
                if success:
                    per_method_success[method] += 1
                writer.writerow(
                    {
                        "piece_index": i,
                        "method": method,
                        "detected": True,
                        "max_error_pct": f"{error_pct:.3f}",
                        "success": success,
                    }
                )

    print(f"\nGenerated {n_generated}/{args.n} synthetic pieces. Results written to {csv_path}\n")
    print(f"{'method':<12}{'coverage':<14}{'median err%':<14}{'mean err%':<14}{'<=3% err':<12}")
    for method in METHODS:
        errors = per_method_errors[method]
        detected = per_method_detected[method]
        coverage = f"{detected}/{n_generated}" if n_generated else "n/a"
        median_err = f"{float(np.median(errors)):.2f}" if errors else "n/a"
        mean_err = f"{float(np.mean(errors)):.2f}" if errors else "n/a"
        success_rate = (
            f"{per_method_success[method]}/{n_generated} ({100.0 * per_method_success[method] / n_generated:.1f}%)"
            if n_generated
            else "n/a"
        )
        print(f"{method:<12}{coverage:<14}{median_err:<14}{mean_err:<14}{success_rate:<12}")


if __name__ == "__main__":
    main()
