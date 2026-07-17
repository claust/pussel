#!/usr/bin/env python3
"""Render synthetic-benchmark failure cases for visual analysis.

Re-runs the same deterministic generation loop as `synth_benchmark.py` and
saves an annotated render for every piece where a detector's max corner error
exceeds the success threshold: contour (white), ground-truth corners (green),
predicted corners (red), and the detector's candidate pool (small blue dots)
when available.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/debug_synth_failures.py --n 200 --seed 42 --max-renders 12
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from common import mask_to_contour
from synth_benchmark import SUCCESS_THRESHOLD_PCT, generate_synthetic_piece, run_detector, score_corners

DEBUG_METHODS = ("polydp", "curvature")


def render_failure(
    mask: np.ndarray,
    contour: np.ndarray,
    predicted: np.ndarray,
    gt_corners: np.ndarray,
    error_pct: float,
    out_path: Path,
) -> None:
    """Save an annotated BGR render of one failure case.

    Args:
        mask: The degraded binary mask the contour came from.
        contour: Nx2 extracted contour.
        predicted: 4x2 predicted corners.
        gt_corners: 4x2 ground-truth corners.
        error_pct: The scored max corner error (percent of diagonal).
        out_path: PNG destination.
    """
    canvas = cv2.cvtColor((mask // 3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.polylines(canvas, [contour.astype(np.int32)], isClosed=True, color=(200, 200, 200), thickness=2)
    for pt in gt_corners:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), 3)
    for pt in predicted:
        cv2.drawMarker(canvas, (int(pt[0]), int(pt[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 4)
    cv2.putText(canvas, f"err {error_pct:.1f}%", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render synthetic corner-benchmark failures.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-renders", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "synth_failures")
    args = parser.parse_args()

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rendered = dict.fromkeys(DEBUG_METHODS, 0)
    for i in range(args.n):
        piece = generate_synthetic_piece(rng)
        if piece is None:
            continue
        contour = mask_to_contour(piece.mask)
        if contour is None:
            continue
        for method in DEBUG_METHODS:
            if rendered[method] >= args.max_renders:
                continue
            result = run_detector(method, contour, piece.mask.shape)
            if result is None:
                continue
            error_pct = score_corners(result.corners, piece.gt_corners)
            if error_pct <= SUCCESS_THRESHOLD_PCT:
                continue
            out_path = args.output_dir / f"piece{i:03d}_{method}_err{error_pct:.0f}.png"
            render_failure(piece.mask, contour, result.corners, piece.gt_corners, error_pct, out_path)
            rendered[method] += 1
        if all(count >= args.max_renders for count in rendered.values()):
            break

    print(f"Rendered failures: {rendered} -> {args.output_dir}")


if __name__ == "__main__":
    main()
