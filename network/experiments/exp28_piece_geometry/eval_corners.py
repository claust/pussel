#!/usr/bin/env python3
"""M2 evaluation + review: run corner detectors on real photos and (optionally) score them.

For every piece with a clean rembg contour (from `extract_contours.py`
output), runs the requested corner detector(s) and renders a per-puzzle x
background contact sheet with the contour and each method's predicted
corners overlaid, to `outputs/review_corners/{puzzle}_{background}.png`.

When `--labels-file` (see `label_corners.py`) exists, also scores each
method's predictions against the hand-labeled ground truth (same
max-corner-error-over-diagonal metric as `synth_benchmark.py`) and writes
`outputs/corner_eval.csv`.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/eval_corners.py --limit 8
    uv run python experiments/exp28_piece_geometry/eval_corners.py \
        --labels-file outputs/corner_labels.json --method all
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from common import PieceRecord, crop_with_margin, load_metadata
from corner_detect import (
    CornerResult,
    InsufficientCornersError,
    detect_corners_curvature,
    detect_corners_polydp,
    detect_corners_shitomasi,
)
from synth_benchmark import score_corners

# Repo-relative default (this script lives in network/experiments/exp28_piece_geometry/,
# so parents[2] is the network/ dir); override with --dataset-root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "north_star" / "v1"
ALL_METHODS = ("curvature", "polydp", "shitomasi")
METHOD_COLORS = {  # BGR
    "curvature": (0, 0, 255),
    "polydp": (255, 128, 0),
    "shitomasi": (255, 0, 255),
}
CELL_WIDTH = 260
HEADER_HEIGHT = 24
TITLE_HEIGHT = 18


def _run_detector(method: str, contour: np.ndarray, image_shape: Tuple[int, int]) -> Optional[CornerResult]:
    """Run one named detector on a contour, returning None on failure.

    Args:
        method: One of "curvature", "polydp", "shitomasi".
        contour: Nx2 contour points in original image coordinates.
        image_shape: (height, width) of the source photo, used by "shitomasi".

    Returns:
        The `CornerResult`, or None when the detector could not find 4 corners.
    """
    try:
        if method == "curvature":
            return detect_corners_curvature(contour)
        if method == "polydp":
            return detect_corners_polydp(contour)
        if method == "shitomasi":
            return detect_corners_shitomasi(contour, image_shape)
    except InsufficientCornersError:
        return None
    raise ValueError(f"Unknown method: {method}")


def _order_by_angle(pts: np.ndarray) -> np.ndarray:
    """Order 4 points around their centroid so opposite indices (0,2) are true diagonal corners.

    Hand-labeled clicks arrive in whatever order the user clicked them; this
    makes the ordering consistent (like the detectors' clockwise ordering) so
    `score_corners`'s diagonal-distance computation is meaningful.

    Args:
        pts: 4x2 array of points, any order.

    Returns:
        4x2 array, reordered by angle around the centroid.
    """
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]


def _load_piece_json(contours_dir: Path, record: PieceRecord) -> Optional[Dict[str, Any]]:
    """Load a piece's saved contour JSON, if present.

    Args:
        contours_dir: The `outputs/contours` directory.
        record: The piece's metadata row.

    Returns:
        The parsed JSON dict, or None.
    """
    path = contours_dir / record.puzzle_id / f"{record.piece_stem}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _render_cell(
    dataset_root: Path,
    record: PieceRecord,
    margin_frac: float,
    contour: Optional[np.ndarray],
    predictions: Dict[str, Optional[CornerResult]],
) -> np.ndarray:
    """Render one contact-sheet cell: piece crop, contour, and per-method corner dots.

    Args:
        dataset_root: The north_star dataset root.
        record: The piece's metadata row.
        margin_frac: Crop margin fraction matching `extract_contours.py`.
        contour: The piece's rembg contour in original image coordinates, or
            None if unavailable.
        predictions: Per-method `CornerResult` (original image coordinates),
            or None per method when detection failed.

    Returns:
        A fixed-size BGR image cell, including a title strip.
    """
    image = cv2.imread(str(dataset_root / record.piece_file))
    crop, offset = crop_with_margin(image, record.bbox, margin_frac=margin_frac)
    offset_arr = np.array(offset)

    scale = CELL_WIDTH / crop.shape[1]
    resized = cv2.resize(crop, (CELL_WIDTH, max(1, round(crop.shape[0] * scale))))

    if contour is not None:
        local = (contour - offset_arr) * scale
        cv2.polylines(resized, [local.astype(np.int32)], isClosed=True, color=(200, 200, 200), thickness=1)

    for method, result in predictions.items():
        if result is None:
            continue
        color = METHOD_COLORS[method]
        local_corners = (result.corners - offset_arr) * scale
        for pt in local_corners.astype(np.int32):
            cv2.circle(resized, tuple(pt), 5, color, -1, lineType=cv2.LINE_AA)

    title = np.full((TITLE_HEIGHT, CELL_WIDTH, 3), 255, dtype=np.uint8)
    label = f"r{record.row:02d}c{record.col:02d}"
    cv2.putText(title, label, (2, TITLE_HEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cell = np.full((TITLE_HEIGHT + resized.shape[0], CELL_WIDTH, 3), 255, dtype=np.uint8)
    cell[:TITLE_HEIGHT] = title
    cell[TITLE_HEIGHT : TITLE_HEIGHT + resized.shape[0]] = resized
    return cell


def _legend_header(width: int, methods: List[str]) -> np.ndarray:
    """Render a header strip listing each method's marker color.

    Args:
        width: Header width in pixels (matches the sheet width).
        methods: Method names to show in the legend.

    Returns:
        A BGR header image of shape (HEADER_HEIGHT, width, 3).
    """
    header = np.full((HEADER_HEIGHT, width, 3), 255, dtype=np.uint8)
    x = 8
    for method in methods:
        color = METHOD_COLORS[method]
        cv2.circle(header, (x, HEADER_HEIGHT // 2), 5, color, -1)
        cv2.putText(
            header, method, (x + 10, HEADER_HEIGHT // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
        )
        x += 12 * len(method) + 30
    return header


def build_corner_sheet(
    dataset_root: Path,
    contours_dir: Path,
    records: List[PieceRecord],
    methods: List[str],
) -> np.ndarray:
    """Build one puzzle x background corner-detection contact sheet.

    Args:
        dataset_root: The north_star dataset root.
        contours_dir: The `outputs/contours` directory.
        records: All piece records for this puzzle x background (one per grid cell).
        methods: Detector methods to run and draw.

    Returns:
        The assembled BGR contact sheet image, including a legend header.
    """
    rows = records[0].rows
    cols = records[0].cols
    by_pos: Dict[Tuple[int, int], PieceRecord] = {(r.row, r.col): r for r in records}

    cells: List[List[np.ndarray]] = []
    cell_h = 0
    for row in range(rows):
        row_cells = []
        for col in range(cols):
            record = by_pos.get((row, col))
            if record is None:
                row_cells.append(np.full((TITLE_HEIGHT + CELL_WIDTH, CELL_WIDTH, 3), 128, dtype=np.uint8))
                continue
            data = _load_piece_json(contours_dir, record)
            rembg = data["methods"].get("rembg") if data else None
            margin_frac = data["piece"]["margin_frac"] if data else 0.15
            contour = np.array(rembg["contour"]) if rembg and rembg.get("contour") else None
            predictions: Dict[str, Optional[CornerResult]] = {}
            if contour is not None and rembg["quality"]["is_clean"]:
                image = cv2.imread(str(dataset_root / record.piece_file))
                for method in methods:
                    predictions[method] = _run_detector(method, contour, image.shape[:2])
            cell = _render_cell(dataset_root, record, margin_frac, contour, predictions)
            cell_h = max(cell_h, cell.shape[0])
            row_cells.append(cell)
        cells.append(row_cells)

    sheet_rows = []
    for row_cells in cells:
        padded = []
        for cell in row_cells:
            if cell.shape[0] < cell_h:
                pad = np.full((cell_h - cell.shape[0], CELL_WIDTH, 3), 255, dtype=np.uint8)
                cell = np.vstack([cell, pad])
            padded.append(cell)
        sheet_rows.append(np.hstack(padded))

    grid = np.vstack(sheet_rows)
    header = _legend_header(grid.shape[1], methods)
    return np.vstack([header, grid])


def evaluate_against_labels(
    dataset_root: Path,
    contours_dir: Path,
    records: List[PieceRecord],
    methods: List[str],
    labels: Dict[str, Any],
) -> Dict[Tuple[str, str], List[float]]:
    """Score each method's predictions against hand-labeled ground truth.

    Args:
        dataset_root: The north_star dataset root.
        contours_dir: The `outputs/contours` directory.
        records: Pieces to evaluate.
        methods: Detector methods to score.
        labels: The loaded labels dict, keyed by piece_file.

    Returns:
        Dict mapping (method, background) to a list of per-piece max-error percentages.
    """
    errors: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for record in records:
        label = labels.get(record.piece_file)
        if label is None:
            continue
        data = _load_piece_json(contours_dir, record)
        rembg = data["methods"].get("rembg") if data else None
        if not rembg or not rembg.get("contour") or not rembg["quality"]["is_clean"]:
            continue
        contour = np.array(rembg["contour"])
        image = cv2.imread(str(dataset_root / record.piece_file))
        gt_corners = _order_by_angle(np.array(label["corners"], dtype=np.float64))

        for method in methods:
            result = _run_detector(method, contour, image.shape[:2])
            if result is None:
                continue
            error_pct = score_corners(result.corners, gt_corners)
            errors[(method, record.background)].append(error_pct)
    return errors


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate and review corner detectors on real piece photos.")
    parser.add_argument("--contours-dir", type=Path, default=Path(__file__).parent / "outputs" / "contours")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--method", choices=[*ALL_METHODS, "all"], default="all")
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    parser.add_argument("--background", type=str, default=None, help="Exact filter on background")
    parser.add_argument("--max-sheets", type=int, default=None)
    args = parser.parse_args()

    methods = list(ALL_METHODS) if args.method == "all" else [args.method]

    records = load_metadata(args.dataset_root)
    if args.puzzle:
        records = [r for r in records if args.puzzle in r.puzzle_id]
    if args.background:
        records = [r for r in records if r.background == args.background]

    groups: Dict[Tuple[str, str], List[PieceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.puzzle_id, record.background)].append(record)

    review_dir = args.output_dir / "review_corners"
    review_dir.mkdir(parents=True, exist_ok=True)

    n_sheets = 0
    for (puzzle_id, background), group_records in sorted(groups.items()):
        if args.max_sheets is not None and n_sheets >= args.max_sheets:
            break
        sheet = build_corner_sheet(args.dataset_root, args.contours_dir, group_records, methods)
        out_path = review_dir / f"{puzzle_id}_{background}.png"
        cv2.imwrite(str(out_path), sheet)
        print(f"Wrote {out_path} ({sheet.shape[1]}x{sheet.shape[0]}, {len(group_records)} pieces)")
        n_sheets += 1

    if args.labels_file is None or not args.labels_file.exists():
        print("\nNo labels file given (or not found yet) -- skipping quantitative evaluation.")
        return

    with open(args.labels_file, encoding="utf-8") as handle:
        labels = json.load(handle)

    errors = evaluate_against_labels(args.dataset_root, args.contours_dir, records, methods, labels)

    csv_path = args.output_dir / "corner_eval.csv"
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("method,background,n,median_error_pct,mean_error_pct,pct_le_3\n")
        print(f"\n{'method':<12}{'background':<14}{'n':<6}{'median err%':<14}{'mean err%':<14}{'<=3% err':<10}")
        for (method, background), values in sorted(errors.items()):
            if not values:
                continue
            median_err = float(np.median(values))
            mean_err = float(np.mean(values))
            pct_le_3 = 100.0 * sum(1 for v in values if v <= 3.0) / len(values)
            print(f"{method:<12}{background:<14}{len(values):<6}{median_err:<14.2f}{mean_err:<14.2f}{pct_le_3:<10.1f}")
            handle.write(f"{method},{background},{len(values)},{median_err:.3f},{mean_err:.3f},{pct_le_3:.1f}\n")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
