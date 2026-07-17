#!/usr/bin/env python3
"""M1 review: render contact sheets of extracted contours for human QA.

For each puzzle x background combination, lays the piece crops out in a
rows x cols grid (matching the puzzle's own grid), draws the extracted
contour on each cell (green if `is_clean`, red otherwise), and titles each
cell "r{row}c{col}". Saves one PNG per puzzle x background to
`outputs/review/{puzzle_id}_{background}_{method}.png`.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/contact_sheets.py
    uv run python experiments/exp28_piece_geometry/contact_sheets.py --puzzle bambi --method threshold
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from common import PieceRecord, crop_with_margin, load_metadata

# Repo-relative default (this script lives in network/experiments/exp28_piece_geometry/,
# so parents[2] is the network/ dir); override with --dataset-root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "north_star" / "v1"
DEFAULT_MARGIN_FRAC = 0.15
CELL_WIDTH = 220
TITLE_HEIGHT = 18
CLEAN_COLOR = (0, 200, 0)  # BGR green
DIRTY_COLOR = (0, 0, 220)  # BGR red
MISSING_COLOR = (0, 165, 255)  # BGR orange


def _load_contour_json(contours_dir: Path, record: PieceRecord) -> Optional[Dict[str, Any]]:
    """Load the saved contour JSON for one piece, if present.

    Args:
        contours_dir: The `outputs/contours` directory.
        record: The piece's metadata row.

    Returns:
        The parsed JSON dict, or None if no file exists for this piece.
    """
    path = contours_dir / record.puzzle_id / f"{record.piece_stem}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _render_cell(
    dataset_root: Path, record: PieceRecord, margin_frac: float, method_data: Optional[Dict[str, Any]]
) -> np.ndarray:
    """Render one contact-sheet cell: the piece crop with its contour overlaid.

    Args:
        dataset_root: The north_star dataset root.
        record: The piece's metadata row.
        margin_frac: Crop margin fraction, matching what `extract_contours.py`
            used so the contour overlay lines up with the crop.
        method_data: The `methods[method]` block from the piece's contour
            JSON, or None if no contour data is available for this method.

    Returns:
        A fixed-size BGR image cell, including a title strip.
    """
    image_path = dataset_root / record.piece_file
    image = cv2.imread(str(image_path))
    crop, offset = crop_with_margin(image, record.bbox, margin_frac=margin_frac)
    offset_x, offset_y = offset

    scale = CELL_WIDTH / crop.shape[1]
    resized = cv2.resize(crop, (CELL_WIDTH, max(1, round(crop.shape[0] * scale))))

    contour = None
    is_clean = False
    if method_data is not None and method_data.get("contour"):
        contour = np.array(method_data["contour"], dtype=np.float64)
        contour = contour - np.array([offset_x, offset_y])
        contour = contour * scale
        is_clean = bool(method_data["quality"]["is_clean"])

    if contour is not None:
        color = CLEAN_COLOR if is_clean else DIRTY_COLOR
        cv2.polylines(resized, [contour.astype(np.int32)], isClosed=True, color=color, thickness=2)
    else:
        cv2.rectangle(resized, (0, 0), (resized.shape[1] - 1, resized.shape[0] - 1), MISSING_COLOR, 2)

    title = np.full((TITLE_HEIGHT, CELL_WIDTH, 3), 255, dtype=np.uint8)
    label = f"r{record.row:02d}c{record.col:02d}"
    cv2.putText(title, label, (2, TITLE_HEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cell = np.full((TITLE_HEIGHT + resized.shape[0], CELL_WIDTH, 3), 255, dtype=np.uint8)
    cell[:TITLE_HEIGHT] = title
    cell[TITLE_HEIGHT : TITLE_HEIGHT + resized.shape[0]] = resized
    return cell


def build_contact_sheet(dataset_root: Path, contours_dir: Path, records: List[PieceRecord], method: str) -> np.ndarray:
    """Build one puzzle x background contact sheet image.

    Args:
        dataset_root: The north_star dataset root.
        contours_dir: The `outputs/contours` directory.
        records: All piece records for this puzzle x background (one per grid cell).
        method: Which method's contour to draw ("rembg" or "threshold").

    Returns:
        The assembled BGR contact sheet image.
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
            data = _load_contour_json(contours_dir, record)
            method_data = data["methods"].get(method) if data else None
            margin_frac = data["piece"]["margin_frac"] if data else DEFAULT_MARGIN_FRAC
            cell = _render_cell(dataset_root, record, margin_frac, method_data)
            cell_h = max(cell_h, cell.shape[0])
            row_cells.append(cell)
        cells.append(row_cells)

    # Pad every cell to a common height so rows stack cleanly.
    sheet_rows = []
    for row_cells in cells:
        padded = []
        for cell in row_cells:
            if cell.shape[0] < cell_h:
                pad = np.full((cell_h - cell.shape[0], CELL_WIDTH, 3), 255, dtype=np.uint8)
                cell = np.vstack([cell, pad])
            padded.append(cell)
        sheet_rows.append(np.hstack(padded))

    return np.vstack(sheet_rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Render contact sheets of extracted contours for review.")
    parser.add_argument("--contours-dir", type=Path, default=Path(__file__).parent / "outputs" / "contours")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--method", choices=["rembg", "threshold"], default="rembg")
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    args = parser.parse_args()

    records = load_metadata(args.dataset_root)
    if args.puzzle:
        records = [r for r in records if args.puzzle in r.puzzle_id]

    groups: Dict[Tuple[str, str], List[PieceRecord]] = defaultdict(list)
    for record in records:
        groups[(record.puzzle_id, record.background)].append(record)

    review_dir = args.output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    for (puzzle_id, background), group_records in sorted(groups.items()):
        sheet = build_contact_sheet(args.dataset_root, args.contours_dir, group_records, args.method)
        out_path = review_dir / f"{puzzle_id}_{background}_{args.method}.png"
        cv2.imwrite(str(out_path), sheet)
        print(f"Wrote {out_path} ({sheet.shape[1]}x{sheet.shape[0]}, {len(group_records)} pieces)")


if __name__ == "__main__":
    main()
