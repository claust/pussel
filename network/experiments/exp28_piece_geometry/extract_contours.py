#!/usr/bin/env python3
"""M1: high-fidelity contour extraction for north_star piece photos.

For every piece photo (optionally filtered by puzzle/background), crops the
photo to the piece bbox with a margin, then extracts a contour with one or
both of:

- **rembg**: segmentation alpha -> hardened mask -> contour.
- **threshold**: grayscale Otsu at both polarities, picking whichever
  polarity's largest component has a plausible area ratio and the best
  solidity, -> contour.

Writes one JSON per piece to `outputs/contours/{puzzle_id}/{piece_stem}.json`
and appends one row per piece x method to `outputs/summary.csv`. Prints
progress every 25 pieces and a final per-background x per-method clean-rate
table.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/extract_contours.py --limit 8
    uv run python experiments/exp28_piece_geometry/extract_contours.py \
        --puzzle bambi --background red_carpet --method rembg
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from common import (
    PieceRecord,
    QualityMetrics,
    alpha_to_mask,
    contour_quality,
    crop_with_margin,
    load_metadata,
    mask_to_contour,
    otsu_masks,
    remove_background_rgba,
)

# Repo-relative default (this script lives in network/experiments/exp28_piece_geometry/,
# so parents[2] is the network/ dir); override with --dataset-root.
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "north_star" / "v1"
DEFAULT_MARGIN_FRAC = 0.15
PROGRESS_INTERVAL = 25


def _extract_rembg(crop_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[QualityMetrics]]:
    """Extract a contour from a crop via rembg segmentation.

    Args:
        crop_bgr: The piece crop in OpenCV BGR order.

    Returns:
        Tuple of (contour in crop-local coordinates, quality metrics), or
        (None, None) when no contour could be found.
    """
    rgba = remove_background_rgba(crop_bgr)
    mask = alpha_to_mask(rgba)
    contour = mask_to_contour(mask)
    if contour is None:
        return None, None
    quality = contour_quality(contour, mask, crop_bgr.shape[:2])
    return contour, quality


def _extract_threshold(crop_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[QualityMetrics], Optional[str]]:
    """Extract a contour from a crop via Otsu thresholding, picking the better polarity.

    Args:
        crop_bgr: The piece crop in OpenCV BGR order.

    Returns:
        Tuple of (contour in crop-local coordinates, quality metrics,
        polarity name "normal"/"inverted"), or (None, None, None) when
        neither polarity yields a contour.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    normal_mask, inverted_mask = otsu_masks(gray)

    candidates: List[Tuple[str, np.ndarray, np.ndarray, QualityMetrics]] = []
    for name, mask in (("normal", normal_mask), ("inverted", inverted_mask)):
        contour = mask_to_contour(mask)
        if contour is None:
            continue
        quality = contour_quality(contour, mask, crop_bgr.shape[:2])
        candidates.append((name, mask, contour, quality))

    if not candidates:
        return None, None, None

    plausible = [c for c in candidates if 0.05 <= c[3].area_ratio <= 0.9]
    pool = plausible if plausible else candidates
    best = max(pool, key=lambda c: c[3].solidity)
    return best[2], best[3], best[0]


def _piece_metadata_dict(record: PieceRecord, bbox: Tuple[int, int, int, int], margin_frac: float) -> Dict[str, Any]:
    """Build the JSON-serializable piece metadata block.

    Args:
        record: The piece's metadata row.
        bbox: The piece bounding box used for cropping.
        margin_frac: The margin fraction used for cropping.

    Returns:
        A dict suitable for `json.dump`.
    """
    return {
        "puzzle_id": record.puzzle_id,
        "piece_file": record.piece_file,
        "rows": record.rows,
        "cols": record.cols,
        "row": record.row,
        "col": record.col,
        "rotation": record.rotation,
        "background": record.background,
        "bbox": list(bbox),
        "margin_frac": margin_frac,
        "flagged": record.flagged,
        "bbox_suspect": record.bbox_suspect,
    }


def process_piece(
    dataset_root: Path, record: PieceRecord, methods: List[str], margin_frac: float = DEFAULT_MARGIN_FRAC
) -> Dict[str, Any]:
    """Extract contours for one piece photo with the requested methods.

    Args:
        dataset_root: The north_star dataset root.
        record: The piece's metadata row.
        methods: Which methods to run: any of "rembg", "threshold".
        margin_frac: Crop margin fraction, as a fraction of bbox size.

    Returns:
        A JSON-serializable dict with the piece metadata and per-method results.
    """
    image_path = dataset_root / record.piece_file
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    crop, offset = crop_with_margin(image, record.bbox, margin_frac=margin_frac)
    offset_x, offset_y = offset

    result: Dict[str, Any] = {
        "piece": _piece_metadata_dict(record, record.bbox, margin_frac),
        "methods": {},
    }

    if "rembg" in methods:
        contour, quality = _extract_rembg(crop)
        if contour is not None and quality is not None:
            original_coords = contour + np.array([offset_x, offset_y])
            result["methods"]["rembg"] = {
                "contour": original_coords.tolist(),
                "quality": quality.to_dict(),
            }
        else:
            result["methods"]["rembg"] = {"contour": None, "quality": None}

    if "threshold" in methods:
        contour, quality, polarity = _extract_threshold(crop)
        if contour is not None and quality is not None:
            original_coords = contour + np.array([offset_x, offset_y])
            result["methods"]["threshold"] = {
                "contour": original_coords.tolist(),
                "quality": quality.to_dict(),
                "polarity": polarity,
            }
        else:
            result["methods"]["threshold"] = {"contour": None, "quality": None, "polarity": None}

    return result


def _filter_records(
    records: List[PieceRecord], puzzle: Optional[str], background: Optional[str], limit: Optional[int]
) -> List[PieceRecord]:
    """Apply CLI filters to the loaded metadata rows.

    Args:
        records: All loaded piece records.
        puzzle: Substring filter on puzzle_id, or None.
        background: Exact-match filter on background, or None.
        limit: Maximum number of records to keep, or None.

    Returns:
        The filtered (and possibly truncated) record list.
    """
    filtered = records
    if puzzle:
        filtered = [r for r in filtered if puzzle in r.puzzle_id]
    if background:
        filtered = [r for r in filtered if r.background == background]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Extract high-fidelity contours from north_star piece photos.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    parser.add_argument("--background", type=str, default=None, help="Exact filter on background")
    parser.add_argument("--limit", type=int, default=None, help="Max number of piece photos to process")
    parser.add_argument("--method", choices=["rembg", "threshold", "both"], default="both")
    args = parser.parse_args()

    methods = ["rembg", "threshold"] if args.method == "both" else [args.method]

    records = load_metadata(args.dataset_root)
    records = _filter_records(records, args.puzzle, args.background, args.limit)
    print(f"Processing {len(records)} piece photos with methods={methods}")

    contours_dir = args.output_dir / "contours"
    contours_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "puzzle",
        "piece",
        "background",
        "method",
        "is_clean",
        "n_large_components",
        "border_touching",
        "area_ratio",
        "solidity",
    ]
    clean_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    total_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, record in enumerate(records, start=1):
            result = process_piece(args.dataset_root, record, methods)

            puzzle_dir = contours_dir / record.puzzle_id
            puzzle_dir.mkdir(parents=True, exist_ok=True)
            out_path = puzzle_dir / f"{record.piece_stem}.json"
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle)

            for method in methods:
                method_result = result["methods"].get(method, {})
                quality = method_result.get("quality")
                is_clean = bool(quality["is_clean"]) if quality else False
                writer.writerow(
                    {
                        "puzzle": record.puzzle_id,
                        "piece": record.piece_stem,
                        "background": record.background,
                        "method": method,
                        "is_clean": is_clean,
                        "n_large_components": quality["n_large_components"] if quality else "",
                        "border_touching": quality["border_touching"] if quality else "",
                        "area_ratio": f"{quality['area_ratio']:.4f}" if quality else "",
                        "solidity": f"{quality['solidity']:.4f}" if quality else "",
                    }
                )
                key = (record.background, method)
                total_counts[key] += 1
                if is_clean:
                    clean_counts[key] += 1

            if i % PROGRESS_INTERVAL == 0:
                print(f"  ...{i}/{len(records)} pieces processed")

    print(f"\nDone. {len(records)} pieces processed. Summary written to {summary_path}")

    backgrounds = sorted({r.background for r in records})
    print("\nClean-rate table (is_clean fraction):")
    header = "background".ljust(14) + "".join(m.ljust(14) for m in methods)
    print(header)
    for bg in backgrounds:
        row = bg.ljust(14)
        for method in methods:
            key = (bg, method)
            total = total_counts[key]
            clean = clean_counts[key]
            rate = f"{clean}/{total} ({100.0 * clean / total:.1f}%)" if total else "n/a"
            row += rate.ljust(14)
        print(row)


if __name__ == "__main__":
    main()
