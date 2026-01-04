#!/usr/bin/env python3
"""Quick timing test for RoMa V2 on puzzle pieces."""

import csv
import sys
import time
from pathlib import Path

# Add RoMa to path before importing
sys.path.insert(0, "/Users/claus/Repos/RoMaV2/src")

from PIL import Image  # noqa: E402
from romav2 import RoMaV2  # noqa: E402

# Paths
DATASET_ROOT = Path("/tmp/roma_test_dataset")
PUZZLE_ROOT = Path("/Users/claus/Repos/pussel/network/datasets/puzzles")
METADATA_PATH = DATASET_ROOT / "metadata.csv"


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """Rotate image by angle degrees (clockwise)."""
    if angle == 0:
        return image
    # PIL rotates counter-clockwise, so negate
    return image.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)


def get_cell_from_coords(x: float, y: float, puzzle_size: int, grid_size: int = 4) -> int:
    """Convert pixel coordinates to cell index."""
    col = int(x / puzzle_size * grid_size)
    row = int(y / puzzle_size * grid_size)
    col = min(max(col, 0), grid_size - 1)
    row = min(max(row, 0), grid_size - 1)
    return row * grid_size + col


def evaluate_piece(
    model: RoMaV2,
    piece_path: Path,
    puzzle_path: Path,
    gt_cx: float,
    gt_cy: float,
    gt_rotation: int,
    num_matches: int = 500,
) -> dict:
    """Evaluate RoMa on a single piece.

    Returns dict with predictions and timing info.
    """
    piece_img = Image.open(piece_path)
    puzzle_img = Image.open(puzzle_path)
    puzzle_w, puzzle_h = puzzle_img.size
    piece_w, piece_h = piece_img.size

    # Ground truth cell
    gt_col = int(gt_cx * 4)
    gt_row = int(gt_cy * 4)
    gt_cell = gt_row * 4 + gt_col
    gt_rotation_idx = gt_rotation // 90

    # Try all 4 rotations
    best_overlap = -1
    best_rotation = 0
    best_matches = None
    rotation_results = []

    start_time = time.time()

    for test_rotation in [0, 90, 180, 270]:
        # Rotate piece to "undo" potential rotation
        corrected = rotate_image(piece_img, test_rotation)

        # Save temp file for RoMa
        temp_path = Path("/tmp/roma_piece_temp.png")
        corrected.save(temp_path)

        # Match against puzzle
        preds = model.match(str(temp_path), str(puzzle_path))
        matches, overlaps, _, _ = model.sample(preds, num_matches)

        mean_overlap = overlaps.mean().item()
        rotation_results.append((test_rotation, mean_overlap))

        if mean_overlap > best_overlap:
            best_overlap = mean_overlap
            best_rotation = test_rotation
            best_matches = matches

    inference_time = time.time() - start_time

    # Get position from best matches
    kpts_piece, kpts_puzzle = model.to_pixel_coordinates(best_matches, piece_h, piece_w, puzzle_h, puzzle_w)

    # Use median for robustness
    pred_x = kpts_puzzle[:, 0].median().item()
    pred_y = kpts_puzzle[:, 1].median().item()
    pred_cell = get_cell_from_coords(pred_x, pred_y, puzzle_w)

    # The rotation we applied to correct the piece
    # If piece was rotated by R degrees, we need to apply (360-R) to undo
    # So if best_rotation is our correction, the original rotation was (360-best_rotation) % 360
    # But wait - gt_rotation is the rotation that was applied to the piece
    # If we apply best_rotation and it matches, then best_rotation should equal (360 - gt_rotation) % 360
    # For prediction, we want to predict what rotation was applied to the piece
    # If best_rotation corrects the piece, then pred_rotation = (360 - best_rotation) % 360
    pred_rotation = (360 - best_rotation) % 360
    pred_rotation_idx = pred_rotation // 90

    return {
        "gt_cell": gt_cell,
        "pred_cell": pred_cell,
        "cell_correct": pred_cell == gt_cell,
        "gt_rotation": gt_rotation,
        "gt_rotation_idx": gt_rotation_idx,
        "pred_rotation": pred_rotation,
        "pred_rotation_idx": pred_rotation_idx,
        "rotation_correct": pred_rotation == gt_rotation,
        "both_correct": (pred_cell == gt_cell) and (pred_rotation == gt_rotation),
        "best_overlap": best_overlap,
        "inference_time": inference_time,
        "rotation_scores": rotation_results,
        "pred_x": pred_x,
        "pred_y": pred_y,
    }


def main() -> None:
    """Run quick timing test on a single puzzle (16 pieces)."""
    print("Loading RoMa V2...")
    start = time.time()
    cfg = RoMaV2.Cfg(compile=False, setting="fast")
    model = RoMaV2(cfg=cfg)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Load metadata
    with open(METADATA_PATH) as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print(f"\nFound {len(samples)} pieces")

    # Run on first N samples
    n_test = 16  # One full puzzle (16 pieces)
    print(f"\nTesting on {n_test} pieces...")
    print("=" * 70)

    results = []
    total_time = 0

    for i, sample in enumerate(samples[:n_test]):
        piece_path = DATASET_ROOT / sample["filename"]
        puzzle_id = sample["puzzle_id"]
        puzzle_path = PUZZLE_ROOT / f"{puzzle_id}.jpg"

        gt_cx = float(sample["cx"])
        gt_cy = float(sample["cy"])
        gt_rotation = int(sample["rotation"])

        print(f"\n[{i + 1}/{n_test}] {sample['filename']}")
        print(f"  GT: cell=({gt_cx:.3f}, {gt_cy:.3f}), rotation={gt_rotation}°")

        result = evaluate_piece(model, piece_path, puzzle_path, gt_cx, gt_cy, gt_rotation)
        results.append(result)
        total_time += result["inference_time"]

        print(
            f"  Pred: cell={result['pred_cell']} (gt={result['gt_cell']}), "
            f"rotation={result['pred_rotation']}° (gt={gt_rotation}°)"
        )
        print(
            f"  Cell: {'✓' if result['cell_correct'] else '✗'}, "
            f"Rotation: {'✓' if result['rotation_correct'] else '✗'}"
        )
        print(f"  Time: {result['inference_time']:.2f}s, Overlap: {result['best_overlap']:.3f}")
        print(f"  Rotation scores: {result['rotation_scores']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    cell_correct = sum(r["cell_correct"] for r in results)
    rot_correct = sum(r["rotation_correct"] for r in results)
    both_correct = sum(r["both_correct"] for r in results)

    print(f"Cell accuracy:     {cell_correct}/{n_test} ({100 * cell_correct / n_test:.1f}%)")
    print(f"Rotation accuracy: {rot_correct}/{n_test} ({100 * rot_correct / n_test:.1f}%)")
    print(f"Both correct:      {both_correct}/{n_test} ({100 * both_correct / n_test:.1f}%)")

    # Position error in normalized coordinates [0, 1]
    # Puzzle is 512x512, so we normalize
    position_errors = []
    for r, sample in zip(results, samples[:n_test]):
        gt_cx = float(sample["cx"])
        gt_cy = float(sample["cy"])
        pred_cx = r["pred_x"] / 512  # normalize to [0, 1]
        pred_cy = r["pred_y"] / 512
        error = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5
        position_errors.append(error)
        r["position_error"] = error

    mean_error = sum(position_errors) / len(position_errors)
    print("\nPosition error (normalized [0,1]):")
    print(f"  Mean: {mean_error:.4f}")
    print(f"  Per piece: {[f'{e:.4f}' for e in position_errors]}")

    # For reference: cell size is 0.25 in normalized coords (1/4)
    # So error < 0.125 means within half a cell
    within_half_cell = sum(1 for e in position_errors if e < 0.125)
    print(f"  Within half cell (<0.125): {within_half_cell}/{n_test}")

    print(f"\nTotal inference time: {total_time:.2f}s")
    print(f"Average per piece:    {total_time / n_test:.2f}s")
    print(f"Time for 2000 pieces: {total_time / n_test * 2000 / 60:.1f} minutes")


if __name__ == "__main__":
    main()
