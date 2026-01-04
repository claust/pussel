#!/usr/bin/env python3
"""Full evaluation of RoMa V2 on realistic puzzle pieces.

Evaluates position and rotation prediction using dense feature matching.
Saves results to outputs/ directory.
"""

import csv
import json
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

# Unbuffered print
print = partial(print, flush=True)

# Add RoMa to path before importing
sys.path.insert(0, "/Users/claus/Repos/RoMaV2/src")

from PIL import Image  # noqa: E402
from romav2 import RoMaV2  # noqa: E402

# Paths
EXPERIMENT_DIR = Path(__file__).parent
DATASET_ROOT = EXPERIMENT_DIR / "dataset"
PUZZLE_ROOT = Path("/Users/claus/Repos/pussel/network/datasets/puzzles")
METADATA_PATH = DATASET_ROOT / "metadata.csv"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """Rotate image by angle degrees (clockwise)."""
    if angle == 0:
        return image
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
    """Evaluate RoMa on a single piece."""
    piece_img = Image.open(piece_path)
    puzzle_img = Image.open(puzzle_path)
    puzzle_w, puzzle_h = puzzle_img.size
    piece_w, piece_h = piece_img.size

    # Ground truth
    gt_col = int(gt_cx * 4)
    gt_row = int(gt_cy * 4)
    gt_cell = gt_row * 4 + gt_col
    gt_rotation_idx = gt_rotation // 90

    # Try all 4 rotations
    best_overlap = -1
    best_rotation = 0
    best_matches = None
    rotation_scores = {}

    for test_rotation in [0, 90, 180, 270]:
        corrected = rotate_image(piece_img, test_rotation)
        temp_path = Path("/tmp/roma_piece_temp.png")
        corrected.save(temp_path)

        preds = model.match(str(temp_path), str(puzzle_path))
        matches, overlaps, _, _ = model.sample(preds, num_matches)

        mean_overlap = overlaps.mean().item()
        rotation_scores[test_rotation] = mean_overlap

        if mean_overlap > best_overlap:
            best_overlap = mean_overlap
            best_rotation = test_rotation
            best_matches = matches

    # Get position from best matches
    kpts_piece, kpts_puzzle = model.to_pixel_coordinates(best_matches, piece_h, piece_w, puzzle_h, puzzle_w)

    pred_x = kpts_puzzle[:, 0].median().item()
    pred_y = kpts_puzzle[:, 1].median().item()
    pred_cell = get_cell_from_coords(pred_x, pred_y, puzzle_w)

    # Predicted rotation (what was applied to piece)
    pred_rotation = (360 - best_rotation) % 360
    pred_rotation_idx = pred_rotation // 90

    # Position error in normalized coordinates
    pred_cx = pred_x / puzzle_w
    pred_cy = pred_y / puzzle_h
    position_error = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2) ** 0.5

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
        "position_error": position_error,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "pred_cx": pred_cx,
        "pred_cy": pred_cy,
        "gt_cx": gt_cx,
        "gt_cy": gt_cy,
        "rotation_scores": rotation_scores,
    }


def main() -> None:
    """Run full RoMa V2 evaluation on realistic puzzle pieces."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RoMa V2 Evaluation on Realistic Puzzle Pieces")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print("\nLoading RoMa V2...")
    start = time.time()
    cfg = RoMaV2.Cfg(compile=False, setting="fast")
    model = RoMaV2(cfg=cfg)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    # Load metadata
    with open(METADATA_PATH) as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    n_total = len(samples)
    print(f"\nDataset: {n_total} pieces from {n_total // 16} puzzles")
    print(f"Estimated time: {n_total * 5.6 / 60:.1f} minutes ({n_total * 5.6 / 3600:.1f} hours)")

    # Run evaluation
    print("\n" + "=" * 70)
    print("Running evaluation...")
    print("=" * 70)

    results = []
    total_time = 0
    start_eval = time.time()

    for i, sample in enumerate(samples):
        piece_path = DATASET_ROOT / sample["filename"]
        puzzle_id = sample["puzzle_id"]
        puzzle_path = PUZZLE_ROOT / f"{puzzle_id}.jpg"

        gt_cx = float(sample["cx"])
        gt_cy = float(sample["cy"])
        gt_rotation = int(sample["rotation"])

        piece_start = time.time()
        result = evaluate_piece(model, piece_path, puzzle_path, gt_cx, gt_cy, gt_rotation)
        piece_time = time.time() - piece_start
        total_time += piece_time

        result["piece_path"] = str(piece_path)
        result["puzzle_id"] = puzzle_id
        result["inference_time"] = piece_time
        results.append(result)

        # Progress update every 10 pieces
        if (i + 1) % 10 == 0 or (i + 1) == n_total:
            elapsed = time.time() - start_eval
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0

            cell_acc = sum(r["cell_correct"] for r in results) / len(results) * 100
            rot_acc = sum(r["rotation_correct"] for r in results) / len(results) * 100

            print(
                f"[{i + 1:4d}/{n_total}] Cell: {cell_acc:5.1f}% | Rot: {rot_acc:5.1f}% | "
                f"Rate: {rate:.2f}/s | ETA: {eta / 60:.1f}min"
            )

    # Final statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    cell_correct = sum(r["cell_correct"] for r in results)
    rot_correct = sum(r["rotation_correct"] for r in results)
    both_correct = sum(r["both_correct"] for r in results)
    mean_pos_error = sum(r["position_error"] for r in results) / len(results)
    mean_overlap = sum(r["best_overlap"] for r in results) / len(results)

    cell_acc = cell_correct / n_total * 100
    rot_acc = rot_correct / n_total * 100
    both_acc = both_correct / n_total * 100

    print("\nAccuracy:")
    print(f"  Cell (position):  {cell_correct:4d}/{n_total} ({cell_acc:.2f}%)")
    print(f"  Rotation:         {rot_correct:4d}/{n_total} ({rot_acc:.2f}%)")
    print(f"  Both correct:     {both_correct:4d}/{n_total} ({both_acc:.2f}%)")

    print("\nPosition metrics:")
    print(f"  Mean error (normalized): {mean_pos_error:.4f}")
    print(f"  Mean overlap score:      {mean_overlap:.4f}")

    print("\nTiming:")
    print(f"  Total time:      {total_time:.1f}s ({total_time / 60:.1f}min)")
    print(f"  Per piece:       {total_time / n_total:.2f}s")

    # Breakdown by rotation
    print("\nRotation confusion matrix:")
    rot_matrix = [[0] * 4 for _ in range(4)]
    for r in results:
        rot_matrix[r["gt_rotation_idx"]][r["pred_rotation_idx"]] += 1

    print("           Pred 0°  90° 180° 270°")
    for i, row in enumerate(rot_matrix):
        print(f"  GT {i * 90:3d}°: {row[0]:5d} {row[1]:4d} {row[2]:4d} {row[3]:4d}")

    # Per-rotation accuracy
    print("\nPer-rotation accuracy:")
    for rot in [0, 90, 180, 270]:
        rot_idx = rot // 90
        gt_count = sum(1 for r in results if r["gt_rotation_idx"] == rot_idx)
        correct = sum(1 for r in results if r["gt_rotation_idx"] == rot_idx and r["rotation_correct"])
        acc = correct / gt_count * 100 if gt_count > 0 else 0
        print(f"  {rot:3d}°: {correct}/{gt_count} ({acc:.1f}%)")

    # Save results
    print(f"\nSaving results to {OUTPUT_DIR}...")

    # Summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_pieces": n_total,
        "n_puzzles": n_total // 16,
        "cell_accuracy": cell_acc,
        "rotation_accuracy": rot_acc,
        "both_accuracy": both_acc,
        "mean_position_error": mean_pos_error,
        "mean_overlap": mean_overlap,
        "total_time_seconds": total_time,
        "time_per_piece": total_time / n_total,
        "model_setting": "fast",
        "rotation_confusion_matrix": rot_matrix,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Detailed CSV
    csv_fields = [
        "puzzle_id",
        "piece_path",
        "gt_cell",
        "pred_cell",
        "cell_correct",
        "gt_rotation",
        "pred_rotation",
        "rotation_correct",
        "both_correct",
        "gt_cx",
        "gt_cy",
        "pred_cx",
        "pred_cy",
        "position_error",
        "best_overlap",
        "inference_time",
    ]

    with open(OUTPUT_DIR / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("  - results.json (summary)")
    print("  - predictions.csv (detailed)")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
