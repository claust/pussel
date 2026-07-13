#!/usr/bin/env python3
"""Re-evaluate the exp20 checkpoint on the test set with FIXED rotation labels.

The original exp20 evaluation used RealisticPieceTestDataset with a bug: the
rotation baked into each piece PNG at generation time was discarded (base
rotation hardcoded to 0), so test rotation labels were scrambled and rotation
accuracy was capped at ~25% regardless of model quality. This script evaluates
the original checkpoint on the same test puzzles with correctly composed
labels ((baked + applied) % 360).

Cell accuracy is unaffected by the label bug, so reproducing the original
~72.9% cell accuracy validates that the regenerated test set is faithful.

Usage:
    python reevaluate_checkpoint.py \
        --checkpoint /path/to/checkpoint_best.pt \
        --dataset-root /path/to/datasets/realistic_4x4_20k_test \
        --puzzle-root /path/to/datasets/puzzles
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset import GRID_SIZE, RealisticPieceTestDataset, get_puzzle_ids  # noqa: E402
from model import FastBackboneModel  # noqa: E402


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    """Evaluate the checkpoint with corrected rotation labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--puzzle-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "reeval_fixed_labels.json",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Model: architecture as trained (train_cuda.py saved a raw state_dict)
    model = FastBackboneModel(backbone_name="shufflenet_v2_x0_5", pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    test_ids = get_puzzle_ids(args.dataset_root)
    dataset = RealisticPieceTestDataset(
        puzzle_ids=test_ids,
        dataset_root=args.dataset_root,
        puzzle_root=args.puzzle_root,
        piece_size=128,
        puzzle_size=256,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_true_cells: list[int] = []
    all_pred_cells: list[int] = []
    all_true_rots: list[int] = []
    all_pred_rots: list[int] = []

    start = time.time()
    with torch.no_grad():
        for i, (pieces, puzzles, _targets, cells, rotations) in enumerate(loader):
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)

            positions, rotation_logits, _ = model(pieces, puzzles)

            # Cell from predicted position (same rule as predict_cell, without
            # a second forward pass)
            col = torch.clamp((positions[:, 0] * GRID_SIZE).long(), 0, GRID_SIZE - 1)
            row = torch.clamp((positions[:, 1] * GRID_SIZE).long(), 0, GRID_SIZE - 1)
            pred_cells = row * GRID_SIZE + col

            all_pred_cells.extend(pred_cells.cpu().tolist())
            all_true_cells.extend(cells.tolist())
            all_pred_rots.extend(rotation_logits.argmax(dim=1).cpu().tolist())
            all_true_rots.extend(rotations.tolist())

            if (i + 1) % 25 == 0:
                done = len(all_true_cells)
                print(f"  [{done}/{len(dataset)}] {time.time() - start:.0f}s", flush=True)

    n = len(all_true_cells)
    cell_acc = sum(p == t for p, t in zip(all_pred_cells, all_true_cells)) / n
    rot_acc = sum(p == t for p, t in zip(all_pred_rots, all_true_rots)) / n
    both_acc = (
        sum(
            pc == tc and pr == tr
            for pc, tc, pr, tr in zip(all_pred_cells, all_true_cells, all_pred_rots, all_true_rots)
        )
        / n
    )

    # Rotation confusion matrix (true x predicted)
    confusion = [[0] * 4 for _ in range(4)]
    for t, p in zip(all_true_rots, all_pred_rots):
        confusion[t][p] += 1

    # Per-true-rotation accuracy
    per_rot_acc = {}
    for r in range(4):
        total = sum(confusion[r])
        per_rot_acc[f"{r * 90}deg"] = confusion[r][r] / total if total else 0.0

    print("\n" + "=" * 60)
    print("RE-EVALUATION WITH FIXED ROTATION LABELS")
    print("=" * 60)
    print(f"Samples: {n} ({len(test_ids)} puzzles x 16 cells x 4 rotations)")
    print(f"Cell accuracy:     {cell_acc:.1%}  (original buggy eval: 72.9%)")
    print(f"Rotation accuracy: {rot_acc:.1%}  (original buggy eval: 24.8%)")
    print(f"Cell AND rotation: {both_acc:.1%}")
    print(f"Per-rotation accuracy: {per_rot_acc}")
    print("Rotation confusion (rows=true 0/90/180/270, cols=pred):")
    for row_counts in confusion:
        print(f"  {row_counts}")

    results = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "n_test_puzzles": len(test_ids),
        "n_samples": n,
        "cell_accuracy": cell_acc,
        "rotation_accuracy": rot_acc,
        "cell_and_rotation_accuracy": both_acc,
        "per_rotation_accuracy": per_rot_acc,
        "rotation_confusion_true_x_pred": confusion,
        "original_buggy_cell_accuracy": 0.7285677083333333,
        "original_buggy_rotation_accuracy": 0.24764322916666667,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
