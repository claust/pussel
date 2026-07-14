"""Training entry point for the realistic 4x4 benchmark (methodology v2).

Unified script for local (MPS/CPU) and CUDA training — it absorbs the
former train_cuda.py (automatic mixed precision is enabled automatically
on CUDA). The methodology follows CRITICAL_REVIEW.md:

- Datasets come from the frozen train/val/test split (splits.py); the
  script refuses to run without the checked-in split JSON.
- The best checkpoint is selected on VALIDATION both-correct accuracy;
  the test set is never evaluated during training.
- Train metrics are measured in eval mode on the frozen train_eval
  subset with the same deterministic protocol as val/test.
- The test set is evaluated once, on the val-selected checkpoint, and
  only when --eval-test is passed. Do not pass it while iterating on
  hyperparameters or architecture — that is what val is for.

Run from the network/ directory:
    uv run python -m experiments.exp20_realistic_pieces.train --epochs 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .dataset import GRID_SIZE, NUM_CELLS, create_datasets_from_split
from .harness import SELECTION_METRIC, evaluate, fit, load_best_checkpoint
from .model import FastBackboneModel, count_parameters
from .visualize import save_prediction_grid, save_training_curves

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "methodology_v2"


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main(
    epochs: int = 50,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    piece_size: int = 128,
    puzzle_size: int = 256,
    dataset_root: Path | str | None = None,
    puzzle_root: Path | str | None = None,
    split_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    eval_test: bool = False,
) -> dict[str, Any]:
    """Train on the frozen split with val-based checkpoint selection.

    Args:
        epochs: Number of training epochs.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone.
        head_lr: Learning rate for heads.
        weight_decay: AdamW weight decay.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        dataset_root: Root directory for realistic pieces dataset.
        puzzle_root: Root directory for source puzzle images.
        split_path: Path to the frozen split JSON (default: checked-in v1).
        output_dir: Directory for checkpoints, results and plots.
        num_workers: Data loader workers.
        eval_test: Evaluate the test set ONCE on the best-val checkpoint
            after training. Leave off while iterating.

    Returns:
        Dictionary with results (also written to results.json).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("REALISTIC 4x4 TRAINING (METHODOLOGY V2: FROZEN SPLIT + VAL SELECTION)")
    print("=" * 70)
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone LR: {backbone_lr}, Head LR: {head_lr}, Weight decay: {weight_decay}")
    print(f"Piece size: {piece_size}, Puzzle size: {puzzle_size}")
    print(f"Output dir: {output_dir}")

    device = get_device()
    use_amp = device.type == "cuda"
    print(f"\nDevice: {device} (AMP: {use_amp})")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Datasets from the frozen split
    dataset_kwargs: dict[str, Any] = {"piece_size": piece_size, "puzzle_size": puzzle_size}
    if dataset_root is not None:
        dataset_kwargs["dataset_root"] = dataset_root
    if puzzle_root is not None:
        dataset_kwargs["puzzle_root"] = puzzle_root
    if split_path is not None:
        dataset_kwargs["split_path"] = split_path
    datasets = create_datasets_from_split(**dataset_kwargs)

    # Model
    backbone_name = "shufflenet_v2_x0_5"
    print(f"\nCreating model with {backbone_name} backbone...")
    model = FastBackboneModel(
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Random baseline (cell): {1.0 / NUM_CELLS:.1%}")

    param_groups = model.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Train with val-based checkpoint selection (test untouched)
    train_start = time.time()
    history, best_epoch, best_val_metrics = fit(
        model,
        train_dataset=datasets["train"],
        train_eval_dataset=datasets["train_eval"],
        val_dataset=datasets["val"],
        optimizer=optimizer,
        device=device,
        grid_size=GRID_SIZE,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        num_workers=num_workers,
        use_amp=use_amp,
    )
    total_time = time.time() - train_start

    results: dict[str, Any] = {
        "experiment": "realistic_4x4_methodology_v2",
        "backbone": backbone_name,
        "grid_size": GRID_SIZE,
        "num_cells": NUM_CELLS,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "piece_size": piece_size,
        "puzzle_size": puzzle_size,
        "device": str(device),
        "amp": use_amp,
        "total_training_time": total_time,
        "selection_metric": f"val_{SELECTION_METRIC}",
        "best_epoch": best_epoch,
        "best_val_cell_acc": best_val_metrics["cell_accuracy"],
        "best_val_rot_acc": best_val_metrics["rotation_accuracy"],
        "best_val_both_acc": best_val_metrics["both_accuracy"],
        "history": history,
    }

    save_training_curves(history, output_dir / "training_curves.png")
    print("Saved training_curves.png")

    # Final test evaluation: ONCE, on the val-selected checkpoint, opt-in.
    if eval_test:
        print("\nEvaluating TEST set once on the best-val checkpoint...")
        checkpoint_epoch = load_best_checkpoint(model, output_dir, device)
        test_metrics = evaluate(
            model,
            datasets["test"],
            device,
            grid_size=GRID_SIZE,
            batch_size=batch_size * 2,
            num_workers=num_workers,
            collect=True,
        )
        print(f"TEST (checkpoint from epoch {checkpoint_epoch}):")
        print(f"  Cell accuracy:     {test_metrics['cell_accuracy']:.1%}")
        print(f"  Rotation accuracy: {test_metrics['rotation_accuracy']:.1%}")
        print(f"  Both correct:      {test_metrics['both_accuracy']:.1%}")

        results["test_cell_acc"] = test_metrics["cell_accuracy"]
        results["test_rot_acc"] = test_metrics["rotation_accuracy"]
        results["test_both_acc"] = test_metrics["both_accuracy"]
        results["test_n_samples"] = test_metrics["n_samples"]

        save_prediction_grid(
            predictions=test_metrics["predictions"],
            targets=test_metrics["targets"],
            pred_cells=test_metrics["pred_cells"],
            true_cells=test_metrics["true_cells"],
            pred_rotations=test_metrics["pred_rotations"],
            true_rotations=test_metrics["true_rotations"],
            output_path=output_dir / "test_predictions.png",
        )
        print("Saved test_predictions.png")
    else:
        print("\nTest set NOT evaluated (pass --eval-test for the one-shot final evaluation).")

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realistic 4x4 training (frozen split, val-based selection)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--piece-size", type=int, default=128, help="Piece image size")
    parser.add_argument("--puzzle-size", type=int, default=256, help="Puzzle image size")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Realistic pieces dataset root")
    parser.add_argument("--puzzle-root", type=Path, default=None, help="Source puzzle images root")
    parser.add_argument("--split-path", type=Path, default=None, help="Frozen split JSON (default: checked-in v1)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the test set ONCE on the best-val checkpoint after training",
    )
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
        dataset_root=args.dataset_root,
        puzzle_root=args.puzzle_root,
        split_path=args.split_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        eval_test=args.eval_test,
    )
