"""Training script for 3x3 grid position prediction.

Exp16: Tests whether the model can scale from 2x2 quadrants (4 positions) to
3x3 grid (9 positions). Uses ShuffleNetV2_x0.5 for fast experimentation.

Key changes from exp15:
- 9 grid cells instead of 4 quadrants
- Single backbone (ShuffleNetV2) instead of comparison
- Focus on verifying convergence with finer grid
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import GRID_SIZE, NUM_CELLS, GridAllRotationsDataset, create_datasets
from .model import FastBackboneModel, count_parameters
from .visualize import save_prediction_grid, save_training_curves


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_epoch(
    model: FastBackboneModel,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_position_loss = 0.0
    total_rotation_loss = 0.0
    total_cell_correct = 0
    total_rotation_correct = 0
    total_samples = 0

    for pieces, puzzles, targets, cells, rotations in loader:
        pieces = pieces.to(device)
        puzzles = puzzles.to(device)
        targets = targets.to(device)
        cells = cells.to(device)
        rotations = rotations.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds, rotation_logits, _ = model(pieces, puzzles)

        # Position loss (MSE)
        position_loss = F.mse_loss(preds, targets)

        # Rotation loss (cross-entropy)
        rotation_loss = F.cross_entropy(rotation_logits, rotations)

        # Combined loss
        loss = position_weight * position_loss + rotation_weight * rotation_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = targets.size(0)
        total_position_loss += position_loss.item() * batch_size
        total_rotation_loss += rotation_loss.item() * batch_size
        total_samples += batch_size

        # Cell accuracy (3x3 grid)
        pred_cells = model.predict_cell(pieces, puzzles, grid_size=GRID_SIZE)
        total_cell_correct += (pred_cells == cells).sum().item()

        # Rotation accuracy
        pred_rotations = rotation_logits.argmax(dim=1)
        total_rotation_correct += (pred_rotations == rotations).sum().item()

    return {
        "position_loss": total_position_loss / total_samples,
        "rotation_loss": total_rotation_loss / total_samples,
        "cell_accuracy": total_cell_correct / total_samples,
        "rotation_accuracy": total_rotation_correct / total_samples,
    }


def compute_metrics(
    model: FastBackboneModel,
    dataset: GridAllRotationsDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compute evaluation metrics on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_cell_correct = 0
    total_rotation_correct = 0
    total_mse = 0.0
    n_samples = 0

    with torch.no_grad():
        for pieces, puzzles, targets, cells, rotations in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)
            targets = targets.to(device)

            preds, rotation_logits, _ = model(pieces, puzzles)

            mse = F.mse_loss(preds, targets, reduction="sum")
            total_mse += mse.item()
            n_samples += targets.size(0)

            pred_cells = model.predict_cell(pieces, puzzles, grid_size=GRID_SIZE)
            total_cell_correct += (pred_cells == cells.to(device)).sum().item()

            pred_rotations = rotation_logits.argmax(dim=1)
            total_rotation_correct += (pred_rotations == rotations.to(device)).sum().item()

    return {
        "mse_loss": total_mse / n_samples,
        "cell_accuracy": total_cell_correct / n_samples,
        "rotation_accuracy": total_rotation_correct / n_samples,
    }


def collect_predictions(
    model: FastBackboneModel,
    dataset: GridAllRotationsDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, list[Any]]:
    """Collect all predictions for visualization.

    Args:
        model: Trained model.
        dataset: Test dataset.
        device: Device to use.
        batch_size: Batch size for inference.

    Returns:
        Dictionary with predictions, targets, cells, rotations.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions: list[tuple[float, float]] = []
    all_targets: list[tuple[float, float]] = []
    all_pred_cells: list[int] = []
    all_true_cells: list[int] = []
    all_pred_rotations: list[int] = []
    all_true_rotations: list[int] = []

    with torch.no_grad():
        for pieces, puzzles, targets, cells, rotations in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)

            preds, rotation_logits, _ = model(pieces, puzzles)

            # Collect position predictions
            for i in range(preds.size(0)):
                all_predictions.append((preds[i, 0].item(), preds[i, 1].item()))
                all_targets.append((targets[i, 0].item(), targets[i, 1].item()))

            # Collect cell predictions
            pred_cells = model.predict_cell(pieces, puzzles, grid_size=GRID_SIZE)
            all_pred_cells.extend(pred_cells.cpu().tolist())
            all_true_cells.extend(cells.tolist())

            # Collect rotation predictions
            pred_rots = rotation_logits.argmax(dim=1)
            all_pred_rotations.extend(pred_rots.cpu().tolist())
            all_true_rotations.extend(rotations.tolist())

    return {
        "predictions": all_predictions,
        "targets": all_targets,
        "pred_cells": all_pred_cells,
        "true_cells": all_true_cells,
        "pred_rotations": all_pred_rotations,
        "true_rotations": all_true_rotations,
    }


def main(
    epochs: int = 10,
    n_train: int = 500,
    n_test: int = 100,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    piece_size: int = 128,
    puzzle_size: int = 256,
) -> dict[str, Any]:
    """Run 3x3 grid training with ShuffleNetV2_x0.5.

    Args:
        epochs: Number of training epochs.
        n_train: Number of training puzzles.
        n_test: Number of test puzzles.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone.
        head_lr: Learning rate for heads.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.

    Returns:
        Dictionary with results.
    """
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 16: 3x3 GRID POSITION PREDICTION")
    print("=" * 70)
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells")
    print(f"Epochs: {epochs}")
    print(f"Training puzzles: {n_train}")
    print(f"Test puzzles: {n_test}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone LR: {backbone_lr}")
    print(f"Head LR: {head_lr}")
    print(f"Piece size: {piece_size}")
    print(f"Puzzle size: {puzzle_size}")

    device = get_device()
    print(f"\nDevice: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset = create_datasets(
        n_train_puzzles=n_train,
        n_test_puzzles=n_test,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    # Create model with ShuffleNetV2_x0.5
    backbone_name = "shufflenet_v2_x0_5"
    print(f"\nCreating model with {backbone_name} backbone...")
    model = FastBackboneModel(
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Feature dimension: {model.feature_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Random baseline for 9 cells = 11.1%
    random_baseline = 1.0 / NUM_CELLS
    print(f"Random baseline (cell): {random_baseline:.1%}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create optimizer
    param_groups = model.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 80)
    epoch_times: list[float] = []
    history: dict[str, list[float]] = {
        "train_pos_loss": [],
        "train_rot_loss": [],
        "train_cell_acc": [],
        "train_rot_acc": [],
        "test_cell_acc": [],
        "test_rot_acc": [],
    }

    total_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        # Evaluate on test set every epoch
        test_metrics = compute_metrics(model, test_dataset, device)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        history["train_pos_loss"].append(train_metrics["position_loss"])
        history["train_rot_loss"].append(train_metrics["rotation_loss"])
        history["train_cell_acc"].append(train_metrics["cell_accuracy"])
        history["train_rot_acc"].append(train_metrics["rotation_accuracy"])
        history["test_cell_acc"].append(test_metrics["cell_accuracy"])
        history["test_rot_acc"].append(test_metrics["rotation_accuracy"])

        print(
            f"Epoch {epoch + 1:2d}/{epochs}: "
            f"pos_loss={train_metrics['position_loss']:.4f}, "
            f"rot_loss={train_metrics['rotation_loss']:.4f}, "
            f"train_cell={train_metrics['cell_accuracy']:.1%}, "
            f"test_cell={test_metrics['cell_accuracy']:.1%}, "
            f"train_rot={train_metrics['rotation_accuracy']:.1%}, "
            f"test_rot={test_metrics['rotation_accuracy']:.1%}, "
            f"time={epoch_time:.1f}s"
        )

    total_time = time.time() - total_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    print("-" * 80)
    print(f"\nTraining completed in {total_time:.1f}s ({avg_epoch_time:.1f}s/epoch)")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} ({NUM_CELLS} cells)")
    print(f"Random baseline: {random_baseline:.1%}")
    print(f"Final train cell accuracy: {history['train_cell_acc'][-1]:.1%}")
    print(f"Final test cell accuracy: {history['test_cell_acc'][-1]:.1%}")
    print(f"Final train rotation accuracy: {history['train_rot_acc'][-1]:.1%}")
    print(f"Final test rotation accuracy: {history['test_rot_acc'][-1]:.1%}")

    # Check if model is learning (above random baseline)
    if history["test_cell_acc"][-1] > random_baseline:
        print(
            f"\nSUCCESS: Model is learning! "
            f"Test accuracy {history['test_cell_acc'][-1]:.1%} > "
            f"random baseline {random_baseline:.1%}"
        )
    else:
        print(
            f"\nWARNING: Model not learning! "
            f"Test accuracy {history['test_cell_acc'][-1]:.1%} <= "
            f"random baseline {random_baseline:.1%}"
        )

    # Compile results
    results = {
        "experiment": "exp16_3x3_grid",
        "backbone": backbone_name,
        "grid_size": GRID_SIZE,
        "num_cells": NUM_CELLS,
        "feature_dim": model.feature_dim,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs": epochs,
        "n_train_puzzles": n_train,
        "n_test_puzzles": n_test,
        "batch_size": batch_size,
        "piece_size": piece_size,
        "puzzle_size": puzzle_size,
        "total_training_time": total_time,
        "avg_epoch_time": avg_epoch_time,
        "epoch_times": epoch_times,
        "random_baseline": random_baseline,
        "final_train_pos_loss": history["train_pos_loss"][-1],
        "final_train_rot_loss": history["train_rot_loss"][-1],
        "final_train_cell_acc": history["train_cell_acc"][-1],
        "final_train_rot_acc": history["train_rot_acc"][-1],
        "final_test_cell_acc": history["test_cell_acc"][-1],
        "final_test_rot_acc": history["test_rot_acc"][-1],
        "history": history,
    }

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Collect predictions for visualization
    pred_data = collect_predictions(model, test_dataset, device)

    # Save prediction grid (scatter plot + confusion matrices)
    save_prediction_grid(
        predictions=pred_data["predictions"],
        targets=pred_data["targets"],
        pred_cells=pred_data["pred_cells"],
        true_cells=pred_data["true_cells"],
        pred_rotations=pred_data["pred_rotations"],
        true_rotations=pred_data["true_rotations"],
        output_path=output_dir / "test_predictions.png",
    )
    print("Saved test_predictions.png")

    # Save training curves
    save_training_curves(
        history=history,
        output_path=output_dir / "training_curves.png",
    )
    print("Saved training_curves.png")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp16: 3x3 grid position prediction with ShuffleNetV2")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--n-train", type=int, default=500, help="Training puzzles")
    parser.add_argument("--n-test", type=int, default=100, help="Test puzzles")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--piece-size", type=int, default=128, help="Piece image size")
    parser.add_argument("--puzzle-size", type=int, default=256, help="Puzzle image size")
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
    )
