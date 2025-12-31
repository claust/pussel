#!/usr/bin/env python3
"""CUDA-optimized training script with mixed precision for RunPod.

This script is optimized for NVIDIA GPUs with:
- Automatic Mixed Precision (AMP) for faster training
- Larger batch sizes
- Multi-worker data loading
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dataset import GRID_SIZE, NUM_CELLS, RealisticPieceTestDataset, create_datasets
from model import FastBackboneModel, count_parameters
from visualize import save_prediction_grid, save_training_curves


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_epoch(
    model: FastBackboneModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch with mixed precision."""
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

        # Forward pass with autocast for mixed precision
        with autocast(device_type="cuda"):
            preds, rotation_logits, _ = model(pieces, puzzles)
            position_loss = F.mse_loss(preds, targets)
            rotation_loss = F.cross_entropy(rotation_logits, rotations)
            loss = position_weight * position_loss + rotation_weight * rotation_loss

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        batch_size = targets.size(0)
        total_position_loss += position_loss.item() * batch_size
        total_rotation_loss += rotation_loss.item() * batch_size
        total_samples += batch_size

        # Cell accuracy
        with torch.no_grad():
            pred_cells = model.predict_cell(pieces, puzzles, grid_size=GRID_SIZE)
            total_cell_correct += (pred_cells == cells).sum().item()
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
    dataset: RealisticPieceTestDataset,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Compute test metrics."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_predictions = []
    all_targets = []
    all_pred_cells = []
    all_true_cells = []
    all_pred_rotations = []
    all_true_rotations = []

    with torch.no_grad():
        for pieces, puzzles, targets, cells, rotations in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)

            with autocast(device_type="cuda"):
                preds, rotation_logits, _ = model(pieces, puzzles)

            pred_cells = model.predict_cell(pieces, puzzles, grid_size=GRID_SIZE)
            pred_rotations = rotation_logits.argmax(dim=1)

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_pred_cells.extend(pred_cells.cpu().numpy().tolist())
            all_true_cells.extend(cells.numpy().tolist())
            all_pred_rotations.extend(pred_rotations.cpu().numpy().tolist())
            all_true_rotations.extend(rotations.numpy().tolist())

    cell_acc = sum(p == t for p, t in zip(all_pred_cells, all_true_cells)) / len(all_pred_cells)
    rot_acc = sum(p == t for p, t in zip(all_pred_rotations, all_true_rotations)) / len(all_pred_rotations)

    return {
        "cell_accuracy": cell_acc,
        "rotation_accuracy": rot_acc,
        "predictions": all_predictions,
        "targets": all_targets,
        "pred_cells": all_pred_cells,
        "true_cells": all_true_cells,
        "pred_rotations": all_pred_rotations,
        "true_rotations": all_true_rotations,
    }


def main(
    epochs: int = 50,
    n_train: int = 10000,
    n_test: int = 1000,
    batch_size: int = 128,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    piece_size: int = 128,
    puzzle_size: int = 256,
    dataset_root: Path | str | None = None,
    num_workers: int = 8,
) -> dict[str, Any]:
    """Run training with CUDA optimizations."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("EXPERIMENT 20: CUDA TRAINING WITH MIXED PRECISION")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Training puzzles: {n_train}")
    print(f"Test puzzles: {n_test}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone LR: {backbone_lr}")
    print(f"Head LR: {head_lr}")
    print(f"Num workers: {num_workers}")

    device = get_device()
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create datasets
    print("\nLoading datasets...")
    create_kwargs: dict[str, Any] = {
        "n_train_puzzles": n_train,
        "n_test_puzzles": n_test,
        "piece_size": piece_size,
        "puzzle_size": puzzle_size,
    }
    if dataset_root is not None:
        create_kwargs["dataset_root"] = dataset_root
    train_dataset, test_dataset = create_datasets(**create_kwargs)

    # Create model
    backbone_name = "shufflenet_v2_x0_5"
    print(f"\nCreating model with {backbone_name} backbone...")
    model = FastBackboneModel(
        backbone_name=backbone_name,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create data loaders with multiple workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # Optimizer with differential learning rates (using model's method)
    param_groups = model.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=0.01,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Gradient scaler for mixed precision
    scaler = GradScaler("cuda")

    # Training history
    history: dict[str, list[float]] = {
        "train_pos_loss": [],
        "train_rot_loss": [],
        "train_cell_acc": [],
        "train_rot_acc": [],
        "test_cell_acc": [],
        "test_rot_acc": [],
    }

    best_test_acc = 0.0
    best_epoch = 0

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)

        # Test
        test_metrics = compute_metrics(model, test_dataset, device, batch_size=batch_size * 2)

        epoch_time = time.time() - start_time

        # Record history
        history["train_pos_loss"].append(train_metrics["position_loss"])
        history["train_rot_loss"].append(train_metrics["rotation_loss"])
        history["train_cell_acc"].append(train_metrics["cell_accuracy"])
        history["train_rot_acc"].append(train_metrics["rotation_accuracy"])
        history["test_cell_acc"].append(test_metrics["cell_accuracy"])
        history["test_rot_acc"].append(test_metrics["rotation_accuracy"])

        # Check for best model
        is_best = test_metrics["cell_accuracy"] > best_test_acc
        if is_best:
            best_test_acc = test_metrics["cell_accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "checkpoint_best.pt")

        # Save last checkpoint
        torch.save(model.state_dict(), output_dir / "checkpoint_last.pt")

        # Print progress
        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"pos_loss={train_metrics['position_loss']:.4f}, "
            f"rot_loss={train_metrics['rotation_loss']:.4f}, "
            f"train_cell={train_metrics['cell_accuracy']:.1%}, "
            f"test_cell={test_metrics['cell_accuracy']:.1%}, "
            f"train_rot={train_metrics['rotation_accuracy']:.1%}, "
            f"test_rot={test_metrics['rotation_accuracy']:.1%}, "
            f"time={epoch_time:.1f}s" + (" *" if is_best else "")
        )

    print("-" * 80)
    print(f"\nBest model: epoch {best_epoch} with test cell acc {best_test_acc:.1%}")

    # Final evaluation
    model.load_state_dict(torch.load(output_dir / "checkpoint_best.pt"))
    final_metrics = compute_metrics(model, test_dataset, device, batch_size=batch_size * 2)

    # Save results
    results = {
        "grid_size": GRID_SIZE,
        "num_cells": NUM_CELLS,
        "epochs": epochs,
        "n_train": n_train,
        "n_test": n_test,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "final_test_cell_accuracy": final_metrics["cell_accuracy"],
        "final_test_rotation_accuracy": final_metrics["rotation_accuracy"],
        "history": history,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    save_prediction_grid(
        predictions=final_metrics["predictions"],
        targets=final_metrics["targets"],
        pred_cells=final_metrics["pred_cells"],
        true_cells=final_metrics["true_cells"],
        pred_rotations=final_metrics["pred_rotations"],
        true_rotations=final_metrics["true_rotations"],
        output_path=output_dir / "test_predictions.png",
    )
    print("Saved test_predictions.png")

    save_training_curves(history, output_dir / "training_curves.png")
    print("Saved training_curves.png")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test Cell Accuracy: {final_metrics['cell_accuracy']:.1%}")
    print(f"Test Rotation Accuracy: {final_metrics['rotation_accuracy']:.1%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA training for exp20")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--dataset-root", type=str, default="datasets/realistic_4x4_20k")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        dataset_root=Path(args.dataset_root),
        num_workers=args.num_workers,
    )
