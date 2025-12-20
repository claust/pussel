"""Training script for high-resolution coarse regression experiment.

This experiment tests whether increasing puzzle resolution from 256x256 to
512x512 improves cross-puzzle generalization for 2x2 quadrant prediction.

Building on exp7's success with spatial correlation (67% test accuracy),
this experiment provides more detail for template matching.

Key change from exp7:
- Puzzle size: 256x256 -> 512x512
- Spatial feature map: 8x8 -> 16x16 (4x more spatial locations)

Metrics tracked:
- MSE Loss: Regression quality
- Quadrant Accuracy: Correct quadrant prediction (random = 25%)
- Distance Error: Euclidean distance between predicted and true center
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import QuadrantDataset, create_datasets
from .model import DualInputRegressorWithCorrelation, count_parameters
from .visualize import save_prediction_grid, save_training_curves


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_metrics(
    model: DualInputRegressorWithCorrelation,
    dataset: QuadrantDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compute evaluation metrics on a dataset.

    Args:
        model: Trained model.
        dataset: Dataset to evaluate.
        device: Computation device.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with metrics.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: list[tuple[float, float]] = []
    all_targets: list[tuple[float, float]] = []
    all_pred_quadrants: list[int] = []
    all_true_quadrants: list[int] = []
    total_mse = 0.0
    n_samples = 0

    with torch.no_grad():
        for pieces, puzzles, targets, quadrants in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)
            targets = targets.to(device)

            # Forward pass (new model returns position and attention map)
            preds, _ = model(pieces, puzzles)

            # MSE loss
            mse = F.mse_loss(preds, targets, reduction="sum")
            total_mse += mse.item()
            n_samples += targets.size(0)

            # Predicted quadrants
            pred_quadrants = model.predict_quadrant(pieces, puzzles)

            # Store for analysis
            for i in range(preds.size(0)):
                all_preds.append((preds[i, 0].item(), preds[i, 1].item()))
                all_targets.append((targets[i, 0].item(), targets[i, 1].item()))
                all_pred_quadrants.append(int(pred_quadrants[i].item()))
                all_true_quadrants.append(int(quadrants[i].item()))

    # Calculate metrics
    mse_loss = total_mse / n_samples

    # Quadrant accuracy
    correct = sum(p == t for p, t in zip(all_pred_quadrants, all_true_quadrants))
    quadrant_acc = correct / len(all_true_quadrants)

    # Distance errors
    distances = []
    for (px, py), (tx, ty) in zip(all_preds, all_targets):
        dist = ((px - tx) ** 2 + (py - ty) ** 2) ** 0.5
        distances.append(dist)

    mean_dist = sum(distances) / len(distances)
    max_dist = max(distances)

    return {
        "mse_loss": mse_loss,
        "quadrant_accuracy": quadrant_acc,
        "mean_distance": mean_dist,
        "max_distance": max_dist,
        "correct": correct,
        "total": len(all_true_quadrants),
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_pred_quadrants": all_pred_quadrants,
        "all_true_quadrants": all_true_quadrants,
    }


def train_epoch(
    model: DualInputRegressorWithCorrelation,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        loader: Training data loader.
        optimizer: Optimizer.
        device: Computation device.

    Returns:
        Dictionary with epoch metrics.
    """
    model.train()
    total_mse = 0.0
    total_correct = 0
    total_samples = 0

    for pieces, puzzles, targets, quadrants in loader:
        pieces = pieces.to(device)
        puzzles = puzzles.to(device)
        targets = targets.to(device)
        quadrants = quadrants.to(device)

        optimizer.zero_grad()

        # Forward pass (new model returns position and attention map)
        preds, _ = model(pieces, puzzles)
        loss = F.mse_loss(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_mse += loss.item() * targets.size(0)
        total_samples += targets.size(0)

        # Quadrant accuracy
        pred_quadrants = model.predict_quadrant(pieces, puzzles)
        total_correct += (pred_quadrants == quadrants).sum().item()

    return {
        "mse_loss": total_mse / total_samples,
        "quadrant_accuracy": total_correct / total_samples,
    }


_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def run_experiment(
    n_train_puzzles: int = 800,
    n_test_puzzles: int = 200,
    epochs: int = 50,
    batch_size: int = 32,  # Smaller batch size for higher resolution
    lr: float = 1e-3,
    piece_size: int = 128,
    puzzle_size: int = 512,  # High resolution
    freeze_backbone: bool = True,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the high-resolution coarse regression experiment.

    Args:
        n_train_puzzles: Number of puzzles for training.
        n_test_puzzles: Number of puzzles for testing.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        freeze_backbone: If True, freeze MobileNetV3 (Phase 1).
        output_dir: Directory for outputs.

    Returns:
        Dictionary with all experiment results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 8: HIGH RESOLUTION COARSE REGRESSION (512x512 puzzle)")
    print("=" * 70)
    print(f"Training puzzles: {n_train_puzzles}")
    print(f"Test puzzles: {n_test_puzzles}")
    print(f"Piece size: {piece_size}x{piece_size}")
    print(f"Puzzle size: {puzzle_size}x{puzzle_size} (HIGH RES)")
    print(f"Backbone frozen: {freeze_backbone} (Phase {'1' if freeze_backbone else '2'})")

    device = get_device()
    print(f"Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset = create_datasets(
        n_train_puzzles=n_train_puzzles,
        n_test_puzzles=n_test_puzzles,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    # Create model with spatial correlation
    model = DualInputRegressorWithCorrelation(
        freeze_backbone=freeze_backbone,
    ).to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\nModel: DualInputRegressorWithCorrelation (MobileNetV3-Small + Spatial Correlation)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Note the spatial feature map size
    print(f"\nWith {puzzle_size}x{puzzle_size} puzzle input:")
    print(f"  Spatial feature map: {puzzle_size // 32}x{puzzle_size // 32} = {(puzzle_size // 32) ** 2} locations")
    print("  (vs 8x8 = 64 locations with 256x256 input)")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS works best with 0 workers
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Create output directory for checkpoints
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    history: dict[str, list[float]] = {
        "train_mse": [],
        "train_acc": [],
        "test_mse": [],
        "test_acc": [],
        "test_dist": [],
    }

    best_test_acc = 0.0
    best_epoch = 0

    eval_every = 10  # Evaluate on test set every N epochs

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        history["train_mse"].append(train_metrics["mse_loss"])
        history["train_acc"].append(train_metrics["quadrant_accuracy"])

        # Evaluate on test set periodically
        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == epochs:
            test_metrics = compute_metrics(model, test_dataset, device)
            history["test_mse"].append(test_metrics["mse_loss"])
            history["test_acc"].append(test_metrics["quadrant_accuracy"])
            history["test_dist"].append(test_metrics["mean_distance"])

            # Track best and save checkpoint
            if test_metrics["quadrant_accuracy"] > best_test_acc:
                best_test_acc = test_metrics["quadrant_accuracy"]
                best_epoch = epoch + 1
                # Save best model
                best_model_path = output_dir / "model_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "test_accuracy": best_test_acc,
                        "test_mse": test_metrics["mse_loss"],
                    },
                    best_model_path,
                )

            # Save latest checkpoint (overwritten each time)
            latest_path = output_dir / "checkpoint_latest.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_mse": train_metrics["mse_loss"],
                    "train_accuracy": train_metrics["quadrant_accuracy"],
                    "test_mse": test_metrics["mse_loss"],
                    "test_accuracy": test_metrics["quadrant_accuracy"],
                },
                latest_path,
            )

            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"train_mse={train_metrics['mse_loss']:.4f}, "
                f"train_acc={train_metrics['quadrant_accuracy']:.1%}, "
                f"test_mse={test_metrics['mse_loss']:.4f}, "
                f"test_acc={test_metrics['quadrant_accuracy']:.1%}, "
                f"dist={test_metrics['mean_distance']:.3f}"
            )
        else:
            # Print training metrics every epoch
            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"train_mse={train_metrics['mse_loss']:.4f}, "
                f"train_acc={train_metrics['quadrant_accuracy']:.1%}"
            )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    train_final = compute_metrics(model, train_dataset, device)
    test_final = compute_metrics(model, test_dataset, device)

    print(f"\nTraining set ({n_train_puzzles} puzzles, {len(train_dataset)} samples):")
    print(f"  MSE Loss: {train_final['mse_loss']:.4f}")
    print(f"  Quadrant Accuracy: {train_final['quadrant_accuracy']:.1%}")
    print(f"  Mean Distance: {train_final['mean_distance']:.4f}")

    print(f"\nTest set ({n_test_puzzles} puzzles, {len(test_dataset)} samples) - HELD OUT:")
    print(f"  MSE Loss: {test_final['mse_loss']:.4f}")
    print(f"  Quadrant Accuracy: {test_final['quadrant_accuracy']:.1%}")
    print(f"  Mean Distance: {test_final['mean_distance']:.4f}")

    random_baseline = 0.25  # Random chance for 4 quadrants
    print(f"\nRandom baseline accuracy: {random_baseline:.1%}")
    print(f"Test vs random: {test_final['quadrant_accuracy'] / random_baseline:.2f}x")
    print(f"Best test accuracy: {best_test_acc:.1%} (epoch {best_epoch})")

    # Success criteria from README
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 50)

    train_mse_ok = train_final["mse_loss"] < 0.02
    test_mse_ok = test_final["mse_loss"] < 0.10
    test_acc_ok = test_final["quadrant_accuracy"] > 0.70

    print(f"Training MSE < 0.02:    {'PASS' if train_mse_ok else 'FAIL'} ({train_final['mse_loss']:.4f})")
    print(f"Test MSE < 0.10:        {'PASS' if test_mse_ok else 'FAIL'} ({test_final['mse_loss']:.4f})")
    print(f"Test Accuracy > 70%:    {'PASS' if test_acc_ok else 'FAIL'} ({test_final['quadrant_accuracy']:.1%})")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualizations
    save_prediction_grid(
        test_final["all_preds"],
        test_final["all_targets"],
        test_final["all_pred_quadrants"],
        test_final["all_true_quadrants"],
        output_dir / "test_predictions.png",
    )
    print(f"\nSaved predictions to {output_dir / 'test_predictions.png'}")

    save_training_curves(history, output_dir / "training_curves.png")
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")

    # Save model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Compile results
    results: dict[str, Any] = {
        "experiment": "exp8_high_res_correlation",
        "phase": 1 if freeze_backbone else 2,
        "n_train_puzzles": n_train_puzzles,
        "n_test_puzzles": n_test_puzzles,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "piece_size": piece_size,
        "puzzle_size": puzzle_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "freeze_backbone": freeze_backbone,
        "training_time_seconds": training_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_mse": train_final["mse_loss"],
        "train_accuracy": train_final["quadrant_accuracy"],
        "train_mean_distance": train_final["mean_distance"],
        "test_mse": test_final["mse_loss"],
        "test_accuracy": test_final["quadrant_accuracy"],
        "test_mean_distance": test_final["mean_distance"],
        "best_test_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "random_baseline": random_baseline,
        "test_vs_random": test_final["quadrant_accuracy"] / random_baseline,
        "criteria": {
            "train_mse_pass": train_mse_ok,
            "test_mse_pass": test_mse_ok,
            "test_accuracy_pass": test_acc_ok,
        },
    }

    # Save results to JSON
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    return results


def main(
    epochs: int = 50,
    n_train: int = 800,
    n_test: int = 200,
    batch_size: int = 32,
    piece_size: int = 128,
    puzzle_size: int = 512,
) -> dict[str, Any]:
    """Run the experiment with given parameters.

    Args:
        epochs: Number of training epochs.
        n_train: Number of training puzzles.
        n_test: Number of test puzzles.
        batch_size: Training batch size.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.

    Returns:
        Experiment results.
    """
    results = run_experiment(
        n_train_puzzles=n_train,
        n_test_puzzles=n_test,
        epochs=epochs,
        batch_size=batch_size,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        freeze_backbone=True,  # Phase 1
    )

    # Print interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    test_acc = results["test_accuracy"]
    random_baseline = results["random_baseline"]

    # Compare to exp7's 67% result
    exp7_baseline = 0.67

    if test_acc < random_baseline * 1.2:  # Less than 30%
        print("Test accuracy near random chance (< 30%).")
        print("CONCLUSION: Model fails to generalize to new puzzles.")
        print("Higher resolution did not help.")
    elif test_acc < 0.50:  # 30-50%
        print("Test accuracy moderately above random (30-50%).")
        print("CONCLUSION: Weak generalization - worse than exp7 (67%).")
        print("Higher resolution may have hurt by providing too much detail.")
    elif test_acc < exp7_baseline:  # 50-67%
        print(f"Test accuracy ({test_acc:.1%}) below exp7 baseline (67%).")
        print("CONCLUSION: Higher resolution did NOT improve results.")
    elif test_acc < 0.70:  # 67-70%
        print(f"Test accuracy ({test_acc:.1%}) similar to exp7 (67%).")
        print("CONCLUSION: Higher resolution provides marginal improvement.")
    else:  # 70%+
        print(f"Test accuracy high ({test_acc:.1%})!")
        print(f"CONCLUSION: Higher resolution helps! Improved from 67% to {test_acc:.1%}.")
        print("The 512x512 puzzle provides better detail for template matching.")
        print("Next: Try fine-tuning backbone or increasing grid resolution.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-resolution coarse regression (512x512 puzzle)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=800,
        help="Number of training puzzles",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=200,
        help="Number of test puzzles",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--piece-size",
        type=int,
        default=128,
        help="Size of piece images",
    )
    parser.add_argument(
        "--puzzle-size",
        type=int,
        default=512,
        help="Size of puzzle images",
    )
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
    )
