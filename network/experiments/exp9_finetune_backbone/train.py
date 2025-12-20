"""Training script for backbone fine-tuning experiment.

This experiment tests whether fine-tuning the MobileNetV3-Small backbone
improves cross-puzzle generalization beyond the 67% achieved with frozen
features in exp7.

Phase 2 approach:
- Unfreeze backbone with lower learning rate
- Use differential LRs: backbone (1e-5 to 1e-4) vs heads (1e-3)
- Optional gradual unfreezing: start with last layers, progressively unfreeze
- Added regularization (dropout, weight decay) to prevent overfitting

Target: Test accuracy > 70% (ideally 75-80%)
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

            # Forward pass
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

        # Forward pass
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


def run_experiment(  # noqa: C901
    n_train_puzzles: int = 800,
    n_test_puzzles: int = 200,
    epochs: int = 100,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    piece_size: int = 128,
    puzzle_size: int = 256,
    gradual_unfreeze: bool = False,
    unfreeze_schedule: list[tuple[int, list[int]]] | None = None,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the backbone fine-tuning experiment.

    Args:
        n_train_puzzles: Number of puzzles for training.
        n_test_puzzles: Number of puzzles for testing.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone parameters.
        head_lr: Learning rate for correlation and refinement heads.
        weight_decay: Weight decay for regularization.
        dropout: Dropout rate.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        gradual_unfreeze: If True, use gradual unfreezing schedule.
        unfreeze_schedule: List of (epoch, [layer_indices]) for gradual unfreezing.
        output_dir: Directory for outputs.

    Returns:
        Dictionary with all experiment results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 9: FINE-TUNE BACKBONE (PHASE 2)")
    print("=" * 70)
    print(f"Training puzzles: {n_train_puzzles}")
    print(f"Test puzzles: {n_test_puzzles}")
    print(f"Piece size: {piece_size}x{piece_size}")
    print(f"Puzzle size: {puzzle_size}x{puzzle_size}")
    print(f"Backbone LR: {backbone_lr}")
    print(f"Head LR: {head_lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Dropout: {dropout}")
    print(f"Gradual unfreeze: {gradual_unfreeze}")

    device = get_device()
    print(f"Device: {device}")

    # Create datasets (same as exp7)
    print("\nLoading datasets...")
    train_dataset, test_dataset = create_datasets(
        n_train_puzzles=n_train_puzzles,
        n_test_puzzles=n_test_puzzles,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    # Create model with dropout but initially frozen (for gradual) or unfrozen
    if gradual_unfreeze:
        # Start with frozen backbone, will unfreeze according to schedule
        model = DualInputRegressorWithCorrelation(
            freeze_backbone=True,
            dropout=dropout,
        ).to(device)
        # Default schedule: unfreeze last 3 layers at epoch 10, more at epoch 30
        if unfreeze_schedule is None:
            unfreeze_schedule = [
                (1, [10, 11, 12]),  # Final blocks at start
                (20, [7, 8, 9]),  # Middle blocks after 20 epochs
            ]
        print(f"Unfreeze schedule: {unfreeze_schedule}")
    else:
        # Fully unfrozen from the start
        model = DualInputRegressorWithCorrelation(
            freeze_backbone=False,
            dropout=dropout,
        ).to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\nModel: DualInputRegressorWithCorrelation (MobileNetV3-Small)")
    print(f"Total parameters: {total_params:,}")
    print(f"Initial trainable parameters: {trainable_params:,}")

    # Show backbone layer info
    print("\nBackbone layers:")
    for info in model.get_layer_info()[-5:]:  # Show last 5 layers
        status = "TRAINABLE" if info["trainable"] else "frozen"
        print(f"  Layer {info['index']:2d}: {info['name']:20s} " f"({info['params']:,} params) - {status}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS works best with 0 workers
    )

    # Create optimizer with differential learning rates
    param_groups = model.get_parameter_groups(
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Print optimizer info
    print("\nOptimizer parameter groups:")
    for group in param_groups:
        param_count = sum(p.numel() for p in group["params"])
        print(f"  {group['name']}: {param_count:,} params, LR={group['lr']}")

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
        # Check for gradual unfreezing
        if gradual_unfreeze and unfreeze_schedule:
            for unfreeze_epoch, layer_indices in unfreeze_schedule:
                if epoch + 1 == unfreeze_epoch:
                    print(f"\n>>> Unfreezing layers {layer_indices} at epoch {epoch + 1}")
                    model.unfreeze_layers(layer_indices)

                    # Rebuild optimizer with new parameters
                    param_groups = model.get_parameter_groups(
                        backbone_lr=backbone_lr,
                        head_lr=head_lr,
                        weight_decay=weight_decay,
                    )
                    optimizer = torch.optim.AdamW(param_groups)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch)

                    new_trainable = count_parameters(model)
                    print(f">>> New trainable parameters: {new_trainable:,}")

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

            # Save latest checkpoint
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

    # Success criteria
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA CHECK (vs exp7 frozen backbone)")
    print("=" * 50)

    exp7_test_acc = 0.67  # exp7's best test accuracy
    test_acc_improved = test_final["quadrant_accuracy"] > exp7_test_acc
    test_acc_target = test_final["quadrant_accuracy"] > 0.70
    test_acc_stretch = test_final["quadrant_accuracy"] > 0.75

    print(f"Exp7 frozen backbone: {exp7_test_acc:.1%}")
    print(f"Exp9 fine-tuned:      {test_final['quadrant_accuracy']:.1%}")
    print(f"Improvement:          {test_final['quadrant_accuracy'] - exp7_test_acc:+.1%}")
    print()
    print(f"Beat exp7 (>67%):    {'PASS' if test_acc_improved else 'FAIL'}")
    print(f"Target (>70%):        {'PASS' if test_acc_target else 'FAIL'}")
    print(f"Stretch (>75%):       {'PASS' if test_acc_stretch else 'FAIL'}")

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
    final_trainable = count_parameters(model)
    results: dict[str, Any] = {
        "experiment": "exp9_finetune_backbone",
        "phase": 2,
        "n_train_puzzles": n_train_puzzles,
        "n_test_puzzles": n_test_puzzles,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "piece_size": piece_size,
        "puzzle_size": puzzle_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "gradual_unfreeze": gradual_unfreeze,
        "training_time_seconds": training_time,
        "total_params": total_params,
        "trainable_params": final_trainable,
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
        "exp7_comparison": {
            "exp7_test_accuracy": exp7_test_acc,
            "improvement": test_final["quadrant_accuracy"] - exp7_test_acc,
            "beat_exp7": test_acc_improved,
            "reached_70_target": test_acc_target,
            "reached_75_stretch": test_acc_stretch,
        },
    }

    # Save results to JSON
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    return results


def main(
    epochs: int = 100,
    n_train: int = 800,
    n_test: int = 200,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    piece_size: int = 128,
    puzzle_size: int = 256,
    gradual_unfreeze: bool = False,
) -> dict[str, Any]:
    """Run the experiment with given parameters.

    Args:
        epochs: Number of training epochs.
        n_train: Number of training puzzles.
        n_test: Number of test puzzles.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone.
        head_lr: Learning rate for heads.
        weight_decay: Weight decay.
        dropout: Dropout rate.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        gradual_unfreeze: Use gradual unfreezing schedule.

    Returns:
        Experiment results.
    """
    results = run_experiment(
        n_train_puzzles=n_train,
        n_test_puzzles=n_test,
        epochs=epochs,
        batch_size=batch_size,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        dropout=dropout,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        gradual_unfreeze=gradual_unfreeze,
    )

    # Print interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    test_acc = results["test_accuracy"]
    exp7_acc = results["exp7_comparison"]["exp7_test_accuracy"]

    if test_acc < exp7_acc:
        print(f"Test accuracy ({test_acc:.1%}) is LOWER than exp7 ({exp7_acc:.1%}).")
        print("Fine-tuning hurt performance - possible overfitting.")
        print("Consider: more regularization, lower backbone LR, gradual unfreezing.")
    elif test_acc < 0.70:
        print(f"Test accuracy ({test_acc:.1%}) improved over exp7 but below 70% target.")
        print("Fine-tuning helped but more work needed.")
        print("Consider: longer training, different LR schedule, architecture changes.")
    elif test_acc < 0.75:
        print(f"Test accuracy ({test_acc:.1%}) exceeded 70% target!")
        print("Fine-tuning successfully improved generalization.")
        print("Next: try finer grids (3x3, 4x4) or different backbones.")
    else:
        print(f"Test accuracy ({test_acc:.1%}) exceeded 75% stretch goal!")
        print("Excellent result - fine-tuning very effective.")
        print("Ready to proceed to finer-grained tasks.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backbone fine-tuning experiment (Phase 2)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
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
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help="Learning rate for backbone (default: 1e-4)",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="Learning rate for heads (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
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
        default=256,
        help="Size of puzzle images (default: 256, same as exp7)",
    )
    parser.add_argument(
        "--gradual-unfreeze",
        action="store_true",
        help="Use gradual unfreezing (start with last layers, add more over time)",
    )
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
        gradual_unfreeze=args.gradual_unfreeze,
    )
