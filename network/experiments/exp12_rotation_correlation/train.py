"""Training script for Rotation Correlation experiment.

This experiment fixes the fundamental flaw in exp10/11:
- Exp10/11: Rotation head only saw piece features, ignoring puzzle
- Exp12: Rotation head COMPARES piece to puzzle for each rotation

Target: >85% test rotation accuracy, >90% test position accuracy
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import QuadrantAllRotationsDataset, create_datasets
from .model import DualInputRegressorWithRotationCorrelation, count_parameters
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
    model: DualInputRegressorWithRotationCorrelation,
    dataset: QuadrantAllRotationsDataset,
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
    all_pred_rotations: list[int] = []
    all_true_rotations: list[int] = []
    total_mse = 0.0
    total_rotation_correct = 0
    n_samples = 0

    with torch.no_grad():
        for pieces, puzzles, targets, quadrants, rotations in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)
            targets = targets.to(device)

            # Forward pass
            preds, rotation_logits, _ = model(pieces, puzzles)

            # MSE loss for position
            mse = F.mse_loss(preds, targets, reduction="sum")
            total_mse += mse.item()
            n_samples += targets.size(0)

            # Predicted quadrants
            pred_quadrants = model.predict_quadrant(pieces, puzzles)

            # Predicted rotations
            pred_rotations = rotation_logits.argmax(dim=1)
            total_rotation_correct += (pred_rotations == rotations.to(device)).sum().item()

            # Store for analysis
            for i in range(preds.size(0)):
                all_preds.append((preds[i, 0].item(), preds[i, 1].item()))
                all_targets.append((targets[i, 0].item(), targets[i, 1].item()))
                all_pred_quadrants.append(int(pred_quadrants[i].item()))
                all_true_quadrants.append(int(quadrants[i].item()))
                all_pred_rotations.append(int(pred_rotations[i].item()))
                all_true_rotations.append(int(rotations[i].item()))

    # Calculate metrics
    mse_loss = total_mse / n_samples

    # Quadrant accuracy
    correct = sum(p == t for p, t in zip(all_pred_quadrants, all_true_quadrants))
    quadrant_acc = correct / len(all_true_quadrants)

    # Rotation accuracy
    rotation_acc = total_rotation_correct / n_samples

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
        "rotation_accuracy": rotation_acc,
        "mean_distance": mean_dist,
        "max_distance": max_dist,
        "correct_quadrants": correct,
        "correct_rotations": total_rotation_correct,
        "total": len(all_true_quadrants),
        "all_preds": all_preds,
        "all_targets": all_targets,
        "all_pred_quadrants": all_pred_quadrants,
        "all_true_quadrants": all_true_quadrants,
        "all_pred_rotations": all_pred_rotations,
        "all_true_rotations": all_true_rotations,
    }


def train_epoch(
    model: DualInputRegressorWithRotationCorrelation,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        loader: Training data loader.
        optimizer: Optimizer.
        device: Computation device.
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.

    Returns:
        Dictionary with epoch metrics.
    """
    model.train()
    total_position_loss = 0.0
    total_rotation_loss = 0.0
    total_quadrant_correct = 0
    total_rotation_correct = 0
    total_samples = 0

    for pieces, puzzles, targets, quadrants, rotations in loader:
        pieces = pieces.to(device)
        puzzles = puzzles.to(device)
        targets = targets.to(device)
        quadrants = quadrants.to(device)
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

        # Quadrant accuracy
        pred_quadrants = model.predict_quadrant(pieces, puzzles)
        total_quadrant_correct += (pred_quadrants == quadrants).sum().item()

        # Rotation accuracy
        pred_rotations = rotation_logits.argmax(dim=1)
        total_rotation_correct += (pred_rotations == rotations).sum().item()

    return {
        "position_loss": total_position_loss / total_samples,
        "rotation_loss": total_rotation_loss / total_samples,
        "quadrant_accuracy": total_quadrant_correct / total_samples,
        "rotation_accuracy": total_rotation_correct / total_samples,
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
    rotation_dropout: float = 0.2,
    piece_size: int = 128,
    puzzle_size: int = 256,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the rotation correlation experiment.

    Args:
        n_train_puzzles: Number of puzzles for training.
        n_test_puzzles: Number of puzzles for testing.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone parameters.
        head_lr: Learning rate for heads.
        weight_decay: Weight decay for regularization.
        dropout: Dropout rate for position head.
        rotation_dropout: Dropout rate for rotation head.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.
        output_dir: Directory for outputs.

    Returns:
        Dictionary with all experiment results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 12: ROTATION CORRELATION")
    print("=" * 70)
    print("Key fix from exp10/11:")
    print("  - Rotation head now COMPARES piece to puzzle (not piece alone)")
    print("  - For each rotation, compute similarity between piece and puzzle region")
    print("  - Select rotation with highest match score")
    print("-" * 70)
    print(f"Training puzzles: {n_train_puzzles}")
    print(f"Test puzzles: {n_test_puzzles}")
    print(f"Piece size: {piece_size}x{piece_size}")
    print(f"Puzzle size: {puzzle_size}x{puzzle_size}")
    print(f"Backbone LR: {backbone_lr}")
    print(f"Head LR: {head_lr}")
    print(f"Weight decay: {weight_decay}")
    print(f"Dropout (position): {dropout}")
    print(f"Dropout (rotation): {rotation_dropout}")
    print(f"Position weight: {position_weight}")
    print(f"Rotation weight: {rotation_weight}")

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

    # Create model with rotation correlation
    model = DualInputRegressorWithRotationCorrelation(
        freeze_backbone=False,
        dropout=dropout,
        rotation_dropout=rotation_dropout,
    ).to(device)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\nModel: DualInputRegressorWithRotationCorrelation (MobileNetV3-Small)")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Count rotation correlation params
    rotation_params = sum(p.numel() for p in model.rotation_correlation.parameters())
    print(f"Rotation correlation module parameters: {rotation_params:,}")

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

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    history: dict[str, list[float]] = {
        "train_pos_loss": [],
        "train_rot_loss": [],
        "train_quad_acc": [],
        "train_rot_acc": [],
        "test_pos_loss": [],
        "test_quad_acc": [],
        "test_rot_acc": [],
        "test_dist": [],
    }

    best_test_combined = 0.0  # quadrant_acc + rotation_acc
    best_epoch = 0

    eval_every = 10  # Evaluate on test set every N epochs

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            position_weight=position_weight,
            rotation_weight=rotation_weight,
        )
        scheduler.step()

        history["train_pos_loss"].append(train_metrics["position_loss"])
        history["train_rot_loss"].append(train_metrics["rotation_loss"])
        history["train_quad_acc"].append(train_metrics["quadrant_accuracy"])
        history["train_rot_acc"].append(train_metrics["rotation_accuracy"])

        # Evaluate on test set periodically
        if (epoch + 1) % eval_every == 0 or epoch == 0 or (epoch + 1) == epochs:
            test_metrics = compute_metrics(model, test_dataset, device)
            history["test_pos_loss"].append(test_metrics["mse_loss"])
            history["test_quad_acc"].append(test_metrics["quadrant_accuracy"])
            history["test_rot_acc"].append(test_metrics["rotation_accuracy"])
            history["test_dist"].append(test_metrics["mean_distance"])

            # Track best (combined accuracy)
            combined = test_metrics["quadrant_accuracy"] + test_metrics["rotation_accuracy"]
            if combined > best_test_combined:
                best_test_combined = combined
                best_epoch = epoch + 1
                # Save best model
                best_model_path = output_dir / "model_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "quadrant_accuracy": test_metrics["quadrant_accuracy"],
                        "rotation_accuracy": test_metrics["rotation_accuracy"],
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
                    "train_metrics": train_metrics,
                    "test_metrics": {
                        "mse_loss": test_metrics["mse_loss"],
                        "quadrant_accuracy": test_metrics["quadrant_accuracy"],
                        "rotation_accuracy": test_metrics["rotation_accuracy"],
                    },
                },
                latest_path,
            )

            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"pos_loss={train_metrics['position_loss']:.4f}, "
                f"rot_loss={train_metrics['rotation_loss']:.4f}, "
                f"train_quad={train_metrics['quadrant_accuracy']:.1%}, "
                f"train_rot={train_metrics['rotation_accuracy']:.1%}, "
                f"test_quad={test_metrics['quadrant_accuracy']:.1%}, "
                f"test_rot={test_metrics['rotation_accuracy']:.1%}"
            )
        else:
            # Print training metrics every epoch
            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"pos_loss={train_metrics['position_loss']:.4f}, "
                f"rot_loss={train_metrics['rotation_loss']:.4f}, "
                f"train_quad={train_metrics['quadrant_accuracy']:.1%}, "
                f"train_rot={train_metrics['rotation_accuracy']:.1%}"
            )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    # For training evaluation, need to create a version with all rotations
    train_all_rot = QuadrantAllRotationsDataset(
        puzzle_ids=train_dataset.puzzle_ids,
        dataset_root=train_dataset.dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    train_final = compute_metrics(model, train_all_rot, device)
    test_final = compute_metrics(model, test_dataset, device)

    print(f"\nTraining set ({n_train_puzzles} puzzles, {len(train_all_rot)} samples):")
    print(f"  Position MSE: {train_final['mse_loss']:.4f}")
    print(f"  Quadrant Accuracy: {train_final['quadrant_accuracy']:.1%}")
    print(f"  Rotation Accuracy: {train_final['rotation_accuracy']:.1%}")
    print(f"  Mean Distance: {train_final['mean_distance']:.4f}")

    print(f"\nTest set ({n_test_puzzles} puzzles, {len(test_dataset)} samples):")
    print(f"  Position MSE: {test_final['mse_loss']:.4f}")
    print(f"  Quadrant Accuracy: {test_final['quadrant_accuracy']:.1%}")
    print(f"  Rotation Accuracy: {test_final['rotation_accuracy']:.1%}")
    print(f"  Mean Distance: {test_final['mean_distance']:.4f}")

    # Baselines
    random_quadrant = 0.25
    random_rotation = 0.25
    print(f"\nRandom baselines: quadrant={random_quadrant:.1%}, " f"rotation={random_rotation:.1%}")
    print(
        f"Test vs random: quadrant {test_final['quadrant_accuracy'] / random_quadrant:.2f}x, "
        f"rotation {test_final['rotation_accuracy'] / random_rotation:.2f}x"
    )

    # Comparison with previous experiments
    print("\n" + "=" * 50)
    print("COMPARISON WITH PREVIOUS EXPERIMENTS")
    print("=" * 50)
    exp10_quad = 0.784
    exp10_rot = 0.614
    exp11_quad = 0.729
    exp11_rot = 0.598

    print("Position (Quadrant Accuracy):")
    print(f"  Exp10: {exp10_quad:.1%}")
    print(f"  Exp11: {exp11_quad:.1%}")
    print(f"  Exp12: {test_final['quadrant_accuracy']:.1%}")
    quad_vs_exp11 = test_final["quadrant_accuracy"] - exp11_quad
    print(f"  vs Exp11: {quad_vs_exp11:+.1%}")

    print("\nRotation Accuracy:")
    print(f"  Exp10: {exp10_rot:.1%}")
    print(f"  Exp11: {exp11_rot:.1%}")
    print(f"  Exp12: {test_final['rotation_accuracy']:.1%}")
    rot_vs_exp11 = test_final["rotation_accuracy"] - exp11_rot
    print(f"  vs Exp11: {rot_vs_exp11:+.1%}")

    # Success criteria
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 50)

    quad_target = test_final["quadrant_accuracy"] >= 0.90
    rot_target = test_final["rotation_accuracy"] >= 0.85
    rot_stretch = test_final["rotation_accuracy"] >= 0.90

    print(f"Position target (>90%):    " f"{'PASS' if quad_target else 'FAIL'} ({test_final['quadrant_accuracy']:.1%})")
    print(f"Rotation target (>85%):    " f"{'PASS' if rot_target else 'FAIL'} ({test_final['rotation_accuracy']:.1%})")
    print(f"Rotation stretch (>90%):   " f"{'PASS' if rot_stretch else 'FAIL'} ({test_final['rotation_accuracy']:.1%})")

    # Analyze rotation confusion
    print("\n" + "=" * 50)
    print("ROTATION CONFUSION ANALYSIS")
    print("=" * 50)
    import numpy as np

    rot_confusion = np.zeros((4, 4), dtype=int)
    for pred, true in zip(test_final["all_pred_rotations"], test_final["all_true_rotations"]):
        rot_confusion[true, pred] += 1

    print("Rotation confusion matrix:")
    print("         Pred: 0°   90°  180°  270°")
    for i, label in enumerate(["  0°", " 90°", "180°", "270°"]):
        row = rot_confusion[i]
        print(f"True {label}: {row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d}")

    # Calculate specific confusion rates
    total_0 = rot_confusion[0].sum()
    total_180 = rot_confusion[2].sum()
    total_90 = rot_confusion[1].sum()
    total_270 = rot_confusion[3].sum()

    if total_0 > 0:
        conf_0_180 = rot_confusion[0, 2] / total_0
        print(f"\n0° confused as 180°: {conf_0_180:.1%}")
    if total_180 > 0:
        conf_180_0 = rot_confusion[2, 0] / total_180
        print(f"180° confused as 0°: {conf_180_0:.1%}")
    if total_90 > 0:
        conf_90_270 = rot_confusion[1, 3] / total_90
        print(f"90° confused as 270°: {conf_90_270:.1%}")
    if total_270 > 0:
        conf_270_90 = rot_confusion[3, 1] / total_270
        print(f"270° confused as 90°: {conf_270_90:.1%}")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualizations
    save_prediction_grid(
        test_final["all_preds"],
        test_final["all_targets"],
        test_final["all_pred_quadrants"],
        test_final["all_true_quadrants"],
        test_final["all_pred_rotations"],
        test_final["all_true_rotations"],
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
        "experiment": "exp12_rotation_correlation",
        "n_train_puzzles": n_train_puzzles,
        "n_test_puzzles": n_test_puzzles,
        "train_samples_per_epoch": len(train_dataset),
        "test_samples": len(test_dataset),
        "piece_size": piece_size,
        "puzzle_size": puzzle_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "rotation_dropout": rotation_dropout,
        "position_weight": position_weight,
        "rotation_weight": rotation_weight,
        "training_time_seconds": training_time,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "rotation_correlation_params": rotation_params,
        "train_mse": train_final["mse_loss"],
        "train_quadrant_accuracy": train_final["quadrant_accuracy"],
        "train_rotation_accuracy": train_final["rotation_accuracy"],
        "train_mean_distance": train_final["mean_distance"],
        "test_mse": test_final["mse_loss"],
        "test_quadrant_accuracy": test_final["quadrant_accuracy"],
        "test_rotation_accuracy": test_final["rotation_accuracy"],
        "test_mean_distance": test_final["mean_distance"],
        "best_epoch": best_epoch,
        "random_baseline_quadrant": random_quadrant,
        "random_baseline_rotation": random_rotation,
        "comparison": {
            "exp10_quadrant_accuracy": exp10_quad,
            "exp10_rotation_accuracy": exp10_rot,
            "exp11_quadrant_accuracy": exp11_quad,
            "exp11_rotation_accuracy": exp11_rot,
            "quadrant_improvement_vs_exp11": quad_vs_exp11,
            "rotation_improvement_vs_exp11": rot_vs_exp11,
        },
        "success_criteria": {
            "position_target_met": quad_target,
            "rotation_target_met": rot_target,
            "rotation_stretch_met": rot_stretch,
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
    rotation_dropout: float = 0.2,
    piece_size: int = 128,
    puzzle_size: int = 256,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
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
        dropout: Dropout rate for position head.
        rotation_dropout: Dropout rate for rotation head.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        position_weight: Weight for position loss.
        rotation_weight: Weight for rotation loss.

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
        rotation_dropout=rotation_dropout,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        position_weight=position_weight,
        rotation_weight=rotation_weight,
    )

    # Print interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    quad_acc = results["test_quadrant_accuracy"]
    rot_acc = results["test_rotation_accuracy"]

    if quad_acc >= 0.90:
        print(f"Position accuracy ({quad_acc:.1%}) maintained at exp9 levels!")
    elif quad_acc >= 0.85:
        print(f"Position accuracy ({quad_acc:.1%}) acceptable but slightly below exp9.")
    else:
        print(f"Position accuracy ({quad_acc:.1%}) regressed - investigate.")

    print()

    if rot_acc >= 0.90:
        print(f"Rotation accuracy ({rot_acc:.1%}) EXCELLENT!")
        print("Rotation correlation successfully solved the puzzle-piece matching!")
    elif rot_acc >= 0.85:
        print(f"Rotation accuracy ({rot_acc:.1%}) achieved target!")
        print("Comparing piece to puzzle for rotation works.")
    elif rot_acc >= 0.70:
        print(f"Rotation accuracy ({rot_acc:.1%}) improved but below target.")
        print("Rotation correlation helps but may need refinement.")
    else:
        print(f"Rotation accuracy ({rot_acc:.1%}) still struggling.")
        print("The correlation approach may need further tuning.")

    # Compare with exp10/11
    print()
    exp11_rot = 0.598
    if rot_acc > exp11_rot + 0.15:
        print(f"Major improvement over exp11 (+{rot_acc - exp11_rot:.1%})!")
        print("Rotation correlation is the correct architectural fix.")
    elif rot_acc > exp11_rot + 0.05:
        print(f"Good improvement over exp11 (+{rot_acc - exp11_rot:.1%}).")
    elif rot_acc > exp11_rot:
        print(f"Slight improvement over exp11 (+{rot_acc - exp11_rot:.1%}).")
    else:
        print("No improvement over exp11. The hypothesis needs revision.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotation Correlation experiment")
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
        help="Dropout rate for position head",
    )
    parser.add_argument(
        "--rotation-dropout",
        type=float,
        default=0.2,
        help="Dropout rate for rotation head",
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
        help="Size of puzzle images",
    )
    parser.add_argument(
        "--position-weight",
        type=float,
        default=1.0,
        help="Weight for position loss",
    )
    parser.add_argument(
        "--rotation-weight",
        type=float,
        default=1.0,
        help="Weight for rotation loss",
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
        rotation_dropout=args.rotation_dropout,
        piece_size=args.piece_size,
        puzzle_size=args.puzzle_size,
        position_weight=args.position_weight,
        rotation_weight=args.rotation_weight,
    )
