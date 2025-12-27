"""Training script for fast backbone comparison.

Quick sanity check to compare training speed of lightweight backbones:
- RepVGG-A0
- MobileOne-S0
- ShuffleNetV2_x0.5

Runs 2 epochs on a reduced dataset to verify models work and compare timing.
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
from .model import BackboneType, FastBackboneModel, count_parameters


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


def compute_metrics(
    model: FastBackboneModel,
    dataset: QuadrantAllRotationsDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compute evaluation metrics on a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_quadrant_correct = 0
    total_rotation_correct = 0
    total_mse = 0.0
    n_samples = 0

    with torch.no_grad():
        for pieces, puzzles, targets, quadrants, rotations in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)
            targets = targets.to(device)

            preds, rotation_logits, _ = model(pieces, puzzles)

            mse = F.mse_loss(preds, targets, reduction="sum")
            total_mse += mse.item()
            n_samples += targets.size(0)

            pred_quadrants = model.predict_quadrant(pieces, puzzles)
            total_quadrant_correct += (pred_quadrants == quadrants.to(device)).sum().item()

            pred_rotations = rotation_logits.argmax(dim=1)
            total_rotation_correct += (pred_rotations == rotations.to(device)).sum().item()

    return {
        "mse_loss": total_mse / n_samples,
        "quadrant_accuracy": total_quadrant_correct / n_samples,
        "rotation_accuracy": total_rotation_correct / n_samples,
    }


def run_backbone_test(
    backbone_name: BackboneType,
    epochs: int = 2,
    n_train_puzzles: int = 500,
    n_test_puzzles: int = 100,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
    piece_size: int = 128,
    puzzle_size: int = 256,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run training test for a single backbone.

    Args:
        backbone_name: Which backbone to test.
        epochs: Number of training epochs.
        n_train_puzzles: Number of training puzzles.
        n_test_puzzles: Number of test puzzles.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone.
        head_lr: Learning rate for heads.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        output_dir: Directory for outputs.

    Returns:
        Dictionary with results.
    """
    print(f"\n{'=' * 60}")
    print(f"TESTING: {backbone_name}")
    print("=" * 60)

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

    # Create model
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
    epoch_times: list[float] = []
    history: dict[str, list[float]] = {
        "train_pos_loss": [],
        "train_rot_loss": [],
        "train_quad_acc": [],
        "train_rot_acc": [],
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

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        history["train_pos_loss"].append(train_metrics["position_loss"])
        history["train_rot_loss"].append(train_metrics["rotation_loss"])
        history["train_quad_acc"].append(train_metrics["quadrant_accuracy"])
        history["train_rot_acc"].append(train_metrics["rotation_accuracy"])

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"pos_loss={train_metrics['position_loss']:.4f}, "
            f"rot_loss={train_metrics['rotation_loss']:.4f}, "
            f"quad_acc={train_metrics['quadrant_accuracy']:.1%}, "
            f"rot_acc={train_metrics['rotation_accuracy']:.1%}, "
            f"time={epoch_time:.1f}s"
        )

    total_time = time.time() - total_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # Quick test evaluation
    print("\nEvaluating on test set...")
    test_metrics = compute_metrics(model, test_dataset, device)
    print(
        f"Test: quad_acc={test_metrics['quadrant_accuracy']:.1%}, " f"rot_acc={test_metrics['rotation_accuracy']:.1%}"
    )

    # Compile results
    results = {
        "backbone": backbone_name,
        "feature_dim": model.feature_dim,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs": epochs,
        "n_train_puzzles": n_train_puzzles,
        "n_test_puzzles": n_test_puzzles,
        "batch_size": batch_size,
        "total_training_time": total_time,
        "avg_epoch_time": avg_epoch_time,
        "epoch_times": epoch_times,
        "final_train_pos_loss": history["train_pos_loss"][-1],
        "final_train_rot_loss": history["train_rot_loss"][-1],
        "final_train_quad_acc": history["train_quad_acc"][-1],
        "final_train_rot_acc": history["train_rot_acc"][-1],
        "test_quad_acc": test_metrics["quadrant_accuracy"],
        "test_rot_acc": test_metrics["rotation_accuracy"],
        "history": history,
    }

    # Save results if output dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"results_{backbone_name}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_file}")

    return results


def main(
    epochs: int = 2,
    n_train: int = 500,
    n_test: int = 100,
    batch_size: int = 64,
    backbone_lr: float = 1e-4,
    head_lr: float = 1e-3,
) -> dict[str, Any]:
    """Run comparison of all fast backbones.

    Args:
        epochs: Number of training epochs per backbone.
        n_train: Number of training puzzles.
        n_test: Number of test puzzles.
        batch_size: Training batch size.
        backbone_lr: Learning rate for backbone.
        head_lr: Learning rate for heads.

    Returns:
        Dictionary with all results.
    """
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    backbones: list[BackboneType] = [
        "repvgg_a0",
        "mobileone_s0",
        "shufflenet_v2_x0_5",
    ]

    all_results: dict[str, Any] = {}

    print("\n" + "=" * 70)
    print("EXPERIMENT 15: FAST BACKBONE COMPARISON")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Training puzzles: {n_train}")
    print(f"Test puzzles: {n_test}")
    print(f"Batch size: {batch_size}")
    print(f"Backbone LR: {backbone_lr}")
    print(f"Head LR: {head_lr}")
    print(f"Backbones to test: {backbones}")

    for backbone_name in backbones:
        try:
            results = run_backbone_test(
                backbone_name=backbone_name,
                epochs=epochs,
                n_train_puzzles=n_train,
                n_test_puzzles=n_test,
                batch_size=batch_size,
                backbone_lr=backbone_lr,
                head_lr=head_lr,
                output_dir=output_dir,
            )
            all_results[backbone_name] = results
        except Exception as e:
            print(f"\nERROR with {backbone_name}: {e}")
            all_results[backbone_name] = {"error": str(e)}

    # Print comparison summary
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Backbone':<25} {'Params':>12} {'Feat Dim':>10} "
        f"{'Epoch Time':>12} {'Train Quad':>12} {'Test Quad':>12}"
    )
    print("-" * 85)

    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<25} ERROR: {res['error']}")
        else:
            print(
                f"{name:<25} {res['trainable_params']:>12,} {res['feature_dim']:>10} "
                f"{res['avg_epoch_time']:>10.1f}s {res['final_train_quad_acc']:>11.1%} "
                f"{res['test_quad_acc']:>11.1%}"
            )

    # Save combined results
    combined_file = output_dir / "all_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to {combined_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast backbone comparison")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs")
    parser.add_argument("--n-train", type=int, default=500, help="Training puzzles")
    parser.add_argument("--n-test", type=int, default=100, help="Test puzzles")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head LR")
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
    )
