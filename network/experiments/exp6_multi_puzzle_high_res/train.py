"""Training script for multi-puzzle high-resolution experiment.

This experiment tests whether training on multiple puzzles with higher resolution
puzzle images enables cross-puzzle generalization.

Experimental Design:
1. Train on 5 puzzles (puzzle_001 to puzzle_005) with 512x512 resolution
2. Test on held-out puzzle_006
3. Compare training accuracy vs test accuracy
4. Random baseline: 1/950 = 0.105% accuracy

Key hypothesis: Multi-puzzle training + higher resolution provides enough
variation and detail to learn a generalizable matching function.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import MultiPuzzleDataset, SinglePuzzleDataset, create_train_test_datasets
from .model import HighResDualInputClassifier, count_backbone_parameters, count_parameters
from .visualize import save_accuracy_grid, save_training_curves


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_accuracy(
    model: HighResDualInputClassifier,
    dataset: SinglePuzzleDataset | MultiPuzzleDataset,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compute accuracy metrics on a dataset.

    Args:
        model: Trained dual-input model.
        dataset: Dataset to evaluate.
        device: Computation device.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with accuracy metrics and predictions.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for pieces, puzzles, targets in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)
            targets = targets.to(device)

            logits, _ = model(pieces, puzzles)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item()
            n_batches += 1

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    # Calculate metrics
    correct = sum(t == p for t, p in zip(all_targets, all_preds))
    top1_acc = correct / len(all_targets)

    # Top-5 accuracy
    top5_correct = 0
    for target, probs in zip(all_targets, all_probs):
        top5_indices = sorted(range(len(probs)), key=lambda x: probs[x], reverse=True)[:5]
        if target in top5_indices:
            top5_correct += 1
    top5_acc = top5_correct / len(all_targets)

    return {
        "loss": total_loss / n_batches,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "correct": correct,
        "total": len(all_targets),
        "all_targets": all_targets,
        "all_preds": all_preds,
    }


def train_epoch(
    model: HighResDualInputClassifier,
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
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for pieces, puzzles, targets in loader:
        pieces = pieces.to(device)
        puzzles = puzzles.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits, _ = model(pieces, puzzles)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    return {
        "loss": total_loss / n_batches,
        "accuracy": total_correct / total_samples,
    }


_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def run_experiment(
    train_puzzle_ids: list[str],
    test_puzzle_id: str,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    puzzle_size: int = 512,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run multi-puzzle training experiment.

    Args:
        train_puzzle_ids: List of puzzle IDs for training.
        test_puzzle_id: Puzzle ID for testing (held out).
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        puzzle_size: Size of puzzle images (default 512).
        output_dir: Directory for outputs.

    Returns:
        Dictionary with all experiment results.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: MULTI-PUZZLE HIGH-RESOLUTION TRAINING")
    print("=" * 70)
    print(f"Training puzzles: {train_puzzle_ids}")
    print(f"Test puzzle: {test_puzzle_id}")
    print(f"Puzzle resolution: {puzzle_size}x{puzzle_size}")

    device = get_device()
    print(f"Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset = create_train_test_datasets(
        train_puzzle_ids=train_puzzle_ids,
        test_puzzle_id=test_puzzle_id,
        piece_size=64,
        puzzle_size=puzzle_size,
    )

    print(f"\nTraining samples: {len(train_dataset)} pieces from {len(train_puzzle_ids)} puzzles")
    print(f"Test samples: {len(test_dataset)} pieces from 1 puzzle")

    # Create model
    model = HighResDualInputClassifier(num_cells=950).to(device)
    total_params = count_parameters(model)
    backbone_params = count_backbone_parameters(model)
    print("\nModel: HighResDualInputClassifier")
    print(f"Total parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS works best with 0 workers
    )

    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_top5": [],
        "test_loss": [],
    }

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        # Evaluate on test set periodically
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            test_metrics = compute_accuracy(model, test_dataset, device)
            history["test_acc"].append(test_metrics["top1_accuracy"])
            history["test_top5"].append(test_metrics["top5_accuracy"])
            history["test_loss"].append(test_metrics["loss"])

            # Track best
            if test_metrics["top1_accuracy"] > best_test_acc:
                best_test_acc = test_metrics["top1_accuracy"]
                best_epoch = epoch + 1

            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.1%}, "
                f"test_acc={test_metrics['top1_accuracy']:.2%}, "
                f"test_top5={test_metrics['top5_accuracy']:.2%}"
            )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    train_final = compute_accuracy(model, train_dataset, device)
    test_final = compute_accuracy(model, test_dataset, device)

    print(f"\nTraining puzzles ({len(train_puzzle_ids)} puzzles):")
    print(f"  Top-1 Accuracy: {train_final['top1_accuracy']:.2%}")
    print(f"  Top-5 Accuracy: {train_final['top5_accuracy']:.2%}")
    print(f"  Loss: {train_final['loss']:.4f}")

    print(f"\nTest puzzle ({test_puzzle_id}) - HELD OUT:")
    print(f"  Top-1 Accuracy: {test_final['top1_accuracy']:.2%}")
    print(f"  Top-5 Accuracy: {test_final['top5_accuracy']:.2%}")
    print(f"  Loss: {test_final['loss']:.4f}")

    random_baseline = 1 / 950
    print(f"\nRandom baseline: {random_baseline:.4%}")
    print(f"Test vs random: {test_final['top1_accuracy'] / random_baseline:.1f}x")
    print(f"Best test accuracy: {best_test_acc:.2%} (epoch {best_epoch})")

    # Save visualizations
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save accuracy grid for test puzzle
    save_accuracy_grid(
        test_dataset.get_puzzle_image(),
        test_final["all_targets"],
        test_final["all_preds"],
        test_dataset.num_cols,
        test_dataset.num_rows,
        output_dir / "test_accuracy_grid.png",
    )
    print(f"\nSaved test accuracy grid to {output_dir / 'test_accuracy_grid.png'}")

    # Save training curves
    save_training_curves(
        history,
        output_dir / "training_curves.png",
    )
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")

    # Compile results
    results: dict[str, Any] = {
        "experiment": "exp6_multi_puzzle_high_res",
        "train_puzzles": train_puzzle_ids,
        "test_puzzle": test_puzzle_id,
        "puzzle_size": puzzle_size,
        "epochs": epochs,
        "training_time_seconds": training_time,
        "model_params": total_params,
        "train_accuracy": train_final["top1_accuracy"],
        "train_top5": train_final["top5_accuracy"],
        "train_loss": train_final["loss"],
        "test_accuracy": test_final["top1_accuracy"],
        "test_top5": test_final["top5_accuracy"],
        "test_loss": test_final["loss"],
        "best_test_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "random_baseline": random_baseline,
        "test_vs_random": test_final["top1_accuracy"] / random_baseline,
    }

    # Save results to JSON
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_file}")

    return results


def main(
    epochs: int = 200,
    puzzle_size: int = 512,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run the full experiment.

    Args:
        epochs: Number of training epochs.
        puzzle_size: Size of puzzle images.
        batch_size: Training batch size.

    Returns:
        Experiment results.
    """
    # Training puzzles (5 puzzles for diversity, all with 950 pieces)
    # Note: puzzle_004 has 1014 pieces (different grid), so we skip it
    train_puzzles = [
        "puzzle_001",
        "puzzle_002",
        "puzzle_003",
        "puzzle_005",
        "puzzle_006",
    ]

    # Held-out test puzzle (also 950 pieces)
    test_puzzle = "puzzle_007"

    results = run_experiment(
        train_puzzle_ids=train_puzzles,
        test_puzzle_id=test_puzzle,
        epochs=epochs,
        batch_size=batch_size,
        puzzle_size=puzzle_size,
    )

    # Print summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    test_acc = results["test_accuracy"]
    random_baseline = results["random_baseline"]

    if test_acc < random_baseline * 2:  # Less than 2x random
        print("Test accuracy near random chance.")
        print("CONCLUSION: Multi-puzzle training did NOT enable generalization.")
        print("The model still memorizes puzzle-specific patterns.")
    elif test_acc < 0.05:  # Less than 5%
        print("Test accuracy slightly above random (but < 5%).")
        print("CONCLUSION: Weak generalization - some matching learned.")
    elif test_acc < 0.10:  # Less than 10%
        print("Test accuracy moderately above random (5-10%).")
        print("CONCLUSION: Partial generalization - matching function emerging.")
    elif test_acc < 0.30:  # Less than 30%
        print("Test accuracy significantly above random (10-30%).")
        print("CONCLUSION: Meaningful generalization achieved!")
        print("Multi-puzzle training helps learn transferable features.")
    else:  # 30% or higher
        print("Test accuracy high (>30%)!")
        print("CONCLUSION: Strong generalization achieved!")
        print("The multi-puzzle high-res approach works well.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-puzzle high-resolution training experiment")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--puzzle-size",
        type=int,
        default=512,
        help="Puzzle image size (default 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    args = parser.parse_args()

    main(epochs=args.epochs, puzzle_size=args.puzzle_size, batch_size=args.batch_size)
