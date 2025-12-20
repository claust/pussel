"""Training script for cross-puzzle generalization experiment.

This experiment tests whether a dual-input model that receives BOTH piece
and puzzle images can learn a matching function that generalizes across puzzles.

Experimental Design:
1. Train on puzzle_001 pieces with puzzle_001 image as context
2. Test on puzzle_002 pieces with puzzle_002 image as context
3. Compare training accuracy vs test accuracy
4. Random baseline: 1/950 = 0.105% accuracy

The key insight: By providing the puzzle image, the model learns to MATCH
pieces to puzzles rather than memorize texture-to-position mappings.
This should enable cross-puzzle generalization.

Expected outcomes:
- Near-random test accuracy: Matching function not learned (unlikely with dual-input)
- Partial test accuracy (10-50%): Some matching ability learned
- High test accuracy (>70%): Strong matching function learned
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import PuzzleDataset, verify_cell_indices
from .model import DualInputCellClassifier, count_backbone_parameters, count_parameters
from .visualize import create_comparison_figure, save_accuracy_grid


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_accuracy(
    model: DualInputCellClassifier,
    dataset: PuzzleDataset,
    device: torch.device,
    batch_size: int = 64,
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

            # Dual-input forward pass
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
    model: DualInputCellClassifier,
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

        # Dual-input forward pass
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
    train_puzzle_id: str,
    test_puzzle_id: str,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    run_id: int = 1,
) -> dict[str, Any]:
    """Run a single cross-puzzle generalization experiment.

    Args:
        train_puzzle_id: Puzzle ID to train on (e.g., "puzzle_001").
        test_puzzle_id: Puzzle ID to test on (e.g., "puzzle_002").
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        output_dir: Directory for outputs.
        run_id: Run identifier for multiple runs.

    Returns:
        Dictionary with all experiment results.
    """
    print("\n" + "=" * 70)
    print(f"CROSS-PUZZLE GENERALIZATION EXPERIMENT (Run {run_id})")
    print(f"Train: {train_puzzle_id} -> Test: {test_puzzle_id}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = PuzzleDataset(
        puzzle_id=train_puzzle_id,
        piece_size=64,
        puzzle_size=256,
        unrotate_pieces=True,
    )
    test_dataset = PuzzleDataset(
        puzzle_id=test_puzzle_id,
        piece_size=64,
        puzzle_size=256,
        unrotate_pieces=True,
    )

    # Verify datasets
    train_verification = verify_cell_indices(train_dataset)
    test_verification = verify_cell_indices(test_dataset)
    print(f"\nTrain dataset: {len(train_dataset)} pieces, " f"{train_verification['unique_cells']} unique cells")
    print(f"Test dataset: {len(test_dataset)} pieces, " f"{test_verification['unique_cells']} unique cells")

    # Create dual-input model
    model = DualInputCellClassifier(num_cells=train_dataset.num_cells).to(device)
    total_params = count_parameters(model)
    backbone_params = count_backbone_parameters(model)
    print("\nModel: DualInputCellClassifier")
    print(f"Total parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS works best with 0 workers
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    start_time = time.time()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_top5": [],
    }

    epochs_trained = 0
    for epoch in range(epochs):
        epochs_trained = epoch + 1
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])

        # Evaluate on test set periodically
        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            test_metrics = compute_accuracy(model, test_dataset, device)
            history["test_acc"].append(test_metrics["top1_accuracy"])
            history["test_top5"].append(test_metrics["top5_accuracy"])

            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.1%}, "
                f"test_acc={test_metrics['top1_accuracy']:.1%}, "
                f"test_top5={test_metrics['top5_accuracy']:.1%}"
            )

            # Early stopping if training accuracy is high enough
            if train_metrics["accuracy"] >= 0.99:
                print(f"\nReached 99% training accuracy at epoch {epoch + 1}. " "Stopping early.")
                break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Final evaluation
    print("\nFinal Evaluation:")
    print("-" * 50)

    train_final = compute_accuracy(model, train_dataset, device)
    test_final = compute_accuracy(model, test_dataset, device)

    print(f"Training puzzle ({train_puzzle_id}):")
    print(f"  Top-1 Accuracy: {train_final['top1_accuracy']:.2%}")
    print(f"  Top-5 Accuracy: {train_final['top5_accuracy']:.2%}")
    print(f"  Loss: {train_final['loss']:.4f}")

    print(f"\nTest puzzle ({test_puzzle_id}):")
    print(f"  Top-1 Accuracy: {test_final['top1_accuracy']:.2%}")
    print(f"  Top-5 Accuracy: {test_final['top5_accuracy']:.2%}")
    print(f"  Loss: {test_final['loss']:.4f}")

    print(f"\nRandom baseline: {1 / train_dataset.num_cells:.4%}")
    print(f"Generalization gap: " f"{train_final['top1_accuracy'] - test_final['top1_accuracy']:.2%}")

    # Save visualizations
    output_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = f"run{run_id}_{train_puzzle_id}_to_{test_puzzle_id}"

    # Save comparison figure
    create_comparison_figure(
        train_dataset.get_puzzle_image(),
        test_dataset.get_puzzle_image(),
        train_final["all_targets"],
        train_final["all_preds"],
        test_final["all_targets"],
        test_final["all_preds"],
        train_dataset.num_cols,
        train_dataset.num_rows,
        output_dir / f"{run_prefix}_comparison.png",
        train_puzzle_id,
        test_puzzle_id,
    )
    print(f"\nSaved comparison figure to {output_dir / f'{run_prefix}_comparison.png'}")

    # Save test accuracy grid
    save_accuracy_grid(
        test_dataset.get_puzzle_image(),
        test_final["all_targets"],
        test_final["all_preds"],
        test_dataset.num_cols,
        test_dataset.num_rows,
        output_dir / f"{run_prefix}_test_accuracy.png",
    )
    print(f"Saved test accuracy grid to {output_dir / f'{run_prefix}_test_accuracy.png'}")

    # Compile results
    results: dict[str, Any] = {
        "run_id": run_id,
        "train_puzzle": train_puzzle_id,
        "test_puzzle": test_puzzle_id,
        "epochs_trained": epochs_trained,
        "training_time_seconds": training_time,
        "train_accuracy": train_final["top1_accuracy"],
        "train_top5": train_final["top5_accuracy"],
        "train_loss": train_final["loss"],
        "test_accuracy": test_final["top1_accuracy"],
        "test_top5": test_final["top5_accuracy"],
        "test_loss": test_final["loss"],
        "random_baseline": 1 / train_dataset.num_cells,
        "generalization_gap": train_final["top1_accuracy"] - test_final["top1_accuracy"],
        "model_params": total_params,
    }

    return results


def main(epochs: int = 300, runs: int = 1) -> list[dict[str, Any]]:
    """Run the full cross-puzzle generalization experiment.

    Args:
        epochs: Number of training epochs per run.
        runs: Number of times to repeat the experiment.

    Returns:
        All experiment results.
    """
    print("=" * 70)
    print("EXPERIMENT 5: CROSS-PUZZLE GENERALIZATION (FIXED)")
    print("=" * 70)
    print("\nThis experiment uses a DUAL-INPUT architecture that receives both")
    print("the piece image AND the puzzle image. This enables learning a")
    print("matching function that can generalize across different puzzles.")
    print("\nKey insight: The model learns WHERE a piece belongs in a GIVEN puzzle,")
    print("not just memorizing texture-to-position mappings for a single puzzle.")

    output_dir = _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for run in range(1, runs + 1):
        # Run experiment: train on puzzle_001, test on puzzle_002
        results = run_experiment(
            train_puzzle_id="puzzle_001",
            test_puzzle_id="puzzle_002",
            epochs=epochs,
            batch_size=128,
            lr=1e-3,
            output_dir=output_dir,
            run_id=run,
        )
        all_results.append(results)

    # Summary across runs
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    avg_train_acc = sum(r["train_accuracy"] for r in all_results) / len(all_results)
    avg_test_acc = sum(r["test_accuracy"] for r in all_results) / len(all_results)
    avg_gap = sum(r["generalization_gap"] for r in all_results) / len(all_results)

    print(f"\nRuns: {len(all_results)}")
    print(f"Average training accuracy: {avg_train_acc:.2%}")
    print(f"Average test accuracy: {avg_test_acc:.2%}")
    print(f"Average generalization gap: {avg_gap:.2%}")
    print(f"Random baseline: {all_results[0]['random_baseline']:.4%}")

    # Interpret results
    print("\n" + "-" * 50)
    print("INTERPRETATION:")
    if avg_test_acc < 0.01:  # Less than 1%
        print("  Test accuracy near random chance.")
        print("  CONCLUSION: Matching function not learned.")
        print("  The model did not generalize across puzzles.")
    elif avg_test_acc < 0.10:  # 1-10%
        print("  Test accuracy slightly above random.")
        print("  CONCLUSION: Weak matching ability learned.")
        print("  Some generalization, but limited.")
    elif avg_test_acc < 0.50:  # 10-50%
        print("  Test accuracy significantly above random.")
        print("  CONCLUSION: Partial matching function learned.")
        print("  The dual-input architecture enables some generalization.")
    else:  # > 50%
        print("  Test accuracy high!")
        print("  CONCLUSION: Strong matching function learned.")
        print("  The model successfully generalizes across puzzles.")

    # Save results to JSON
    results_file = output_dir / "experiment_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "experiment": "exp5_cross_puzzle_generalization_fixed",
                "architecture": "DualInputCellClassifier",
                "description": (
                    "Dual-input model that receives both piece and puzzle images, "
                    "enabling learning a matching function for cross-puzzle "
                    "generalization."
                ),
                "summary": {
                    "num_runs": len(all_results),
                    "avg_train_accuracy": avg_train_acc,
                    "avg_test_accuracy": avg_test_acc,
                    "avg_generalization_gap": avg_gap,
                    "random_baseline": all_results[0]["random_baseline"],
                },
                "runs": all_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {results_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-puzzle generalization experiment (fixed)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs per run",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat the experiment",
    )
    args = parser.parse_args()

    main(epochs=args.epochs, runs=args.runs)
