"""Visualization utilities for multi-puzzle high-resolution experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def save_accuracy_grid(
    puzzle_tensor: torch.Tensor,
    targets: list[int],
    predictions: list[int],
    num_cols: int,
    num_rows: int,
    output_path: Path,
) -> None:
    """Save a visualization of prediction accuracy on the puzzle grid.

    Args:
        puzzle_tensor: Puzzle image tensor (C, H, W).
        targets: List of target cell indices.
        predictions: List of predicted cell indices.
        num_cols: Number of columns in the grid.
        num_rows: Number of rows in the grid.
        output_path: Path to save the image.
    """
    # Convert puzzle to numpy for display
    puzzle_np = puzzle_tensor.permute(1, 2, 0).numpy()

    # Create accuracy grid
    correct_grid = np.zeros((num_rows, num_cols), dtype=bool)
    for target, pred in zip(targets, predictions):
        target_row = target // num_cols
        target_col = target % num_cols
        correct_grid[target_row, target_col] = target == pred

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Show puzzle
    ax.imshow(puzzle_np)

    # Calculate cell dimensions
    h, w = puzzle_np.shape[:2]
    cell_w = w / num_cols
    cell_h = h / num_rows

    # Draw grid with color coding
    for row in range(num_rows):
        for col in range(num_cols):
            x = col * cell_w
            y = row * cell_h

            # Color based on correctness
            if correct_grid[row, col]:
                color = "green"
                alpha = 0.3
            else:
                color = "red"
                alpha = 0.4

            rect = Rectangle(
                (x, y),
                cell_w,
                cell_h,
                linewidth=0.5,
                edgecolor=color,
                facecolor=color,
                alpha=alpha,
            )
            ax.add_patch(rect)

    # Calculate accuracy
    accuracy = sum(t == p for t, p in zip(targets, predictions)) / len(targets)

    ax.set_title(f"Test Puzzle Accuracy: {accuracy:.1%} ({sum(correct_grid.flat)}/{len(targets)} correct)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save training curves visualization.

    Args:
        history: Dictionary with training history.
        output_path: Path to save the image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax1 = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)

    if history["test_loss"]:
        # Test loss is sampled less frequently
        test_epochs = np.linspace(1, len(history["train_loss"]), len(history["test_loss"]))
        ax1.plot(test_epochs, history["test_loss"], "r-", label="Test Loss", linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2 = axes[1]
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-", label="Train Acc", linewidth=2)

    if history["test_acc"]:
        test_epochs = np.linspace(1, len(history["train_acc"]), len(history["test_acc"]))
        ax2.plot(test_epochs, [a * 100 for a in history["test_acc"]], "r-", label="Test Acc", linewidth=2)

        if history["test_top5"]:
            ax2.plot(
                test_epochs,
                [a * 100 for a in history["test_top5"]],
                "r--",
                label="Test Top-5",
                linewidth=2,
                alpha=0.7,
            )

    # Add random baseline
    ax2.axhline(y=100 / 950, color="gray", linestyle=":", label="Random (0.105%)", alpha=0.7)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion_analysis(
    puzzle_tensor: torch.Tensor,
    targets: list[int],
    predictions: list[int],
    num_cols: int,
    num_rows: int,
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Save analysis of prediction errors.

    Args:
        puzzle_tensor: Puzzle image tensor (C, H, W).
        targets: List of target cell indices.
        predictions: List of predicted cell indices.
        num_cols: Number of columns in the grid.
        num_rows: Number of rows in the grid.
        output_path: Path to save the image.
        top_k: Number of error examples to show.
    """
    # Find errors
    errors = []
    for i, (target, pred) in enumerate(zip(targets, predictions)):
        if target != pred:
            target_row, target_col = target // num_cols, target % num_cols
            pred_row, pred_col = pred // num_cols, pred % num_cols
            distance = abs(target_row - pred_row) + abs(target_col - pred_col)
            errors.append((i, target, pred, distance))

    # Sort by distance (show worst errors first)
    errors.sort(key=lambda x: -x[3])
    errors = errors[:top_k]

    if not errors:
        print("No errors to visualize!")
        return

    # Convert puzzle to numpy
    puzzle_np = puzzle_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(puzzle_np)

    h, w = puzzle_np.shape[:2]
    cell_w = w / num_cols
    cell_h = h / num_rows

    # Draw error arrows
    for _idx, target, pred, _distance in errors:
        target_row, target_col = target // num_cols, target % num_cols
        pred_row, pred_col = pred // num_cols, pred % num_cols

        target_x = (target_col + 0.5) * cell_w
        target_y = (target_row + 0.5) * cell_h
        pred_x = (pred_col + 0.5) * cell_w
        pred_y = (pred_row + 0.5) * cell_h

        # Draw arrow from prediction to target
        ax.annotate(
            "",
            xy=(target_x, target_y),
            xytext=(pred_x, pred_y),
            arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 2},
        )

        # Mark target (green) and prediction (red)
        ax.plot(target_x, target_y, "go", markersize=8)
        ax.plot(pred_x, pred_y, "ro", markersize=8)

    ax.set_title(f"Top {len(errors)} Errors (yellow arrows: prediction -> target)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities for exp6")
    print("Run the training script to generate visualizations.")
