"""Visualization utilities for high-resolution coarse regression experiment.

Provides functions for visualizing:
- Quadrant predictions on a 2x2 grid
- Training curves (MSE loss and accuracy)
- Prediction scatter plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_prediction_grid(
    predictions: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    pred_quadrants: list[int],
    true_quadrants: list[int],
    output_path: Path,
) -> None:
    """Save visualization of predictions on 2x2 grid.

    Args:
        predictions: List of predicted (cx, cy) coordinates.
        targets: List of target (cx, cy) coordinates.
        pred_quadrants: List of predicted quadrant indices.
        true_quadrants: List of true quadrant indices.
        output_path: Path to save the image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter plot of predictions vs targets
    ax1 = axes[0]

    preds_x = [p[0] for p in predictions]
    preds_y = [p[1] for p in predictions]

    # Color by correctness
    colors = ["green" if p == t else "red" for p, t in zip(pred_quadrants, true_quadrants)]

    ax1.scatter(preds_x, preds_y, c=colors, alpha=0.5, s=20, label="Predictions")

    # Draw quadrant boundaries
    ax1.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    ax1.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)

    # Mark quadrant centers
    centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    for i, (cx, cy) in enumerate(centers):
        ax1.plot(cx, cy, "k*", markersize=15, zorder=10)
        ax1.annotate(f"Q{i}", (cx, cy), xytext=(5, 5), textcoords="offset points")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.invert_yaxis()  # Flip y-axis so top-left is (0,0)
    ax1.set_xlabel("Predicted cx")
    ax1.set_ylabel("Predicted cy")
    ax1.set_title("Predictions (green=correct, red=wrong)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confusion matrix by quadrant
    ax2 = axes[1]

    # Build confusion matrix
    confusion = np.zeros((4, 4), dtype=int)
    for pred, true in zip(pred_quadrants, true_quadrants):
        confusion[true, pred] += 1

    im = ax2.imshow(confusion, cmap="Blues")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(["Q0\n(TL)", "Q1\n(TR)", "Q2\n(BL)", "Q3\n(BR)"])
    ax2.set_yticklabels(["Q0 (TL)", "Q1 (TR)", "Q2 (BL)", "Q3 (BR)"])
    ax2.set_xlabel("Predicted Quadrant")
    ax2.set_ylabel("True Quadrant")
    ax2.set_title("Confusion Matrix")

    # Add text annotations
    for i in range(4):
        for j in range(4):
            color = "white" if confusion[i, j] > confusion.max() / 2 else "black"
            ax2.text(j, i, str(confusion[i, j]), ha="center", va="center", color=color)

    # Add colorbar
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # Calculate accuracy
    correct = sum(p == t for p, t in zip(pred_quadrants, true_quadrants))
    accuracy = correct / len(pred_quadrants)
    fig.suptitle(f"Quadrant Prediction Results - Accuracy: {accuracy:.1%}", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save training curves visualization.

    Args:
        history: Dictionary with training history (train_mse, train_acc, test_mse, test_acc).
        output_path: Path to save the image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    epochs = range(1, len(history["train_mse"]) + 1)

    # Plot 1: MSE Loss
    ax1 = axes[0]
    ax1.plot(epochs, history["train_mse"], "b-", label="Train MSE", linewidth=2)

    if history["test_mse"]:
        test_epochs = np.linspace(1, len(history["train_mse"]), len(history["test_mse"]))
        ax1.plot(test_epochs, history["test_mse"], "r-", label="Test MSE", linewidth=2)

    # Target line
    ax1.axhline(y=0.02, color="green", linestyle="--", alpha=0.7, label="Target (0.02)")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Quadrant Accuracy
    ax2 = axes[1]
    ax2.plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        "b-",
        label="Train Acc",
        linewidth=2,
    )

    if history["test_acc"]:
        test_epochs = np.linspace(1, len(history["train_acc"]), len(history["test_acc"]))
        ax2.plot(
            test_epochs,
            [a * 100 for a in history["test_acc"]],
            "r-",
            label="Test Acc",
            linewidth=2,
        )

    # Baselines
    ax2.axhline(y=25, color="gray", linestyle=":", alpha=0.7, label="Random (25%)")
    ax2.axhline(y=67, color="orange", linestyle=":", alpha=0.7, label="Exp7 (67%)")
    ax2.axhline(y=70, color="green", linestyle="--", alpha=0.7, label="Target (70%)")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Quadrant Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    # Plot 3: Mean Distance Error
    ax3 = axes[2]
    if history["test_dist"]:
        test_epochs = np.linspace(1, len(history["train_acc"]), len(history["test_dist"]))
        ax3.plot(test_epochs, history["test_dist"], "r-", label="Test Distance", linewidth=2)

    # Theoretical distances
    # Perfect: 0.0
    # Random within quadrant: ~0.15 (avg distance within 0.5x0.5 square)
    # Wrong quadrant: ~0.5 (distance to other center)
    ax3.axhline(y=0.0, color="green", linestyle="--", alpha=0.7, label="Perfect (0.0)")
    ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7, label="Wrong quadrant (~0.5)")

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean Distance")
    ax3.set_title("Mean Distance Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_sample_predictions(
    puzzle_tensors: list,
    predictions: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    output_path: Path,
    n_samples: int = 8,
) -> None:
    """Save visualization of sample predictions on puzzle images.

    Args:
        puzzle_tensors: List of puzzle image tensors (C, H, W).
        predictions: List of predicted (cx, cy) coordinates.
        targets: List of target (cx, cy) coordinates.
        output_path: Path to save the image.
        n_samples: Number of samples to show.
    """
    import torch

    n_samples = min(n_samples, len(puzzle_tensors))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_samples):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        # Convert tensor to numpy
        if isinstance(puzzle_tensors[i], torch.Tensor):
            img = puzzle_tensors[i].permute(1, 2, 0).numpy()
        else:
            img = puzzle_tensors[i]

        ax.imshow(img)

        h, w = img.shape[:2]
        pred_x, pred_y = predictions[i]
        target_x, target_y = targets[i]

        # Plot target (green) and prediction (red)
        ax.plot(target_x * w, target_y * h, "g*", markersize=20, label="Target")
        ax.plot(pred_x * w, pred_y * h, "ro", markersize=10, label="Prediction")

        # Draw arrow from prediction to target
        ax.annotate(
            "",
            xy=(target_x * w, target_y * h),
            xytext=(pred_x * w, pred_y * h),
            arrowprops={"arrowstyle": "->", "color": "yellow", "lw": 2},
        )

        # Draw quadrant lines
        ax.axhline(y=h / 2, color="white", linestyle="--", alpha=0.5)
        ax.axvline(x=w / 2, color="white", linestyle="--", alpha=0.5)

        dist = ((pred_x - target_x) ** 2 + (pred_y - target_y) ** 2) ** 0.5
        ax.set_title(f"Sample {i}: dist={dist:.3f}")
        ax.axis("off")

    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities for exp8_high_res_correlation")
    print("Run the training script to generate visualizations.")
