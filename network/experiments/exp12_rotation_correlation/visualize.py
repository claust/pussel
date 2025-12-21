"""Visualization utilities for rotation correlation experiment.

Provides functions for visualizing:
- Quadrant predictions on a 2x2 grid
- Rotation predictions (confusion matrix)
- Training curves (position loss, rotation loss, accuracies)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_prediction_grid(
    predictions: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    pred_quadrants: list[int],
    true_quadrants: list[int],
    pred_rotations: list[int],
    true_rotations: list[int],
    output_path: Path,
) -> None:
    """Save visualization of predictions on 2x2 grid with rotation info.

    Args:
        predictions: List of predicted (cx, cy) coordinates.
        targets: List of target (cx, cy) coordinates.
        pred_quadrants: List of predicted quadrant indices.
        true_quadrants: List of true quadrant indices.
        pred_rotations: List of predicted rotation indices.
        true_rotations: List of true rotation indices.
        output_path: Path to save the image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Position scatter plot
    ax1 = axes[0]

    preds_x = [p[0] for p in predictions]
    preds_y = [p[1] for p in predictions]

    # Color by correctness (quadrant)
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
    ax1.invert_yaxis()
    ax1.set_xlabel("Predicted cx")
    ax1.set_ylabel("Predicted cy")

    quad_acc = sum(p == t for p, t in zip(pred_quadrants, true_quadrants)) / len(pred_quadrants)
    ax1.set_title(f"Position Predictions\nQuadrant Accuracy: {quad_acc:.1%}")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Quadrant confusion matrix
    ax2 = axes[1]

    quad_confusion = np.zeros((4, 4), dtype=int)
    for pred, true in zip(pred_quadrants, true_quadrants):
        quad_confusion[true, pred] += 1

    im2 = ax2.imshow(quad_confusion, cmap="Blues")
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(["Q0\n(TL)", "Q1\n(TR)", "Q2\n(BL)", "Q3\n(BR)"])
    ax2.set_yticklabels(["Q0 (TL)", "Q1 (TR)", "Q2 (BL)", "Q3 (BR)"])
    ax2.set_xlabel("Predicted Quadrant")
    ax2.set_ylabel("True Quadrant")
    ax2.set_title("Quadrant Confusion Matrix")

    for i in range(4):
        for j in range(4):
            color = "white" if quad_confusion[i, j] > quad_confusion.max() / 2 else "black"
            ax2.text(j, i, str(quad_confusion[i, j]), ha="center", va="center", color=color)

    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot 3: Rotation confusion matrix
    ax3 = axes[2]

    rot_confusion = np.zeros((4, 4), dtype=int)
    for pred, true in zip(pred_rotations, true_rotations):
        rot_confusion[true, pred] += 1

    im3 = ax3.imshow(rot_confusion, cmap="Oranges")
    ax3.set_xticks(range(4))
    ax3.set_yticks(range(4))
    rotation_labels = ["0", "90", "180", "270"]
    ax3.set_xticklabels(rotation_labels)
    ax3.set_yticklabels(rotation_labels)
    ax3.set_xlabel("Predicted Rotation")
    ax3.set_ylabel("True Rotation")

    rot_acc = sum(p == t for p, t in zip(pred_rotations, true_rotations)) / len(pred_rotations)
    ax3.set_title(f"Rotation Confusion Matrix\nAccuracy: {rot_acc:.1%}")

    for i in range(4):
        for j in range(4):
            color = "white" if rot_confusion[i, j] > rot_confusion.max() / 2 else "black"
            ax3.text(j, i, str(rot_confusion[i, j]), ha="center", va="center", color=color)

    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Exp12 Rotation Correlation - Quad: {quad_acc:.1%}, Rot: {rot_acc:.1%}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save training curves visualization for multi-task learning.

    Args:
        history: Dictionary with training history.
        output_path: Path to save the image.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Position Loss
    ax1 = axes[0, 0]
    epochs = range(1, len(history["train_pos_loss"]) + 1)
    ax1.plot(epochs, history["train_pos_loss"], "b-", label="Train", linewidth=2)

    if history["test_pos_loss"]:
        test_epochs = np.linspace(1, len(history["train_pos_loss"]), len(history["test_pos_loss"]))
        ax1.plot(test_epochs, history["test_pos_loss"], "r-", label="Test", linewidth=2)

    ax1.axhline(y=0.02, color="green", linestyle="--", alpha=0.7, label="Target")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Position MSE Loss")
    ax1.set_title("Position Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Rotation Loss
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["train_rot_loss"], "b-", label="Train", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Rotation Cross-Entropy Loss")
    ax2.set_title("Rotation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Quadrant Accuracy
    ax3 = axes[1, 0]
    ax3.plot(
        epochs,
        [a * 100 for a in history["train_quad_acc"]],
        "b-",
        label="Train",
        linewidth=2,
    )

    if history["test_quad_acc"]:
        test_epochs = np.linspace(1, len(history["train_quad_acc"]), len(history["test_quad_acc"]))
        ax3.plot(
            test_epochs,
            [a * 100 for a in history["test_quad_acc"]],
            "r-",
            label="Test",
            linewidth=2,
        )

    ax3.axhline(y=25, color="gray", linestyle=":", alpha=0.7, label="Random (25%)")
    ax3.axhline(y=90, color="green", linestyle="--", alpha=0.7, label="Target (90%)")
    ax3.axhline(y=73, color="orange", linestyle=":", alpha=0.7, label="Exp11 (73%)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Quadrant Accuracy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 105)

    # Plot 4: Rotation Accuracy
    ax4 = axes[1, 1]
    ax4.plot(
        epochs,
        [a * 100 for a in history["train_rot_acc"]],
        "b-",
        label="Train",
        linewidth=2,
    )

    if history["test_rot_acc"]:
        test_epochs = np.linspace(1, len(history["train_rot_acc"]), len(history["test_rot_acc"]))
        ax4.plot(
            test_epochs,
            [a * 100 for a in history["test_rot_acc"]],
            "r-",
            label="Test",
            linewidth=2,
        )

    ax4.axhline(y=25, color="gray", linestyle=":", alpha=0.7, label="Random (25%)")
    ax4.axhline(y=85, color="green", linestyle="--", alpha=0.7, label="Target (85%)")
    ax4.axhline(y=60, color="orange", linestyle=":", alpha=0.7, label="Exp11 (60%)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.set_title("Rotation Accuracy (vs Exp10/11)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)

    plt.suptitle("Exp12: Rotation Correlation Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities for exp12_rotation_correlation")
    print("Run the training script to generate visualizations.")
