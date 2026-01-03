"""Visualization utilities for exp21: Masked rotation correlation.

Same as exp20 visualizations but with updated titles.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import CELL_CENTERS, GRID_SIZE, NUM_CELLS


def save_prediction_grid(
    predictions: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    pred_cells: list[int],
    true_cells: list[int],
    pred_rotations: list[int],
    true_rotations: list[int],
    output_path: Path,
) -> None:
    """Save visualization of predictions on 4x4 grid with rotation info."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Plot 1: Position scatter plot
    ax1 = axes[0]

    preds_x = [p[0] for p in predictions]
    preds_y = [p[1] for p in predictions]

    colors = ["green" if p == t else "red" for p, t in zip(pred_cells, true_cells)]

    ax1.scatter(preds_x, preds_y, c=colors, alpha=0.5, s=20, label="Predictions")

    for i in range(1, GRID_SIZE):
        ax1.axhline(y=i / GRID_SIZE, color="black", linestyle="--", alpha=0.5)
        ax1.axvline(x=i / GRID_SIZE, color="black", linestyle="--", alpha=0.5)

    for i, (cx, cy) in enumerate(CELL_CENTERS):
        ax1.plot(cx, cy, "k*", markersize=8, zorder=10)
        ax1.annotate(f"{i}", (cx, cy), xytext=(2, 2), textcoords="offset points", fontsize=7)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.invert_yaxis()
    ax1.set_xlabel("Predicted cx")
    ax1.set_ylabel("Predicted cy")

    cell_acc = sum(p == t for p, t in zip(pred_cells, true_cells)) / len(pred_cells)
    ax1.set_title(f"Position Predictions\nCell Accuracy: {cell_acc:.1%}")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cell confusion matrix (16x16)
    ax2 = axes[1]

    cell_confusion = np.zeros((NUM_CELLS, NUM_CELLS), dtype=int)
    for pred, true in zip(pred_cells, true_cells):
        cell_confusion[true, pred] += 1

    im2 = ax2.imshow(cell_confusion, cmap="Blues")
    ax2.set_xticks(range(NUM_CELLS))
    ax2.set_yticks(range(NUM_CELLS))
    ax2.set_xticklabels([str(i) for i in range(NUM_CELLS)], fontsize=6)
    ax2.set_yticklabels([str(i) for i in range(NUM_CELLS)], fontsize=6)
    ax2.set_xlabel("Predicted Cell")
    ax2.set_ylabel("True Cell")
    ax2.set_title(f"Cell Confusion Matrix ({GRID_SIZE}x{GRID_SIZE})")

    max_val = cell_confusion.max()
    for i in range(NUM_CELLS):
        for j in range(NUM_CELLS):
            if cell_confusion[i, j] > 0:
                color = "white" if cell_confusion[i, j] > max_val / 2 else "black"
                ax2.text(
                    j,
                    i,
                    str(cell_confusion[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=5,
                )

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
        f"Exp21: Masked Rotation - Cell: {cell_acc:.1%}, Rot: {rot_acc:.1%}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curves(
    history: dict[str, list[float]],
    output_path: Path,
) -> None:
    """Save training curves visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history["train_pos_loss"]) + 1)

    # Plot 1: Position Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_pos_loss"], "b-", label="Train", linewidth=2)
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

    # Plot 3: Cell Accuracy
    ax3 = axes[1, 0]
    ax3.plot(
        epochs,
        [a * 100 for a in history["train_cell_acc"]],
        "b-",
        label="Train",
        linewidth=2,
    )
    if "test_cell_acc" in history and history["test_cell_acc"]:
        ax3.plot(
            epochs,
            [a * 100 for a in history["test_cell_acc"]],
            "r-",
            label="Test",
            linewidth=2,
        )

    random_baseline = 100.0 / NUM_CELLS
    ax3.axhline(
        y=random_baseline,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"Random ({random_baseline:.1f}%)",
    )
    ax3.axhline(y=70, color="green", linestyle="--", alpha=0.7, label="Target (70%)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title(f"Cell Accuracy ({GRID_SIZE}x{GRID_SIZE} Grid)")
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
    if "test_rot_acc" in history and history["test_rot_acc"]:
        ax4.plot(
            epochs,
            [a * 100 for a in history["test_rot_acc"]],
            "r-",
            label="Test",
            linewidth=2,
        )

    ax4.axhline(y=25, color="gray", linestyle=":", alpha=0.7, label="Random (25%)")
    ax4.axhline(y=50, color="green", linestyle="--", alpha=0.7, label="Target (50%)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.set_title("Rotation Accuracy (Masked Correlation)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 105)

    plt.suptitle("Exp21: Masked Rotation Correlation - Training Curves", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Visualization utilities for exp21_masked_rotation")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells")
    print("Run train_cuda.py to generate visualizations.")
