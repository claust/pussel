"""Visualization helpers for cross-puzzle generalization experiment.

Provides heatmap overlays and accuracy grids to visualize how well
a model trained on one puzzle performs on another puzzle.
"""

from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

# Optional imports
Image: Any = None
plt: ModuleType | None = None

try:
    from PIL import Image
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt  # type: ignore[no-redef]
except ImportError:
    pass


def probs_to_heatmap(
    probs: torch.Tensor | np.ndarray,
    num_cols: int,
    num_rows: int,
) -> np.ndarray:
    """Convert probability vector to 2D heatmap.

    Args:
        probs: Probability vector of shape (num_cells,).
        num_cols: Number of columns in the grid.
        num_rows: Number of rows in the grid.

    Returns:
        Heatmap array of shape (num_rows, num_cols).
    """
    probs_arr: np.ndarray = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs
    heatmap = probs_arr.reshape(num_rows, num_cols)
    return heatmap


def draw_cell_border(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle border on the image.

    Args:
        image: RGB image as numpy array (H, W, 3).
        x1, y1, x2, y2: Rectangle coordinates.
        color: RGB color tuple.
        thickness: Line thickness.

    Returns:
        Image with border drawn.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Clamp coordinates
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # Draw horizontal lines
    for t in range(thickness):
        if y1 + t < h:
            img[y1 + t, x1:x2] = color
        if y2 - t >= 0 and y2 - t < h:
            img[y2 - t, x1:x2] = color

    # Draw vertical lines
    for t in range(thickness):
        if x1 + t < w:
            img[y1:y2, x1 + t] = color
        if x2 - t >= 0 and x2 - t < w:
            img[y1:y2, x2 - t] = color

    return img


def create_heatmap_overlay(
    puzzle_image: torch.Tensor | np.ndarray,
    probs: torch.Tensor | np.ndarray,
    num_cols: int,
    num_rows: int,
    alpha: float = 0.5,
    target_cell: int | None = None,
    pred_cell: int | None = None,
) -> np.ndarray:
    """Overlay probability heatmap on puzzle image.

    Args:
        puzzle_image: Puzzle tensor (C, H, W) or numpy array (H, W, C).
        probs: Probability vector of shape (num_cells,).
        num_cols: Number of columns in the grid.
        num_rows: Number of rows in the grid.
        alpha: Transparency of heatmap overlay.
        target_cell: If provided, mark the target cell with a green border.
        pred_cell: If provided, mark the predicted cell with a red border.

    Returns:
        Overlay image as numpy array (H, W, 3).
    """
    # Convert puzzle image to numpy
    if isinstance(puzzle_image, torch.Tensor):
        img = puzzle_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
    else:
        img = puzzle_image.copy()

    h, w = img.shape[:2]

    # Create heatmap
    heatmap = probs_to_heatmap(probs, num_cols, num_rows)

    # Resize heatmap to image size
    if plt is not None:
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.ndimage import zoom

        scale_y = h / num_rows
        scale_x = w / num_cols
        heatmap_resized = zoom(heatmap, (scale_y, scale_x), order=1)

        colors = [(0.1, 0.1, 0.3), (1.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
        cmap = LinearSegmentedColormap.from_list("prob_heatmap", colors)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        overlay = (1 - alpha) * img.astype(float) + alpha * heatmap_colored.astype(float)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        from scipy.ndimage import zoom

        scale_y = h / num_rows
        scale_x = w / num_cols
        heatmap_resized = zoom(heatmap, (scale_y, scale_x), order=1)
        heatmap_gray = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = np.stack([heatmap_gray, heatmap_gray // 2, np.zeros_like(heatmap_gray)], axis=-1)

        overlay = (1 - alpha) * img.astype(float) + alpha * heatmap_colored.astype(float)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw cell borders for target and prediction
    cell_h = h / num_rows
    cell_w = w / num_cols

    if target_cell is not None:
        target_row = target_cell // num_cols
        target_col = target_cell % num_cols
        y1 = int(target_row * cell_h)
        y2 = int((target_row + 1) * cell_h)
        x1 = int(target_col * cell_w)
        x2 = int((target_col + 1) * cell_w)
        overlay = draw_cell_border(overlay, x1, y1, x2, y2, color=(0, 255, 0), thickness=2)

    if pred_cell is not None:
        pred_row = pred_cell // num_cols
        pred_col = pred_cell % num_cols
        y1 = int(pred_row * cell_h)
        y2 = int((pred_row + 1) * cell_h)
        x1 = int(pred_col * cell_w)
        x2 = int((pred_col + 1) * cell_w)
        overlay = draw_cell_border(overlay, x1, y1, x2, y2, color=(255, 0, 0), thickness=2)

    return overlay


def save_heatmap_overlay(
    puzzle_image: torch.Tensor,
    probs: torch.Tensor | np.ndarray,
    num_cols: int,
    num_rows: int,
    save_path: str | Path,
    target_cell: int | None = None,
    pred_cell: int | None = None,
    alpha: float = 0.5,
) -> None:
    """Save heatmap overlay to file.

    Args:
        puzzle_image: Puzzle tensor (C, H, W).
        probs: Probability vector.
        num_cols: Number of columns.
        num_rows: Number of rows.
        save_path: Path to save the image.
        target_cell: Optional target cell index.
        pred_cell: Optional predicted cell index.
        alpha: Heatmap transparency.
    """
    overlay = create_heatmap_overlay(
        puzzle_image,
        probs,
        num_cols,
        num_rows,
        alpha=alpha,
        target_cell=target_cell,
        pred_cell=pred_cell,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is not None:
        pil_img = Image.fromarray(overlay)
        pil_img.save(save_path)
    elif plt is not None:
        plt.imsave(str(save_path), overlay)
    else:
        np.save(str(save_path).replace(".png", ".npy"), overlay)


def save_accuracy_grid(
    puzzle_image: torch.Tensor,
    all_targets: list[int],
    all_preds: list[int],
    num_cols: int,
    num_rows: int,
    save_path: str | Path,
) -> None:
    """Save a grid showing correct (green) and incorrect (red) predictions.

    Args:
        puzzle_image: Puzzle tensor (C, H, W).
        all_targets: List of all target cell indices.
        all_preds: List of all predicted cell indices.
        num_cols: Number of columns.
        num_rows: Number of rows.
        save_path: Path to save the image.
    """
    # Convert puzzle image to numpy
    img = puzzle_image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]

    cell_h = h / num_rows
    cell_w = w / num_cols

    # Draw each cell with color based on correctness
    for target, pred in zip(all_targets, all_preds):
        row = target // num_cols
        col = target % num_cols

        y1 = int(row * cell_h)
        y2 = int((row + 1) * cell_h)
        x1 = int(col * cell_w)
        x2 = int((col + 1) * cell_w)

        color = (0, 255, 0) if target == pred else (255, 0, 0)
        img = draw_cell_border(img, x1, y1, x2, y2, color, thickness=1)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is not None:
        pil_img = Image.fromarray(img)
        pil_img.save(save_path)
    elif plt is not None:
        plt.imsave(str(save_path), img)


def create_comparison_figure(
    train_puzzle_img: torch.Tensor,
    test_puzzle_img: torch.Tensor,
    train_targets: list[int],
    train_preds: list[int],
    test_targets: list[int],
    test_preds: list[int],
    num_cols: int,
    num_rows: int,
    save_path: str | Path,
    train_puzzle_id: str,
    test_puzzle_id: str,
) -> None:
    """Create a side-by-side comparison of train vs test puzzle accuracy.

    Args:
        train_puzzle_img: Training puzzle image tensor.
        test_puzzle_img: Test puzzle image tensor.
        train_targets: Training targets.
        train_preds: Training predictions.
        test_targets: Test targets.
        test_preds: Test predictions.
        num_cols: Number of columns.
        num_rows: Number of rows.
        save_path: Path to save the figure.
        train_puzzle_id: ID of training puzzle.
        test_puzzle_id: ID of test puzzle.
    """
    if plt is None:
        print("Warning: matplotlib not available, skipping comparison figure")
        return

    # Calculate accuracies
    train_correct = sum(t == p for t, p in zip(train_targets, train_preds))
    train_acc = train_correct / len(train_targets)
    test_correct = sum(t == p for t, p in zip(test_targets, test_preds))
    test_acc = test_correct / len(test_targets)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Training puzzle
    train_img = train_puzzle_img.permute(1, 2, 0).cpu().numpy()
    train_img = (train_img * 255).astype(np.uint8).copy()
    h, w = train_img.shape[:2]
    cell_h = h / num_rows
    cell_w = w / num_cols

    for target, pred in zip(train_targets, train_preds):
        row = target // num_cols
        col = target % num_cols
        y1 = int(row * cell_h)
        y2 = int((row + 1) * cell_h)
        x1 = int(col * cell_w)
        x2 = int((col + 1) * cell_w)
        color = (0, 255, 0) if target == pred else (255, 0, 0)
        train_img = draw_cell_border(train_img, x1, y1, x2, y2, color, thickness=1)

    axes[0].imshow(train_img)
    axes[0].set_title(f"Train: {train_puzzle_id}\nAccuracy: {train_acc:.1%}")
    axes[0].axis("off")

    # Test puzzle
    test_img = test_puzzle_img.permute(1, 2, 0).cpu().numpy()
    test_img = (test_img * 255).astype(np.uint8).copy()

    for target, pred in zip(test_targets, test_preds):
        row = target // num_cols
        col = target % num_cols
        y1 = int(row * cell_h)
        y2 = int((row + 1) * cell_h)
        x1 = int(col * cell_w)
        x2 = int((col + 1) * cell_w)
        color = (0, 255, 0) if target == pred else (255, 0, 0)
        test_img = draw_cell_border(test_img, x1, y1, x2, y2, color, thickness=1)

    axes[1].imshow(test_img)
    axes[1].set_title(f"Test: {test_puzzle_id}\nAccuracy: {test_acc:.1%}")
    axes[1].axis("off")

    plt.suptitle(f"Cross-Puzzle Generalization: Train on {train_puzzle_id}, Test on {test_puzzle_id}")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
