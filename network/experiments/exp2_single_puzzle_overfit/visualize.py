"""Visualization helpers for single puzzle overfit experiment.

Functions to visualize:
1. Piece grid with predictions (similar to baseline)
2. Predictions overlaid on the full puzzle image
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


def draw_center_marker(
    image: np.ndarray,
    cx: float,
    cy: float,
    color: tuple[int, int, int],
    marker_size: int = 5,
) -> np.ndarray:
    """Draw a cross marker at the center position.

    Args:
        image: RGB image as numpy array (H, W, 3).
        cx, cy: Normalized center coordinates [0, 1].
        color: RGB color tuple.
        marker_size: Size of the cross marker in pixels.

    Returns:
        Image with marker drawn.
    """
    img = image.copy()
    h, w = img.shape[:2]

    px = int(cx * w)
    py = int(cy * h)

    # Draw cross
    for dx in range(-marker_size, marker_size + 1):
        x = px + dx
        if 0 <= x < w and 0 <= py < h:
            img[py, x] = color
    for dy in range(-marker_size, marker_size + 1):
        y = py + dy
        if 0 <= y < h and 0 <= px < w:
            img[y, px] = color

    return img


def draw_box_from_center(
    image: np.ndarray,
    cx: float,
    cy: float,
    box_size: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """Draw a box centered at the given position.

    Args:
        image: RGB image as numpy array (H, W, 3).
        cx, cy: Normalized center coordinates [0, 1].
        box_size: Size of the box in pixels.
        color: RGB color tuple.
        thickness: Line thickness.

    Returns:
        Image with box drawn.
    """
    img = image.copy()
    h, w = img.shape[:2]

    px = int(cx * w)
    py = int(cy * h)

    half = box_size // 2
    x1 = max(0, px - half)
    y1 = max(0, py - half)
    x2 = min(w - 1, px + half)
    y2 = min(h - 1, py + half)

    for t in range(thickness):
        for x in range(x1, x2 + 1):
            if y1 + t < h:
                img[y1 + t, x] = color
            if y2 - t >= 0:
                img[y2 - t, x] = color
        for y in range(y1, y2 + 1):
            if x1 + t < w:
                img[y, x1 + t] = color
            if x2 - t >= 0:
                img[y, x2 - t] = color

    return img


def draw_line(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw a line between two points using Bresenham's algorithm.

    Args:
        image: RGB image as numpy array (H, W, 3).
        x1, y1: Start point (pixels).
        x2, y2: End point (pixels).
        color: RGB color tuple.

    Returns:
        Image with line drawn.
    """
    img = image.copy()
    h, w = img.shape[:2]

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < w and 0 <= y1 < h:
            img[y1, x1] = color

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return img


def visualize_piece_prediction(
    piece_image: torch.Tensor | np.ndarray,
    pred_xy: tuple[float, float],
    target_xy: tuple[float, float],
    box_size: int = 16,
) -> np.ndarray:
    """Visualize prediction vs ground truth on a piece image.

    Args:
        piece_image: Piece tensor (C, H, W) or numpy array (H, W, C).
        pred_xy: Predicted (cx, cy) normalized.
        target_xy: Ground truth (cx, cy) normalized.
        box_size: Size of marker boxes.

    Returns:
        Annotated image as numpy array.
    """
    if isinstance(piece_image, torch.Tensor):
        img = piece_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
    else:
        img = piece_image.copy()

    # Draw ground truth box (green)
    img = draw_box_from_center(img, target_xy[0], target_xy[1], box_size, (0, 255, 0), thickness=2)

    # Draw predicted center marker and box (red)
    img = draw_center_marker(img, pred_xy[0], pred_xy[1], (255, 0, 0), marker_size=3)
    img = draw_box_from_center(img, pred_xy[0], pred_xy[1], box_size, (255, 0, 0), thickness=1)

    return img


def create_grid_visualization(
    pieces: list[torch.Tensor],
    preds: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    save_path: str | Path,
    box_size: int = 16,
    grid_size: tuple[int, int] = (2, 4),
) -> None:
    """Create a grid of piece predictions.

    Args:
        pieces: List of piece image tensors (C, H, W).
        preds: List of predicted (cx, cy).
        targets: List of ground truth (cx, cy).
        save_path: Path to save the grid image.
        box_size: Size of marker boxes.
        grid_size: (rows, cols) for the grid.
    """
    rows, cols = grid_size
    n_samples = min(len(pieces), rows * cols)

    if plt is not None:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes_flat = axes.flatten() if rows * cols > 1 else [axes]

        for i in range(n_samples):
            vis_img = visualize_piece_prediction(pieces[i], preds[i], targets[i], box_size)
            axes_flat[i].imshow(vis_img)
            axes_flat[i].axis("off")
            pred_str = f"P:({preds[i][0]:.2f},{preds[i][1]:.2f})"
            tgt_str = f"T:({targets[i][0]:.2f},{targets[i][1]:.2f})"
            axes_flat[i].set_title(f"{pred_str}\n{tgt_str}", fontsize=8)

        for i in range(n_samples, len(axes_flat)):
            axes_flat[i].axis("off")

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        print("Warning: matplotlib not available, skipping grid visualization")


def save_prediction_overlay(
    puzzle_image: torch.Tensor,
    preds: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    save_path: str | Path,
    marker_size: int = 8,
) -> None:
    """Overlay predictions on the puzzle image.

    Draws lines from predicted to target positions to show error.

    Args:
        puzzle_image: Puzzle tensor (C, H, W).
        preds: List of predicted (cx, cy) normalized.
        targets: List of ground truth (cx, cy) normalized.
        save_path: Path to save the overlay image.
        marker_size: Size of center markers.
    """
    # Convert to numpy
    img = puzzle_image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]

    for pred, target in zip(preds, targets):
        # Draw target marker (green)
        img = draw_center_marker(img, target[0], target[1], (0, 255, 0), marker_size)

        # Draw predicted marker (red)
        img = draw_center_marker(img, pred[0], pred[1], (255, 0, 0), marker_size)

        # Draw line from prediction to target (yellow)
        px_pred = int(pred[0] * w)
        py_pred = int(pred[1] * h)
        px_tgt = int(target[0] * w)
        py_tgt = int(target[1] * h)
        img = draw_line(img, px_pred, py_pred, px_tgt, py_tgt, (255, 255, 0))

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is not None:
        pil_img = Image.fromarray(img)
        pil_img.save(save_path)
    elif plt is not None:
        plt.imsave(str(save_path), img)
    else:
        np.save(str(save_path).replace(".png", ".npy"), img)


def create_full_overlay(
    puzzle_image: torch.Tensor,
    all_preds: list[tuple[float, float]],
    all_targets: list[tuple[float, float]],
    save_path: str | Path,
) -> None:
    """Create overlay with ALL predictions on the puzzle.

    Useful for seeing the overall prediction distribution.

    Args:
        puzzle_image: Puzzle tensor (C, H, W).
        all_preds: All predicted positions.
        all_targets: All ground truth positions.
        save_path: Path to save the image.
    """
    img = puzzle_image.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    h, w = img.shape[:2]

    # Draw all targets as small green dots
    for target in all_targets:
        tx = int(target[0] * w)
        ty = int(target[1] * h)
        if 0 <= tx < w and 0 <= ty < h:
            # Small green square
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= tx + dx < w and 0 <= ty + dy < h:
                        img[ty + dy, tx + dx] = (0, 255, 0)

    # Draw all predictions as small red dots
    for pred in all_preds:
        px = int(pred[0] * w)
        py = int(pred[1] * h)
        if 0 <= px < w and 0 <= py < h:
            img[py, px] = (255, 0, 0)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is not None:
        pil_img = Image.fromarray(img)
        pil_img.save(save_path)
    elif plt is not None:
        plt.imsave(str(save_path), img)


if __name__ == "__main__":
    from .dataset import SinglePuzzleDataset

    print("Testing visualization functions...")

    # Load dataset
    dataset = SinglePuzzleDataset(
        puzzle_id="puzzle_001",
        piece_size=64,
        puzzle_size=256,
    )

    # Get some samples
    pieces = []
    targets = []
    fake_preds = []

    for i in range(8):
        piece, target = dataset[i]
        pieces.append(piece)
        targets.append((target[0].item(), target[1].item()))
        # Fake predictions: slightly offset
        fake_preds.append((target[0].item() + 0.05, target[1].item() - 0.03))

    # Test grid visualization
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    create_grid_visualization(pieces, fake_preds, targets, output_dir / "test_grid.png", grid_size=(2, 4))
    print(f"Saved grid to {output_dir / 'test_grid.png'}")

    # Test overlay visualization
    puzzle_tensor = dataset.get_puzzle_image()
    save_prediction_overlay(puzzle_tensor, fake_preds, targets, output_dir / "test_overlay.png")
    print(f"Saved overlay to {output_dir / 'test_overlay.png'}")

    print("Visualization test complete!")
