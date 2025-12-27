"""Visualization helpers for cell classification experiment.

Key visualization: Heatmap overlay showing probability distribution over cells.
This is a major benefit of classification - we can see where the model thinks
the piece belongs and its confidence across all cells.
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

    # Reshape to grid
    heatmap = probs_arr.reshape(num_rows, num_cols)
    return heatmap


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
        # Use matplotlib for colormap
        from scipy.ndimage import zoom

        scale_y = h / num_rows
        scale_x = w / num_cols
        heatmap_resized = zoom(heatmap, (scale_y, scale_x), order=1)

        # Apply colormap (hot: black -> red -> yellow -> white)
        cmap = plt.cm.hot  # type: ignore[attr-defined]
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # RGB only
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Blend with original image
        overlay = (1 - alpha) * img.astype(float) + alpha * heatmap_colored.astype(float)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        # Simple grayscale overlay without matplotlib
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


def create_grid_comparison(
    pieces: list[torch.Tensor],
    probs_list: list[torch.Tensor | np.ndarray],
    targets: list[int],
    preds: list[int],
    num_cols: int,
    num_rows: int,
    save_path: str | Path,
    grid_size: tuple[int, int] = (2, 4),
) -> None:
    """Create a grid showing pieces with their probability heatmaps.

    Args:
        pieces: List of piece image tensors.
        probs_list: List of probability vectors.
        targets: List of target cell indices.
        preds: List of predicted cell indices.
        num_cols: Number of columns in puzzle grid.
        num_rows: Number of rows in puzzle grid.
        save_path: Path to save the grid.
        grid_size: (rows, cols) for the visualization grid.
    """
    if plt is None:
        print("Warning: matplotlib not available, skipping grid comparison")
        return

    rows, cols = grid_size
    n_samples = min(len(pieces), rows * cols)

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2))

    for i in range(n_samples):
        row_idx = i // cols
        col_idx = i % cols

        # Piece image
        ax_piece = axes[row_idx, col_idx * 2] if rows > 1 else axes[col_idx * 2]
        piece_img = pieces[i].permute(1, 2, 0).cpu().numpy()
        ax_piece.imshow(piece_img)
        ax_piece.axis("off")
        ax_piece.set_title(f"Piece {i}", fontsize=8)

        # Heatmap
        ax_heatmap = axes[row_idx, col_idx * 2 + 1] if rows > 1 else axes[col_idx * 2 + 1]
        heatmap = probs_to_heatmap(probs_list[i], num_cols, num_rows)
        ax_heatmap.imshow(heatmap, cmap="hot", aspect="auto")
        ax_heatmap.axis("off")

        # Mark target and prediction
        target_row = targets[i] // num_cols
        target_col = targets[i] % num_cols
        pred_row = preds[i] // num_cols
        pred_col = preds[i] % num_cols

        correct = targets[i] == preds[i]
        status = "OK" if correct else "X"
        ax_heatmap.set_title(f"T:({target_col},{target_row}) P:({pred_col},{pred_row}) {status}", fontsize=7)

    # Hide unused axes
    for i in range(n_samples, rows * cols):
        row_idx = i // cols
        col_idx = i % cols
        if rows > 1:
            axes[row_idx, col_idx * 2].axis("off")
            axes[row_idx, col_idx * 2 + 1].axis("off")
        else:
            axes[col_idx * 2].axis("off")
            axes[col_idx * 2 + 1].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()


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


if __name__ == "__main__":
    import torch.nn.functional as F

    from .dataset import CellClassificationDataset

    print("Testing visualization functions...")

    # Load dataset
    dataset = CellClassificationDataset(
        puzzle_id="puzzle_001",
        piece_size=64,
        puzzle_size=256,
    )

    # Get some samples
    pieces = []
    targets = []

    for i in range(8):
        piece, cell_index = dataset[i]
        pieces.append(piece)
        targets.append(cell_index.item())

    # Create fake probability distributions
    # For testing: put 90% probability on target, spread rest
    probs_list = []
    preds = []
    for target in targets:
        probs = torch.ones(dataset.num_cells) * 0.1 / (dataset.num_cells - 1)
        probs[target] = 0.9
        probs = F.softmax(probs * 10, dim=0)  # Sharpen
        probs_list.append(probs)
        preds.append(target)  # Perfect prediction for test

    # Test heatmap overlay
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    puzzle_tensor = dataset.get_puzzle_image()
    save_heatmap_overlay(
        puzzle_tensor,
        probs_list[0],
        dataset.num_cols,
        dataset.num_rows,
        output_dir / "test_heatmap.png",
        target_cell=targets[0],
        pred_cell=preds[0],
    )
    print(f"Saved heatmap to {output_dir / 'test_heatmap.png'}")

    # Test grid comparison
    create_grid_comparison(
        pieces,
        probs_list,
        targets,
        preds,
        dataset.num_cols,
        dataset.num_rows,
        output_dir / "test_grid.png",
        grid_size=(2, 4),
    )
    print(f"Saved grid to {output_dir / 'test_grid.png'}")

    # Test accuracy grid
    save_accuracy_grid(
        puzzle_tensor,
        targets,
        preds,
        dataset.num_cols,
        dataset.num_rows,
        output_dir / "test_accuracy.png",
    )
    print(f"Saved accuracy grid to {output_dir / 'test_accuracy.png'}")

    print("Visualization test complete!")
