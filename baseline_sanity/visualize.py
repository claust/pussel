"""
Helper functions to visualize predictions vs ground truth.

Draws boxes on images:
- Red: Predicted center
- Green: Ground truth center
"""

import numpy as np
from pathlib import Path
import torch

# Try to import PIL, fall back to matplotlib if not available
try:
    from PIL import Image, ImageDraw

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def draw_center_marker(
    image: np.ndarray,
    cx: float,
    cy: float,
    color: tuple[int, int, int],
    marker_size: int = 5,
) -> np.ndarray:
    """
    Draw a cross marker at the center position.

    Args:
        image: RGB image as numpy array (H, W, 3)
        cx, cy: Normalized center coordinates [0, 1]
        color: RGB color tuple
        marker_size: Size of the cross marker

    Returns:
        Image with marker drawn
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Convert normalized coords to pixel coords
    px = int(cx * w)
    py = int(cy * h)

    # Draw cross
    for dx in range(-marker_size, marker_size + 1):
        x = px + dx
        if 0 <= x < w:
            img[py, x] = color
    for dy in range(-marker_size, marker_size + 1):
        y = py + dy
        if 0 <= y < h:
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
    """
    Draw a box centered at the given position.

    Args:
        image: RGB image as numpy array (H, W, 3)
        cx, cy: Normalized center coordinates [0, 1]
        box_size: Size of the box in pixels
        color: RGB color tuple
        thickness: Line thickness

    Returns:
        Image with box drawn
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Convert normalized coords to pixel coords
    px = int(cx * w)
    py = int(cy * h)

    # Box corners
    half = box_size // 2
    x1 = max(0, px - half)
    y1 = max(0, py - half)
    x2 = min(w - 1, px + half)
    y2 = min(h - 1, py + half)

    # Draw rectangle
    for t in range(thickness):
        # Top and bottom edges
        for x in range(x1, x2 + 1):
            if y1 + t < h:
                img[y1 + t, x] = color
            if y2 - t >= 0:
                img[y2 - t, x] = color
        # Left and right edges
        for y in range(y1, y2 + 1):
            if x1 + t < w:
                img[y, x1 + t] = color
            if x2 - t >= 0:
                img[y, x2 - t] = color

    return img


def visualize_prediction(
    image: torch.Tensor | np.ndarray,
    pred_xy: tuple[float, float],
    target_xy: tuple[float, float],
    square_size: int = 16,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """
    Visualize prediction vs ground truth on an image.

    Args:
        image: Image tensor (C, H, W) or numpy array (H, W, C)
        pred_xy: Predicted (cx, cy) normalized
        target_xy: Ground truth (cx, cy) normalized
        square_size: Size of the square being detected
        save_path: Optional path to save the image

    Returns:
        Annotated image as numpy array
    """
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        img = image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
    else:
        img = image.copy()

    # Draw ground truth box (green)
    img = draw_box_from_center(img, target_xy[0], target_xy[1], square_size, (0, 255, 0), thickness=2)

    # Draw predicted center marker (red)
    img = draw_center_marker(img, pred_xy[0], pred_xy[1], (255, 0, 0), marker_size=3)

    # Draw predicted box (red, thinner)
    img = draw_box_from_center(img, pred_xy[0], pred_xy[1], square_size, (255, 0, 0), thickness=1)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_PIL:
            pil_img = Image.fromarray(img)
            pil_img.save(save_path)
        elif HAS_MPL:
            plt.imsave(str(save_path), img)
        else:
            # Fallback: save as raw numpy
            np.save(str(save_path).replace(".png", ".npy"), img)

    return img


def create_grid_visualization(
    images: list[torch.Tensor],
    preds: list[tuple[float, float]],
    targets: list[tuple[float, float]],
    save_path: str | Path,
    square_size: int = 16,
    grid_size: tuple[int, int] = (2, 4),
) -> None:
    """
    Create a grid of visualizations.

    Args:
        images: List of image tensors
        preds: List of predicted (cx, cy)
        targets: List of ground truth (cx, cy)
        save_path: Path to save the grid image
        square_size: Size of the detected squares
        grid_size: (rows, cols) for the grid
    """
    rows, cols = grid_size
    n_samples = min(len(images), rows * cols)

    if HAS_MPL:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if rows * cols > 1 else [axes]

        for i in range(n_samples):
            vis_img = visualize_prediction(images[i], preds[i], targets[i], square_size)
            axes[i].imshow(vis_img)
            axes[i].axis("off")
            pred_str = f"P:({preds[i][0]:.2f},{preds[i][1]:.2f})"
            tgt_str = f"T:({targets[i][0]:.2f},{targets[i][1]:.2f})"
            axes[i].set_title(f"{pred_str}\n{tgt_str}", fontsize=8)

        # Hide unused axes
        for i in range(n_samples, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
    else:
        # Fallback: save individual images
        for i in range(n_samples):
            save_p = Path(save_path).parent / f"sample_{i}.png"
            visualize_prediction(images[i], preds[i], targets[i], square_size, save_p)


if __name__ == "__main__":
    # Quick test
    from dataset import SquareDataset

    dataset = SquareDataset(size=8, seed=42)

    images = []
    targets = []
    fake_preds = []

    for i in range(8):
        img, tgt = dataset[i]
        images.append(img)
        targets.append((tgt[0].item(), tgt[1].item()))
        # Fake predictions: slightly offset from ground truth
        fake_preds.append((tgt[0].item() + 0.05, tgt[1].item() - 0.03))

    create_grid_visualization(
        images, fake_preds, targets, "outputs/test_visualization.png", grid_size=(2, 4)
    )
    print("Saved test visualization to outputs/test_visualization.png")
