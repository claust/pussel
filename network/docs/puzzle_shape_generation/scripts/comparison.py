"""Comparison logic for puzzle pieces."""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from geometry import generate_piece_path
from io_utils import load_pieces_from_json

# PIECE_COLOR should be imported from rendering to stay consistent
from rendering import PIECE_COLOR


def _extract_and_normalize_contour(image_path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    """Extract contour from reference image and normalize to unit coordinates."""
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Get alpha channel or convert to grayscale
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    contour = largest.reshape(-1, 2).astype(np.float64)

    # Normalize: center and scale to fit in [0, 1] range
    min_pt = contour.min(axis=0)
    max_pt = contour.max(axis=0)
    size = (max_pt - min_pt).max()

    if size > 0:
        # Center the contour
        center = (min_pt + max_pt) / 2
        contour = contour - center
        # Scale to unit size
        contour = contour / size
        # Shift to [0, 1] centered at 0.5
        contour = contour + 0.5

    # Flip Y axis (image coordinates are inverted)
    contour[:, 1] = 1.0 - contour[:, 1]

    return contour[:, 0], contour[:, 1]


def generate_reference_comparison(
    output_path: str | Path = "../outputs/reference_comparison.png",
    json_path: str | Path | None = None,
) -> Path:
    """Generate a side-by-side comparison of reference pieces and generated matches."""
    from PIL import Image

    output_path = Path(output_path)
    ref_dir = Path(__file__).parent.parent / "reference_images" / "standardized"

    # Load pieces from JSON file
    if json_path is None:
        json_path = Path(__file__).parent.parent / "reference_pieces.json"
    pieces = load_pieces_from_json(json_path)

    num_pieces = len(pieces)
    fig, axes = plt.subplots(3, num_pieces, figsize=(3 * num_pieces, 9))
    fig.suptitle("Reference vs Generated Pieces", fontsize=16, fontweight="bold")

    for i, config in enumerate(pieces):
        # Top row: reference pieces
        ref_img_path = ref_dir / f"piece_{i + 1}.png"
        if ref_img_path.exists():
            ref_img = Image.open(ref_img_path)
            axes[0, i].imshow(ref_img)
        axes[0, i].set_title(f"Reference {i + 1}")
        axes[0, i].axis("off")

        # Middle row: generated pieces from JSON
        x_coords, y_coords = generate_piece_path(config)

        axes[1, i].fill(x_coords, y_coords, color=PIECE_COLOR, edgecolor=PIECE_COLOR, linewidth=2)
        axes[1, i].set_aspect("equal")
        axes[1, i].axis("off")
        margin = 0.35 * config.size
        axes[1, i].set_xlim(-margin, config.size + margin)
        axes[1, i].set_ylim(-margin, config.size + margin)
        axes[1, i].set_title(f"Generated {i + 1}")

        # Bottom row: overlapped comparison
        ax = axes[2, i]

        # Normalize generated piece to [0, 1] range for comparison
        gen_x = np.array(x_coords)
        gen_y = np.array(y_coords)
        gen_min = min(gen_x.min(), gen_y.min())
        gen_max = max(gen_x.max(), gen_y.max())
        gen_size = gen_max - gen_min
        if gen_size > 0:
            gen_center_x = (gen_x.min() + gen_x.max()) / 2
            gen_center_y = (gen_y.min() + gen_y.max()) / 2
            gen_x_norm = (gen_x - gen_center_x) / gen_size + 0.5
            gen_y_norm = (gen_y - gen_center_y) / gen_size + 0.5
        else:
            gen_x_norm, gen_y_norm = gen_x, gen_y

        # Extract and normalize reference contour
        ref_contour = _extract_and_normalize_contour(ref_img_path)

        if ref_contour is not None:
            ref_x, ref_y = ref_contour
            # Plot reference shape (blue, semi-transparent)
            ax.fill(ref_x, ref_y, color="blue", alpha=0.4, label="Reference")
            ax.plot(ref_x, ref_y, color="blue", linewidth=1.5, alpha=0.8)

        # Plot generated shape (red, semi-transparent)
        ax.fill(gen_x_norm, gen_y_norm, color="red", alpha=0.4, label="Generated")
        ax.plot(gen_x_norm, gen_y_norm, color="red", linewidth=1.5, alpha=0.8)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Overlap {i + 1}")

    # Add legend to first overlap plot
    axes[2, 0].legend(loc="upper left", fontsize=8)

    axes[0, 0].set_ylabel("Reference", fontsize=12)
    axes[1, 0].set_ylabel("Generated", fontsize=12)
    axes[2, 0].set_ylabel("Overlap", fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path
