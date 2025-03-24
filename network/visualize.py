#!/usr/bin/env python
"""Visualization script for puzzle piece predictions."""

import argparse
from typing import Tuple, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import PuzzleCNN


def load_model(checkpoint_path: str) -> PuzzleCNN:
    """Load a trained puzzle prediction model.

    Args:
        checkpoint_path: Path to model checkpoint file

    Returns:
        Loaded model
    """
    model = PuzzleCNN.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def preprocess_image(
    image_path: str, input_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Preprocess a puzzle piece image for model input.

    Args:
        image_path: Path to image file
        input_size: Target size for preprocessing

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W)
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(input_size)

    # Convert to numpy array
    image_np = np.array(image)

    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def visualize_prediction(
    model: PuzzleCNN,
    piece_path: str,
    puzzle_path: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Visualize puzzle piece prediction.

    Args:
        model: Trained puzzle model
        piece_path: Path to puzzle piece image
        puzzle_path: Optional path to complete puzzle image for comparison
        save_path: Optional path to save visualization
        show: Whether to display the visualization
    """
    # Preprocess piece image
    piece_tensor = preprocess_image(piece_path)

    # Get prediction
    with torch.no_grad():
        position, rotation_class, rotation_probs = model.predict_piece(piece_tensor)

    # Convert to numpy for visualization
    position_np = position.numpy()
    rotation_degrees = rotation_class * 90

    # Load original images
    piece_img = Image.open(piece_path).convert("RGB")
    piece_np = np.array(piece_img)

    # Create figure
    fig, axes = plt.subplots(1, 2 if puzzle_path else 1, figsize=(12, 6))

    if not puzzle_path:
        axes = [axes]  # Make axes iterable if only one subplot

    # Plot piece
    axes[0].imshow(piece_np)
    axes[0].set_title(f"Puzzle Piece\nPredicted Rotation: {rotation_degrees}°")
    axes[0].axis("off")

    # Plot puzzle with prediction if available
    if puzzle_path:
        puzzle_img = Image.open(puzzle_path).convert("RGB")
        puzzle_np = np.array(puzzle_img)

        # Get puzzle dimensions
        puzzle_h, puzzle_w, _ = puzzle_np.shape

        # Convert normalized coordinates to pixel coordinates
        x1 = int(position_np[0] * puzzle_w)
        y1 = int(position_np[1] * puzzle_h)
        x2 = int(position_np[2] * puzzle_w)
        y2 = int(position_np[3] * puzzle_h)

        # Plot puzzle
        axes[1].imshow(puzzle_np)
        axes[1].set_title("Puzzle with Predicted Position")

        # Create rectangle patch for predicted position
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
        )

        # Add patch to plot
        axes[1].add_patch(rect)
        axes[1].axis("off")

    # Add rotation probabilities as text
    rotation_text = "\n".join(
        [f"{i * 90}°: {prob:.2f}" for i, prob in enumerate(rotation_probs.numpy())]
    )
    plt.figtext(0.02, 0.02, f"Rotation Probabilities:\n{rotation_text}", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save visualization if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Show visualization if requested
    if show:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to run visualization."""
    # Create argument parser
    parser = argparse.ArgumentParser(description="Visualize puzzle piece predictions")

    # Add arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--piece", type=str, required=True, help="Path to puzzle piece image"
    )
    parser.add_argument(
        "--puzzle",
        type=str,
        default=None,
        help="Path to complete puzzle image (optional)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save visualization (optional)"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't display visualization"
    )

    # Parse arguments
    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint)

    # Visualize prediction
    visualize_prediction(
        model=model,
        piece_path=args.piece,
        puzzle_path=args.puzzle,
        save_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
