#!/usr/bin/env python
"""
Puzzle Piece Visualizer.

This utility visualizes a puzzle piece's placement on the original puzzle.
It creates an image showing the original puzzle with the piece overlaid
and a red outline marking its correct position.
"""

# flake8: noqa: I100, I201
import argparse
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw


def parse_piece_filename(filename):
    """
    Parse a puzzle piece filename to extract position and rotation information.

    Expected format: piece_NNN_xLEFT_yTOP_xRIGHT_yBOTTOM_rROTATION.png

    Args:
        filename: Puzzle piece filename

    Returns:
        tuple: (piece_id, x1, y1, x2, y2, rotation)
    """
    pattern = r"piece_(\d+)_x(\d+)_y(\d+)_x(\d+)_y(\d+)_r(\d+)"
    match = re.match(pattern, os.path.basename(filename))

    if not match:
        raise ValueError(f"Invalid puzzle piece filename format: {filename}")

    piece_id = int(match.group(1))
    x1 = int(match.group(2))
    y1 = int(match.group(3))
    x2 = int(match.group(4))
    y2 = int(match.group(5))
    rotation = int(match.group(6))

    return piece_id, x1, y1, x2, y2, rotation


def _create_semi_transparent_overlay(piece):
    """Create a semi-transparent version of the piece for overlay.

    Args:
        piece: PIL Image of the puzzle piece

    Returns:
        PIL Image with semi-transparency applied
    """
    piece_array = np.array(piece)
    if piece_array.shape[2] == 4:  # If the piece has an alpha channel
        # Make non-transparent pixels semi-transparent (50%)
        mask = piece_array[:, :, 3] > 0
        piece_array[mask, 3] = 128  # 50% opacity

    return Image.fromarray(piece_array)


def _get_output_path(piece_path, puzzle_path, output_path=None):
    """Determine the output path for the visualization.

    Args:
        piece_path: Path to the puzzle piece image
        puzzle_path: Path to the full puzzle image
        output_path: User-specified output path

    Returns:
        Path to save the visualization
    """
    if not output_path:
        piece_name = os.path.splitext(os.path.basename(piece_path))[0]
        output_dir = os.path.dirname(puzzle_path)
        return os.path.join(output_dir, f"temp_{piece_name}.png")
    return output_path


def _draw_piece_outline(draw, bbox, color=(255, 0, 0, 255), thickness=3):
    """Draw an outline around the piece's position.

    Args:
        draw: ImageDraw object
        bbox: Bounding box tuple (x1, y1, x2, y2)
        color: Outline color as (R, G, B, A)
        thickness: Outline thickness in pixels
    """
    x1, y1, x2, y2 = bbox
    for i in range(thickness):
        draw.rectangle(
            (x1 - i, y1 - i, x2 + i, y2 + i),
            outline=color,
        )


def _calculate_paste_position(bbox, overlay_piece):
    """Calculate the position for pasting a piece overlay.

    Args:
        bbox: Bounding box tuple (x1, y1, x2, y2)
        overlay_piece: PIL Image to be pasted

    Returns:
        Tuple of (paste_x, paste_y) coordinates
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    paste_x = center_x - overlay_piece.width // 2
    paste_y = center_y - overlay_piece.height // 2
    return paste_x, paste_y


def _load_and_prepare_images(puzzle_path, piece_path, rotation):
    """Load and prepare the puzzle and piece images.

    Args:
        puzzle_path: Path to the full puzzle image
        piece_path: Path to the puzzle piece image
        rotation: Rotation to apply to the piece (in degrees)

    Returns:
        Tuple of (puzzle_image, piece_image)
    """
    # Load images
    puzzle = Image.open(puzzle_path).convert("RGBA")
    piece = Image.open(piece_path).convert("RGBA")

    # Rotate piece back to original orientation if needed
    if rotation != 0:
        reverse_rotation = (360 - rotation) % 360
        piece = piece.rotate(reverse_rotation, expand=True)

    return puzzle, piece


def _extract_piece_info(piece_path):
    """Extract piece information from the filename.

    Args:
        piece_path: Path to the puzzle piece image

    Returns:
        Tuple of (bbox, rotation) where bbox is (x1, y1, x2, y2)
    """
    filename = os.path.basename(piece_path)
    _, x1, y1, x2, y2, rotation = parse_piece_filename(filename)
    return (x1, y1, x2, y2), rotation


def visualize_piece_placement(puzzle_path, piece_path, output_path=None):
    """Create a visualization of a puzzle piece's placement.

    The visualization shows the piece on the original puzzle with a
    red outline marking its correct position.

    Args:
        puzzle_path: Path to the full puzzle image
        piece_path: Path to the puzzle piece image
        output_path: Path to save the visualization
            (default: temp_{piece_name}.png)

    Returns:
        Path to the output visualization
    """
    # Extract piece info from filename
    bbox, rotation = _extract_piece_info(piece_path)

    # Load and prepare images
    puzzle, piece = _load_and_prepare_images(puzzle_path, piece_path, rotation)

    # Create visualization canvas
    visualization = puzzle.copy()
    draw = ImageDraw.Draw(visualization)

    # Draw outline and create overlay
    _draw_piece_outline(draw, bbox)
    overlay_piece = _create_semi_transparent_overlay(piece)

    # Position and paste the overlay
    paste_position = _calculate_paste_position(bbox, overlay_piece)
    visualization.paste(overlay_piece, paste_position, overlay_piece)

    # Save and return result
    final_output_path = _get_output_path(piece_path, puzzle_path, output_path)
    visualization.save(final_output_path)
    print(f"Visualization saved to {final_output_path}")

    return final_output_path


def main():
    """Process command-line arguments and run the puzzle piece visualizer."""
    parser = argparse.ArgumentParser(
        description="Visualize puzzle piece placement on the original puzzle"
    )
    parser.add_argument("puzzle", help="Path to the original puzzle image")
    parser.add_argument("piece", help="Path to the puzzle piece image")
    parser.add_argument(
        "--output",
        help="Path to save the visualization (default: temp_{piece_name}.png)",
    )

    args = parser.parse_args()

    # Use absolute paths
    puzzle_path = os.path.abspath(args.puzzle)
    piece_path = os.path.abspath(args.piece)
    output_path = args.output
    if output_path:
        output_path = os.path.abspath(output_path)

    try:
        visualize_piece_placement(puzzle_path, piece_path, output_path)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
