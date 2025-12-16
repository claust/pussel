#!/usr/bin/env python
"""Puzzle Piece Visualizer.

This utility visualizes a puzzle piece's placement on the original puzzle.
It creates an image showing the original puzzle with the piece overlaid
and a red outline marking its correct position.

This visualizer is designed to work with processed pieces and their metadata.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw


def parse_piece_filename(filename):
    """Parse a puzzle piece filename to extract piece ID and rotation information.

    Expected format: puzzle_ID_NNN_rROTATION.png

    Args:
        filename: Puzzle piece filename

    Returns:
        tuple: (piece_id, rotation)
    """
    # Parse the new format
    pattern = r"([a-zA-Z0-9_]+)_(\d+)_r(\d+)"
    match = re.match(pattern, os.path.basename(filename))

    if not match:
        raise ValueError(f"Invalid puzzle piece filename format: {filename}")

    puzzle_id = match.group(1)
    piece_number_str = match.group(2)  # Keep as string to preserve leading zeros
    rotation = int(match.group(3))
    piece_id = f"{puzzle_id}_{piece_number_str}"

    return piece_id, rotation


def load_piece_metadata(metadata_path, piece_id):
    """Load a piece's metadata from the CSV file.

    Args:
        metadata_path: Path to the metadata CSV file
        piece_id: ID of the piece to find

    Returns:
        dict: Piece metadata
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["piece_id"] == piece_id:
                return row

    raise ValueError(f"Piece ID '{piece_id}' not found in metadata")


def find_metadata_path(piece_path):
    """Attempt to find the metadata.csv file based on the piece path.

    Args:
        piece_path: Path to the piece image

    Returns:
        str: Path to the metadata file or None if not found
    """
    # Assuming the piece is in a 'pieces' directory with metadata.csv in the parent
    # directory
    piece_dir = os.path.dirname(piece_path)

    if os.path.basename(piece_dir) == "pieces":
        parent_dir = os.path.dirname(piece_dir)
        metadata_path = os.path.join(parent_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            return metadata_path

    return None


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


def _resize_piece_to_bbox(piece, bbox):
    """Resize the piece image to match the bounding box dimensions.

    This fixes the issue with processed pieces (224x224) being too large
    compared to their actual position in the puzzle.

    Args:
        piece: PIL Image of the puzzle piece
        bbox: Bounding box tuple (x1, y1, x2, y2)

    Returns:
        Resized PIL Image
    """
    # Calculate dimensions
    x1, y1, x2, y2 = bbox
    bbox_dimensions = {"width": x2 - x1, "height": y2 - y1}

    # Add margin and calculate target size
    margin = 0.1  # 10% margin
    target_size = {
        "width": int(bbox_dimensions["width"] * (1 + margin)),
        "height": int(bbox_dimensions["height"] * (1 + margin)),
    }

    # Calculate new dimensions preserving aspect ratio
    piece_aspect = piece.width / piece.height
    target_aspect = target_size["width"] / target_size["height"]

    if piece_aspect > target_aspect:
        # Piece is wider than target
        new_size = (target_size["width"], int(target_size["width"] / piece_aspect))
    else:
        # Piece is taller than target
        new_size = (int(target_size["height"] * piece_aspect), target_size["height"])

    return piece.resize(new_size, Image.Resampling.LANCZOS)


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


def get_piece_info(piece_path, metadata_path=None):
    """Extract piece information from metadata.

    Args:
        piece_path: Path to the puzzle piece image
        metadata_path: Path to the metadata CSV file

    Returns:
        Tuple of (bbox, rotation) where bbox is (x1, y1, x2, y2)
    """
    filename = os.path.basename(piece_path)
    piece_id, rotation = parse_piece_filename(filename)

    # Try to find metadata file automatically if not provided
    if metadata_path is None:
        metadata_path = find_metadata_path(piece_path)
        if metadata_path is None:
            raise ValueError(
                "Cannot determine piece position: metadata file not found. "
                "Please provide --metadata argument."
            )

    # Load from metadata
    piece_data = load_piece_metadata(metadata_path, piece_id)
    x1 = int(piece_data["x1"])
    y1 = int(piece_data["y1"])
    x2 = int(piece_data["x2"])
    y2 = int(piece_data["y2"])

    return (x1, y1, x2, y2), rotation


def visualize_piece_placement(
    puzzle_path, piece_path, output_path=None, metadata_path=None
):
    """Create a visualization of a puzzle piece's placement.

    The visualization shows the piece on the original puzzle with a
    red outline marking its correct position.

    Args:
        puzzle_path: Path to the full puzzle image
        piece_path: Path to the puzzle piece image
        output_path: Path to save the visualization
            (default: temp_{piece_name}.png)
        metadata_path: Path to metadata CSV file

    Returns:
        Path to the output visualization
    """
    # Extract piece info from metadata and process images
    bbox, rotation = get_piece_info(piece_path, metadata_path)
    puzzle, piece = _load_and_prepare_images(puzzle_path, piece_path, rotation)

    # Create visualization
    visualization = _create_visualization(puzzle, piece, bbox)

    # Save and return result
    final_output_path = _get_output_path(piece_path, puzzle_path, output_path)
    visualization.save(final_output_path)
    print(f"Visualization saved to {final_output_path}")

    return final_output_path


def _create_visualization(puzzle, piece, bbox):
    """Create the visualization by overlaying the piece on the puzzle.

    Args:
        puzzle: PIL Image of the full puzzle
        piece: PIL Image of the puzzle piece
        bbox: Bounding box tuple (x1, y1, x2, y2)

    Returns:
        PIL Image with the visualization
    """
    # Resize the piece to match the bounding box dimensions
    piece = _resize_piece_to_bbox(piece, bbox)

    # Create visualization canvas
    visualization = puzzle.copy()
    draw = ImageDraw.Draw(visualization)

    # Draw outline and create overlay
    _draw_piece_outline(draw, bbox)
    overlay_piece = _create_semi_transparent_overlay(piece)

    # Position and paste the overlay
    paste_position = _calculate_paste_position(bbox, overlay_piece)
    visualization.paste(overlay_piece, paste_position, overlay_piece)

    return visualization


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
    parser.add_argument(
        "--metadata",
        help="Path to the metadata CSV file "
        "(will try to find automatically if not provided)",
    )

    args = parser.parse_args()

    # Use absolute paths
    puzzle_path = os.path.abspath(args.puzzle)
    piece_path = os.path.abspath(args.piece)
    output_path = args.output
    if output_path:
        output_path = os.path.abspath(output_path)

    metadata_path = args.metadata
    if metadata_path:
        metadata_path = os.path.abspath(metadata_path)

    try:
        visualize_piece_placement(puzzle_path, piece_path, output_path, metadata_path)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
