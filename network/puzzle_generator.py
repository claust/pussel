#!/usr/bin/env python
"""Puzzle Piece Generator.

This utility takes a complete puzzle image and generates a specified number
of puzzle pieces. Each piece is named according to its position and rotation in
the original image.
"""

import argparse
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# Type aliases to shorten long return type annotations
BBox = Tuple[int, int, int, int]
PieceData = Tuple[Image.Image, BBox, int]


def generate_mask(width: int, height: int, piece_size: int) -> np.ndarray:
    """Generate a mask for cutting an image into jigsaw puzzle pieces.

    Args:
        width: Width of the image
        height: Height of the image
        piece_size: Average size of a puzzle piece

    Returns:
        A numpy array of labels where each unique value represents a piece
    """
    # Calculate number of pieces across and down
    pieces_x = max(2, width // piece_size)
    pieces_y = max(2, height // piece_size)

    # Create a grid of initial piece IDs
    piece_grid = np.zeros((height, width), dtype=np.int32)
    for y in range(pieces_y):
        for x in range(pieces_x):
            # Create boundaries for each piece
            y_start = int(y * height / pieces_y)
            y_end = int((y + 1) * height / pieces_y)
            x_start = int(x * width / pieces_x)
            x_end = int((x + 1) * width / pieces_x)

            # Assign unique ID to each piece
            piece_id = y * pieces_x + x + 1
            piece_grid[y_start:y_end, x_start:x_end] = piece_id

    # Return regular grid without distortion
    return piece_grid


def _extract_single_piece(
    image: Image.Image, mask: np.ndarray, piece_id: int
) -> Optional[PieceData]:
    """Extract a single puzzle piece from an image.

    Args:
        image: PIL Image to extract piece from
        mask: Numpy array mask
        piece_id: ID of the piece to extract

    Returns:
        Tuple of (piece_image, bounding_box, rotation) or None if no piece
        found
    """
    # Create a binary mask for this piece
    piece_mask = (mask == piece_id).astype(np.uint8) * 255
    piece_mask_img = Image.fromarray(piece_mask)

    # Find bounding box
    bbox = piece_mask_img.getbbox()
    if bbox is None:
        return None

    # Extract piece
    x1, y1, x2, y2 = bbox
    piece_img = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))

    def _get_pixel_color(x: int, y: int) -> Tuple[int, int, int, int]:
        """Return the appropriate pixel color based on mask value."""
        if mask[y, x] == piece_id:
            r, g, b = image.getpixel((x, y))[:3]
            return (r, g, b, 255)
        return (0, 0, 0, 0)

    # Extract pixel data
    for y in range(y1, y2):
        for x in range(x1, x2):
            piece_img.putpixel((x - x1, y - y1), _get_pixel_color(x, y))

    # Random rotation for training diversity
    rotation = random.choice([0, 90, 180, 270])
    if rotation:
        piece_img = piece_img.rotate(rotation, expand=True)

    return piece_img, bbox, rotation


def extract_pieces(image: Image.Image, mask: np.ndarray) -> List[PieceData]:
    """Extract puzzle pieces from an image using a mask.

    Args:
        image: PIL Image to extract pieces from
        mask: Numpy array where each unique value represents a puzzle piece

    Returns:
        List of (piece_image, bounding_box, rotation) tuples
    """
    pieces = []
    unique_ids = np.unique(mask)

    # Skip the background value (0)
    for piece_id in unique_ids[1:]:
        piece_data = _extract_single_piece(image, mask, piece_id)
        if piece_data:
            pieces.append(piece_data)

    return pieces


def _save_puzzle_pieces(
    pieces: List[PieceData],
    output_dir: str,
    puzzle_name: str,
) -> int:
    """Save extracted puzzle pieces to disk.

    Args:
        pieces: List of (piece_image, bounding_box, rotation) tuples
        output_dir: Base directory to save pieces
        puzzle_name: Name of the puzzle for folder creation

    Returns:
        Number of pieces saved
    """
    # Create output directory
    piece_dir = os.path.join(output_dir, puzzle_name)
    os.makedirs(piece_dir, exist_ok=True)

    # Save pieces
    for i, (piece_img, bbox, rotation) in enumerate(pieces, 1):
        x1, y1, x2, y2 = bbox
        filename = f"piece_{i:03d}_x{x1}_y{y1}_x{x2}_y{y2}_r{rotation}.png"
        piece_path = os.path.join(piece_dir, filename)
        piece_img.save(piece_path)

    return len(pieces)


def generate_puzzle_pieces(
    input_image_path: str, output_dir: str, num_pieces: int = 500
):
    """Generate puzzle pieces from a complete puzzle image.

    Args:
        input_image_path: Path to the input puzzle image
        output_dir: Directory to save the generated pieces
        num_pieces: Approximate number of pieces to generate
    """
    # Load the image
    image = Image.open(input_image_path).convert("RGB")
    width, height = image.size

    # Calculate piece size based on desired piece count
    piece_size = int(np.sqrt((width * height) / num_pieces))

    # Generate mask
    mask = generate_mask(width, height, piece_size)
    actual_pieces = len(np.unique(mask)) - 1  # Subtract 1 for background

    print(
        f"Generating approximately {actual_pieces} pieces from " f"{input_image_path}"
    )

    # Extract pieces
    pieces = extract_pieces(image, mask)

    # Save pieces
    puzzle_name = os.path.splitext(os.path.basename(input_image_path))[0]
    saved_count = _save_puzzle_pieces(pieces, output_dir, puzzle_name)

    print(
        f"Generated {saved_count} pieces in " f"{os.path.join(output_dir, puzzle_name)}"
    )


def main():
    """Process command-line arguments and run the puzzle piece generator."""
    parser = argparse.ArgumentParser(
        description="Generate puzzle pieces from complete puzzle images"
    )
    parser.add_argument(
        "image_path", help="Path to the puzzle image or directory of images"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/example/pieces",
        help="Directory to save the generated pieces",
    )
    parser.add_argument(
        "--pieces",
        type=int,
        default=500,
        help="Approximate number of pieces to generate",
    )

    args = parser.parse_args()

    # Create full output path
    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Process single image or directory
    image_path = args.image_path
    if os.path.isdir(image_path):
        for filename in os.listdir(image_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_path, filename)
                generate_puzzle_pieces(img_path, output_dir, args.pieces)
    else:
        generate_puzzle_pieces(image_path, output_dir, args.pieces)


if __name__ == "__main__":
    main()
