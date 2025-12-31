#!/usr/bin/env python3
"""Generate realistic puzzle piece dataset for exp20.

Creates a dataset of puzzle pieces with Bezier curve edges (tabs/blanks)
from source puzzle images. Pieces are saved with their center coordinates
encoded in the filename.

Usage:
    python generate_dataset.py --n-puzzles 500 --output-dir datasets/realistic_4x4
    python generate_dataset.py --n-puzzles 100 --seed 42 --output-dir test_dataset
"""

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image
from puzzle_shapes import CoordinateMapper, cut_piece, generate_edge_grid, generate_piece_polygon

# Default paths
DEFAULT_SOURCE_DIR = Path(__file__).parent.parent.parent / "datasets" / "puzzles"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4"

# Grid configuration
GRID_SIZE = 4  # 4x4 grid
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 16 cells

# Rotation options
ROTATION_ANGLES = [0, 90, 180, 270]


def get_cell_center(row: int, col: int, grid_size: int = GRID_SIZE) -> tuple[float, float]:
    """Get normalized center coordinates for a cell.

    Args:
        row: Row index (0-indexed from top).
        col: Column index (0-indexed from left).
        grid_size: Size of the grid.

    Returns:
        Tuple of (cx, cy) normalized coordinates in [0, 1].
    """
    cx = (col + 0.5) / grid_size
    cy = (row + 0.5) / grid_size
    return cx, cy


def fill_transparent_with_black(image: Image.Image) -> Image.Image:
    """Fill transparent pixels with black, convert to RGB.

    Args:
        image: RGBA image with transparent background.

    Returns:
        RGB image with black background.
    """
    if image.mode != "RGBA":
        return image.convert("RGB")

    # Create black background
    background = Image.new("RGB", image.size, (0, 0, 0))
    # Paste the image using alpha channel as mask
    background.paste(image, mask=image.split()[3])
    return background


def rotate_piece(image: Image.Image, rotation: int) -> Image.Image:
    """Rotate piece image by specified degrees.

    Args:
        image: Piece image.
        rotation: Rotation in degrees (0, 90, 180, 270).

    Returns:
        Rotated image.
    """
    if rotation == 0:
        return image
    # PIL rotates counter-clockwise, so negate for clockwise rotation
    return image.rotate(-rotation, expand=False, resample=Image.Resampling.BILINEAR)


def generate_pieces_for_puzzle(
    puzzle_path: Path,
    output_dir: Path,
    seed: int | None = None,
    padding: int = 20,
    points_per_curve: int = 20,
) -> list[dict]:
    """Generate all realistic pieces for a single puzzle.

    Args:
        puzzle_path: Path to the source puzzle image.
        output_dir: Directory to save piece images.
        seed: Random seed for edge generation (None for random).
        padding: Padding around pieces for tab protrusions.
        points_per_curve: Number of points to sample per Bezier curve.

    Returns:
        List of metadata dictionaries for each piece.
    """
    puzzle_id = puzzle_path.stem
    piece_records = []

    # Load puzzle image
    puzzle_img = Image.open(puzzle_path)
    if puzzle_img.mode not in ("RGB", "RGBA"):
        puzzle_img = puzzle_img.convert("RGB")
    width, height = puzzle_img.size

    # Generate edge grid with realistic interlocking edges
    edge_grid = generate_edge_grid(GRID_SIZE, GRID_SIZE, seed=seed)
    mapper = CoordinateMapper(image_width=width, image_height=height, rows=GRID_SIZE, cols=GRID_SIZE)

    # Create output directory for this puzzle
    puzzle_output_dir = output_dir / puzzle_id
    puzzle_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each piece
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Get center coordinates
            cx, cy = get_cell_center(row, col)

            # Generate piece polygon
            polygon = generate_piece_polygon(
                edge_grid,
                mapper,
                row,
                col,
                points_per_curve=points_per_curve,
            )

            # Cut piece from puzzle image
            piece_rgba, offset = cut_piece(puzzle_img, polygon, padding=padding)

            # Random rotation for training variety
            rotation = random.choice(ROTATION_ANGLES)

            # Fill transparency with black and rotate
            piece_rgb = fill_transparent_with_black(piece_rgba)
            piece_rotated = rotate_piece(piece_rgb, rotation)

            # Save piece with center coordinates in filename
            filename = f"{puzzle_id}_x{cx:.3f}_y{cy:.3f}_rot{rotation}.png"
            piece_path = puzzle_output_dir / filename
            piece_rotated.save(piece_path, "PNG")

            # Record metadata
            piece_records.append(
                {
                    "puzzle_id": puzzle_id,
                    "filename": f"{puzzle_id}/{filename}",
                    "cx": cx,
                    "cy": cy,
                    "rotation": rotation,
                    "row": row,
                    "col": col,
                    "piece_width": piece_rotated.width,
                    "piece_height": piece_rotated.height,
                }
            )

    return piece_records


def generate_dataset(
    source_dir: Path,
    output_dir: Path,
    n_puzzles: int = 500,
    seed: int = 42,
    padding: int = 20,
    points_per_curve: int = 20,
) -> None:
    """Generate realistic puzzle piece dataset.

    Args:
        source_dir: Directory containing source puzzle images.
        output_dir: Directory to save generated pieces.
        n_puzzles: Number of puzzles to process.
        seed: Random seed for reproducibility.
        padding: Padding around pieces.
        points_per_curve: Points per Bezier curve.
    """
    # Set random seed
    random.seed(seed)

    # Get all puzzle images
    puzzle_files = sorted(source_dir.glob("puzzle_*.jpg"))
    if not puzzle_files:
        print(f"Error: No puzzle images found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    # Limit to requested number
    if n_puzzles < len(puzzle_files):
        puzzle_files = puzzle_files[:n_puzzles]
    else:
        print(f"Warning: Only {len(puzzle_files)} puzzles available (requested {n_puzzles})")

    print(f"Generating realistic pieces for {len(puzzle_files)} puzzles...")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} pieces per puzzle")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {seed}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all puzzles and collect metadata
    all_records: list[dict] = []
    total_pieces = len(puzzle_files) * NUM_CELLS

    for i, puzzle_path in enumerate(puzzle_files):
        # Use deterministic seed per puzzle for reproducibility
        puzzle_seed = seed + i if seed is not None else None

        records = generate_pieces_for_puzzle(
            puzzle_path,
            output_dir,
            seed=puzzle_seed,
            padding=padding,
            points_per_curve=points_per_curve,
        )
        all_records.extend(records)

        # Progress update
        done = (i + 1) * NUM_CELLS
        print(f"  [{done}/{total_pieces}] {puzzle_path.name}", end="\r")

    print(f"\n\nGenerated {len(all_records)} pieces from {len(puzzle_files)} puzzles")

    # Write metadata CSV
    metadata_path = output_dir / "metadata.csv"
    fieldnames = ["puzzle_id", "filename", "cx", "cy", "rotation", "row", "col", "piece_width", "piece_height"]

    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"Saved metadata to {metadata_path}")

    # Summary statistics
    print("\nDataset summary:")
    print(f"  Total pieces: {len(all_records)}")
    print(f"  Puzzles: {len(puzzle_files)}")
    print(f"  Pieces per puzzle: {NUM_CELLS}")
    print(f"  Grid size: {GRID_SIZE}x{GRID_SIZE}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate realistic puzzle piece dataset for exp20",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing source puzzle images (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for generated pieces (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--n-puzzles",
        type=int,
        default=500,
        help="Number of puzzles to process (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around pieces in pixels (default: 20)",
    )
    parser.add_argument(
        "--points-per-curve",
        type=int,
        default=20,
        help="Points to sample per Bezier curve (default: 20)",
    )

    args = parser.parse_args()

    generate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        n_puzzles=args.n_puzzles,
        seed=args.seed,
        padding=args.padding,
        points_per_curve=args.points_per_curve,
    )


if __name__ == "__main__":
    main()
