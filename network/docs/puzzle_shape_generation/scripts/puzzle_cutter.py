#!/usr/bin/env python3
"""Puzzle Image Cutter - Cut images into jigsaw puzzle pieces.

Takes an input image and generates individual puzzle piece PNGs with
realistic Bezier curve edges and transparent backgrounds.

Usage:
    python puzzle_cutter.py input.jpg --pieces 100 --output-dir output/
    python puzzle_cutter.py input.jpg --rows 5 --cols 5 --output-dir output/
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

from edge_grid import EdgeGrid, calculate_grid_dimensions, generate_edge_grid
from image_masking import CoordinateMapper, cut_piece, generate_piece_polygon
from PIL import Image


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Cut an image into jigsaw puzzle pieces with Bezier curve edges.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python puzzle_cutter.py photo.jpg --pieces 100
  python puzzle_cutter.py photo.jpg --rows 5 --cols 8 --output-dir my_puzzle/
  python puzzle_cutter.py photo.jpg --pieces 50 --seed 42 --padding 30
""",
    )

    parser.add_argument("input_image", type=Path, help="Path to the input image file")
    parser.add_argument(
        "--pieces",
        type=int,
        default=100,
        help="Target number of pieces (default: 100). Grid dimensions are auto-calculated.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        help="Explicit number of rows (overrides --pieces when used with --cols)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Explicit number of columns (overrides --pieces when used with --rows)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for piece images (default: output/)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around each piece in pixels for tab protrusions (default: 20)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible edge generation")
    parser.add_argument(
        "--points-per-curve",
        type=int,
        default=20,
        help="Number of points to sample per Bezier curve (default: 20)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="piece",
        help="Filename prefix for output pieces (default: piece)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    return parser


def load_image(image_path: Path) -> Image.Image:
    """Load and validate an image file."""
    if not image_path.exists():
        print(f"Error: Input image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        image = Image.open(image_path)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)


def determine_grid_dimensions(args: argparse.Namespace, width: int, height: int) -> Tuple[int, int]:
    """Determine grid dimensions from arguments or calculate automatically."""
    if args.rows and args.cols:
        return args.rows, args.cols
    elif args.rows or args.cols:
        print("Error: Both --rows and --cols must be specified together", file=sys.stderr)
        sys.exit(1)
    else:
        return calculate_grid_dimensions(width, height, args.pieces)


def cut_and_save_pieces(
    image: Image.Image,
    edge_grid: EdgeGrid,
    mapper: CoordinateMapper,
    args: argparse.Namespace,
) -> None:
    """Cut all pieces from the image and save them."""
    rows, cols = edge_grid.rows, edge_grid.cols
    total_pieces = rows * cols

    if not args.quiet:
        print(f"Cutting {total_pieces} pieces...")

    for r in range(rows):
        for c in range(cols):
            polygon = generate_piece_polygon(
                edge_grid,
                mapper,
                r,
                c,
                points_per_curve=args.points_per_curve,
            )
            piece_img, _offset = cut_piece(image, polygon, padding=args.padding)
            output_path = args.output_dir / f"{args.prefix}_r{r:02d}_c{c:02d}.png"
            piece_img.save(output_path, "PNG")

            if not args.quiet:
                piece_num = r * cols + c + 1
                print(f"  [{piece_num}/{total_pieces}] {output_path.name}", end="\r")

    if not args.quiet:
        print(f"\nSaved {total_pieces} pieces to {args.output_dir}/")


def main() -> None:
    """Main entry point for the puzzle cutter CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()

    image = load_image(args.input_image)
    width, height = image.size

    rows, cols = determine_grid_dimensions(args, width, height)

    if not args.quiet:
        print(f"Input image: {args.input_image} ({width}x{height})")
        print(f"Grid: {rows} rows x {cols} cols = {rows * cols} pieces")
        print(f"Piece size: ~{width / cols:.0f}x{height / rows:.0f} pixels")
        if args.seed is not None:
            print(f"Random seed: {args.seed}")
        print("Generating edge grid...")

    edge_grid = generate_edge_grid(rows, cols, seed=args.seed)
    mapper = CoordinateMapper(image_width=width, image_height=height, rows=rows, cols=cols)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cut_and_save_pieces(image, edge_grid, mapper, args)


if __name__ == "__main__":
    main()
