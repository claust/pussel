#!/usr/bin/env python3
"""Bézier Curve Puzzle Piece Shape Generator (CLI Wrapper).

This script provides a command-line interface for generating realistic puzzle
piece shapes using cubic Bézier curves.
"""

import argparse
from pathlib import Path

from comparison import generate_reference_comparison
from io_utils import load_pieces_from_json, save_pieces_to_json
from rendering import generate_pieces_from_json, render_piece_to_png


def main() -> None:
    """Main entry point for the Bézier piece generator CLI."""
    # Define default paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    default_json_path = base_dir / "reference_pieces.json"

    # Load reference pieces for presets
    try:
        reference_pieces_list = load_pieces_from_json(default_json_path)
        reference_pieces = {f"ref{i + 1}": p for i, p in enumerate(reference_pieces_list)}
    except FileNotFoundError:
        reference_pieces = {}

    parser = argparse.ArgumentParser(
        description="Generate realistic puzzle piece shapes using Bézier curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bezier_piece_generator.py                    # Generate random piece
  python bezier_piece_generator.py -o my_piece.png    # Custom output path
  python bezier_piece_generator.py --size 256         # Smaller image
  python bezier_piece_generator.py --preset ref1      # Use reference preset
  python bezier_piece_generator.py --compare          # Compare with references
  python bezier_piece_generator.py --json pieces.json # Generate from JSON config
  python bezier_piece_generator.py --export-json out.json  # Export presets to JSON
""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="piece.png",
        help="Output path for the generated piece (default: piece.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated pieces when using --json",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of the output image in pixels (default: 512)",
    )
    parser.add_argument(
        "--no-transparent",
        action="store_true",
        help="Use white background instead of transparent",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(reference_pieces.keys()),
        help="Use a preset configuration (ref1-ref6)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison with reference pieces",
    )
    parser.add_argument(
        "--json",
        type=str,
        metavar="FILE",
        help="Generate pieces from a JSON configuration file",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        metavar="FILE",
        help="Export the reference presets to a JSON file for editing",
    )

    args = parser.parse_args()

    if args.export_json:
        pieces = list(reference_pieces.values())
        save_pieces_to_json(pieces, args.export_json)
        print(f"Exported {len(pieces)} piece configurations to: {args.export_json}")
    elif args.json:
        generate_pieces_from_json(
            args.json,
            output_dir=args.output_dir,
            size_px=args.size,
            transparent_bg=not args.no_transparent,
        )
    elif args.compare:
        output_path = generate_reference_comparison()
        print(f"Generated reference comparison: {output_path}")
    else:
        config = reference_pieces.get(args.preset) if args.preset else None
        output_path = render_piece_to_png(
            config=config,
            output_path=args.output,
            size_px=args.size,
            transparent_bg=not args.no_transparent,
        )
        print(f"Generated puzzle piece: {output_path}")


if __name__ == "__main__":
    main()
