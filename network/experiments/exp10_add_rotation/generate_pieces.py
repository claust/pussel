"""Generate 2x2 quadrant pieces with rotations from puzzle images.

This script generates training data by cutting puzzle images into 4 quadrants
and optionally rotating them. Useful for visualization and debugging.

Output structure:
    outputs/generated_pieces/
    +-- puzzle_001/
    |   +-- piece_q0_r0.jpg  (top-left, 0 degrees)
    |   +-- piece_q0_r90.jpg  (top-left, 90 degrees)
    |   +-- ...
    +-- ...
"""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .dataset import QUADRANT_CENTERS, ROTATION_ANGLES, get_puzzle_ids

DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "generated_pieces"


def extract_quadrant(puzzle_img: Image.Image, quadrant_idx: int) -> Image.Image:
    """Extract a quadrant from the puzzle image.

    Args:
        puzzle_img: Full puzzle PIL Image.
        quadrant_idx: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right

    Returns:
        Cropped quadrant as PIL Image.
    """
    w, h = puzzle_img.size
    half_w, half_h = w // 2, h // 2

    if quadrant_idx == 0:  # top-left
        box = (0, 0, half_w, half_h)
    elif quadrant_idx == 1:  # top-right
        box = (half_w, 0, w, half_h)
    elif quadrant_idx == 2:  # bottom-left
        box = (0, half_h, half_w, h)
    else:  # bottom-right (3)
        box = (half_w, half_h, w, h)

    return puzzle_img.crop(box)


def generate_pieces_for_puzzle(
    puzzle_id: str,
    dataset_root: Path,
    output_dir: Path,
    piece_size: int | None = None,
    include_rotations: bool = True,
) -> None:
    """Generate 2x2 pieces with rotations for a single puzzle.

    Args:
        puzzle_id: ID of the puzzle (e.g., "puzzle_001").
        dataset_root: Root directory containing puzzles/.
        output_dir: Output directory for generated pieces.
        piece_size: If provided, resize pieces to this size.
        include_rotations: If True, generate all 4 rotations per quadrant.
    """
    puzzle_path = dataset_root / "puzzles" / f"{puzzle_id}.jpg"
    puzzle_img = Image.open(puzzle_path).convert("RGB")

    # Create output directory
    puzzle_output_dir = output_dir / puzzle_id
    puzzle_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pieces for each quadrant
    quadrant_names = ["q0_top_left", "q1_top_right", "q2_bottom_left", "q3_bottom_right"]

    for i in range(4):
        piece = extract_quadrant(puzzle_img, i)

        if piece_size is not None:
            piece = piece.resize((piece_size, piece_size), Image.Resampling.LANCZOS)

        cx, cy = QUADRANT_CENTERS[i]

        if include_rotations:
            for _rot_idx, angle in enumerate(ROTATION_ANGLES):
                if angle == 0:
                    rotated = piece
                else:
                    rotated = piece.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

                piece_path = puzzle_output_dir / f"piece_{quadrant_names[i]}_r{angle}_cx{cx:.2f}_cy{cy:.2f}.jpg"
                rotated.save(piece_path, quality=95)
        else:
            piece_path = puzzle_output_dir / f"piece_{quadrant_names[i]}_cx{cx:.2f}_cy{cy:.2f}.jpg"
            piece.save(piece_path, quality=95)


def generate_all_pieces(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    piece_size: int | None = 128,
    n_puzzles: int | None = None,
    include_rotations: bool = True,
) -> None:
    """Generate pieces for all puzzles.

    Args:
        dataset_root: Root directory containing puzzles/.
        output_dir: Output directory for generated pieces.
        piece_size: If provided, resize pieces to this size.
        n_puzzles: If provided, limit to first N puzzles.
        include_rotations: If True, generate all 4 rotations per quadrant.
    """
    puzzle_ids = get_puzzle_ids(dataset_root)

    if n_puzzles is not None:
        puzzle_ids = puzzle_ids[:n_puzzles]

    pieces_per_puzzle = 16 if include_rotations else 4
    print(f"Generating pieces for {len(puzzle_ids)} puzzles ({pieces_per_puzzle} per puzzle)...")
    print(f"Output directory: {output_dir}")
    if piece_size:
        print(f"Piece size: {piece_size}x{piece_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for puzzle_id in tqdm(puzzle_ids, desc="Generating pieces"):
        generate_pieces_for_puzzle(
            puzzle_id=puzzle_id,
            dataset_root=dataset_root,
            output_dir=output_dir,
            piece_size=piece_size,
            include_rotations=include_rotations,
        )

    print(f"\nGenerated {len(puzzle_ids) * pieces_per_puzzle} pieces total.")
    print(f"Output directory: {output_dir}")


def visualize_rotations(
    puzzle_id: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path | None = None,
) -> None:
    """Visualize how a piece looks at different rotations.

    Args:
        puzzle_id: ID of the puzzle to visualize.
        dataset_root: Root directory containing puzzles/.
        output_path: Path to save visualization (optional).
    """
    import matplotlib.pyplot as plt

    puzzle_path = dataset_root / "puzzles" / f"{puzzle_id}.jpg"
    puzzle_img = Image.open(puzzle_path).convert("RGB")

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    # First column: quadrant labels
    for i in range(4):
        ax = axes[i, 0]
        piece = extract_quadrant(puzzle_img, i)
        ax.imshow(piece)
        ax.set_title(f"Q{i}\n({QUADRANT_CENTERS[i][0]:.2f}, {QUADRANT_CENTERS[i][1]:.2f})")
        ax.axis("off")

    # Remaining columns: rotations
    for q_idx in range(4):
        piece = extract_quadrant(puzzle_img, q_idx)

        for r_idx, angle in enumerate(ROTATION_ANGLES):
            ax = axes[q_idx, r_idx + 1]
            if angle == 0:
                rotated = piece
            else:
                rotated = piece.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

            ax.imshow(rotated)
            ax.set_title(f"{angle}")
            ax.axis("off")

    # Add column headers
    for i, angle in enumerate(ROTATION_ANGLES):
        axes[0, i + 1].set_title(f"Rotation: {angle}")

    fig.suptitle(f"Quadrants and Rotations: {puzzle_id}", fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2x2 quadrant pieces with rotations")
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Visualize rotations for a specific puzzle ID",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate pieces for all puzzles",
    )
    parser.add_argument(
        "--n-puzzles",
        type=int,
        default=None,
        help="Limit to first N puzzles",
    )
    parser.add_argument(
        "--piece-size",
        type=int,
        default=128,
        help="Size to resize pieces to",
    )
    parser.add_argument(
        "--no-rotations",
        action="store_true",
        help="Skip rotation variants (only generate original orientation)",
    )
    args = parser.parse_args()

    if args.visualize:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_rotations(
            puzzle_id=args.visualize,
            output_path=output_dir / f"rotation_visualization_{args.visualize}.png",
        )
    elif args.generate:
        generate_all_pieces(
            piece_size=args.piece_size,
            n_puzzles=args.n_puzzles,
            include_rotations=not args.no_rotations,
        )
    else:
        # Default: visualize one example
        puzzle_ids = get_puzzle_ids()
        if puzzle_ids:
            print(f"Visualizing example puzzle: {puzzle_ids[0]}")
            visualize_rotations(puzzle_id=puzzle_ids[0])
        else:
            print("No puzzles found in dataset.")
