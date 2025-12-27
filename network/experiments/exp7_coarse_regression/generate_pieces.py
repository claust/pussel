"""Generate 2x2 quadrant pieces from puzzle images.

This script generates training data by cutting puzzle images into 4 quadrants.
While the dataset.py module generates pieces on-the-fly, this script is useful
for visualization and debugging.

Output structure:
    outputs/generated_pieces/
    ├── puzzle_001/
    │   ├── piece_q0.jpg  (top-left)
    │   ├── piece_q1.jpg  (top-right)
    │   ├── piece_q2.jpg  (bottom-left)
    │   └── piece_q3.jpg  (bottom-right)
    └── ...
"""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .dataset import QUADRANT_CENTERS, get_puzzle_ids

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
) -> None:
    """Generate 2x2 pieces for a single puzzle.

    Args:
        puzzle_id: ID of the puzzle (e.g., "puzzle_001").
        dataset_root: Root directory containing puzzles/.
        output_dir: Output directory for generated pieces.
        piece_size: If provided, resize pieces to this size.
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
        piece_path = puzzle_output_dir / f"piece_{quadrant_names[i]}_cx{cx:.2f}_cy{cy:.2f}.jpg"
        piece.save(piece_path, quality=95)


def generate_all_pieces(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    piece_size: int | None = 128,
    n_puzzles: int | None = None,
) -> None:
    """Generate pieces for all puzzles.

    Args:
        dataset_root: Root directory containing puzzles/.
        output_dir: Output directory for generated pieces.
        piece_size: If provided, resize pieces to this size.
        n_puzzles: If provided, limit to first N puzzles.
    """
    puzzle_ids = get_puzzle_ids(dataset_root)

    if n_puzzles is not None:
        puzzle_ids = puzzle_ids[:n_puzzles]

    print(f"Generating 2x2 pieces for {len(puzzle_ids)} puzzles...")
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
        )

    print(f"\nGenerated {len(puzzle_ids) * 4} pieces total.")
    print(f"Output directory: {output_dir}")


def visualize_quadrants(
    puzzle_id: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path | None = None,
) -> None:
    """Visualize how a puzzle is split into quadrants.

    Args:
        puzzle_id: ID of the puzzle to visualize.
        dataset_root: Root directory containing puzzles/.
        output_path: Path to save visualization (optional).
    """
    import matplotlib.pyplot as plt

    puzzle_path = dataset_root / "puzzles" / f"{puzzle_id}.jpg"
    puzzle_img = Image.open(puzzle_path).convert("RGB")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # Show full puzzle with quadrant overlay
    axes[0].imshow(puzzle_img)
    w, h = puzzle_img.size

    # Draw quadrant boundaries
    axes[0].axhline(y=h / 2, color="red", linewidth=2)
    axes[0].axvline(x=w / 2, color="red", linewidth=2)

    # Label quadrants
    labels = ["Q0\n(0.25, 0.25)", "Q1\n(0.75, 0.25)", "Q2\n(0.25, 0.75)", "Q3\n(0.75, 0.75)"]
    positions = [(w / 4, h / 4), (3 * w / 4, h / 4), (w / 4, 3 * h / 4), (3 * w / 4, 3 * h / 4)]
    for label, (x, y) in zip(labels, positions):
        axes[0].text(x, y, label, ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    axes[0].set_title(f"Full Puzzle: {puzzle_id}")
    axes[0].axis("off")

    # Show each quadrant
    quadrant_titles = ["Q0: Top-Left", "Q1: Top-Right", "Q2: Bottom-Left", "Q3: Bottom-Right"]
    for i in range(4):
        piece = extract_quadrant(puzzle_img, i)
        axes[i + 1].imshow(piece)
        cx, cy = QUADRANT_CENTERS[i]
        axes[i + 1].set_title(f"{quadrant_titles[i]}\ncenter: ({cx:.2f}, {cy:.2f})")
        axes[i + 1].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2x2 quadrant pieces from puzzles")
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Visualize quadrants for a specific puzzle ID",
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
    args = parser.parse_args()

    if args.visualize:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_quadrants(
            puzzle_id=args.visualize,
            output_path=output_dir / f"quadrant_visualization_{args.visualize}.png",
        )
    elif args.generate:
        generate_all_pieces(
            piece_size=args.piece_size,
            n_puzzles=args.n_puzzles,
        )
    else:
        # Default: visualize one example
        puzzle_ids = get_puzzle_ids()
        if puzzle_ids:
            print(f"Visualizing example puzzle: {puzzle_ids[0]}")
            visualize_quadrants(puzzle_id=puzzle_ids[0])
        else:
            print("No puzzles found in dataset.")
