#!/usr/bin/env python3
r"""Generate RGBA realistic puzzle pieces for exp30 (lossless base rotation).

Byte-for-byte the exp26 generator — same seed scheme, same Bezier cut, same
per-puzzle ``random.Random(seed)`` draw of the base rotation, same
``puzzle_id_x{cx}_y{cy}_rot{rot}.png`` filename convention, same RGBA
output, same ``--workers`` / ``--skip-existing`` CLI — with ONE difference:
the discrete base rotation is applied with :func:`framing.rotate_lossless`
(``Image.transpose``) instead of ``rotate(expand=False, BILINEAR)``.

That single change kills both generator bugs from
``docs/synthetic-dataset-realism.html`` section 4:

- **4.1 border-touch shortcut**: ``expand=False`` on the tight, non-square
  crop clipped ~60 px of tabs at 90deg/270deg and left black bands, making
  "alpha touches all four borders" a 100%-precision test for
  rotation in {0, 180}. Transpose swaps the canvas dimensions instead, so
  every rotation keeps the full silhouette and the same border profile.
- **4.2 half-pixel blur cue**: bilinear resampling on a non-square canvas
  made 90deg/270deg ~32% blurrier than the pixel-exact 0deg/180deg.
  Transpose is an exact permutation, so all four rotations are equally
  sharp.

Pieces stay cropped tight on disk; the 8%-margin square framing that the
real path uses is applied at **load** time (see ``framed_dataset.py``), so
the same PNGs serve both the augmented train path and the deterministic
eval path.

Usage (run from the network/ directory):
    uv run python -m experiments.exp30_generator_fixes.generate_dataset \\
        --n-puzzles 20 --output-dir datasets/realistic_4x4_rgba_v2
"""

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from puzzle_shapes import CoordinateMapper, cut_piece, generate_edge_grid, generate_piece_polygon

from .framing import rotate_lossless

DEFAULT_SOURCE_DIR = Path(__file__).parent.parent.parent / "datasets" / "puzzles"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4_rgba_v2"

GRID_SIZE = 4
NUM_CELLS = GRID_SIZE * GRID_SIZE
ROTATION_ANGLES = [0, 90, 180, 270]


def get_cell_center(row: int, col: int, grid_size: int = GRID_SIZE) -> tuple[float, float]:
    """Return the normalized (cx, cy) center of a grid cell.

    Args:
        row: Row index (0-indexed from top).
        col: Column index (0-indexed from left).
        grid_size: Size of the grid.

    Returns:
        Tuple of (cx, cy) in [0, 1].
    """
    cx = (col + 0.5) / grid_size
    cy = (row + 0.5) / grid_size
    return cx, cy


def generate_pieces_for_puzzle(
    puzzle_path: Path,
    output_dir: Path,
    seed: int | None = None,
    padding: int = 20,
    points_per_curve: int = 20,
) -> int:
    """Generate all RGBA pieces for one puzzle.

    Args:
        puzzle_path: Path to the source puzzle image.
        output_dir: Root output directory (a per-puzzle subdir is created).
        seed: Random seed for edge generation and base rotations (None for
            random). A per-puzzle ``random.Random(seed)`` is used instead of
            the process-global RNG so the dataset is deterministic even when
            puzzles are generated in parallel worker processes.
        padding: Padding around pieces for tab protrusions.
        points_per_curve: Points sampled per Bezier curve.

    Returns:
        The number of pieces written.
    """
    puzzle_id = puzzle_path.stem
    rng = random.Random(seed)

    with Image.open(puzzle_path) as opened:
        puzzle_img = opened.convert("RGB") if opened.mode not in ("RGB", "RGBA") else opened.copy()
    width, height = puzzle_img.size

    edge_grid = generate_edge_grid(GRID_SIZE, GRID_SIZE, seed=seed)
    mapper = CoordinateMapper(image_width=width, image_height=height, rows=GRID_SIZE, cols=GRID_SIZE)

    puzzle_output_dir = output_dir / puzzle_id
    puzzle_output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cx, cy = get_cell_center(row, col)
            polygon = generate_piece_polygon(edge_grid, mapper, row, col, points_per_curve=points_per_curve)
            piece_rgba, _offset = cut_piece(puzzle_img, polygon, padding=padding)
            if piece_rgba.mode != "RGBA":
                piece_rgba = piece_rgba.convert("RGBA")

            rotation = rng.choice(ROTATION_ANGLES)
            piece_rotated = rotate_lossless(piece_rgba, rotation)

            filename = f"{puzzle_id}_x{cx:.3f}_y{cy:.3f}_rot{rotation}.png"
            piece_rotated.save(puzzle_output_dir / filename, "PNG")
            count += 1

    return count


def _generate_one(job: tuple[Path, Path, int, int, int]) -> int:
    """Worker for the process pool: generate one puzzle's pieces.

    Args:
        job: Tuple of (puzzle_path, output_dir, seed, padding, points_per_curve).

    Returns:
        The number of pieces written for this puzzle.
    """
    puzzle_path, output_dir, seed, padding, points_per_curve = job
    return generate_pieces_for_puzzle(
        puzzle_path,
        output_dir,
        seed=seed,
        padding=padding,
        points_per_curve=points_per_curve,
    )


def _is_complete(output_dir: Path, puzzle_id: str) -> bool:
    """Return True if a puzzle already has all ``NUM_CELLS`` pieces on disk.

    Args:
        output_dir: Root output directory.
        puzzle_id: Puzzle ID to check.

    Returns:
        True when the puzzle's directory already holds every piece.
    """
    puzzle_dir = output_dir / puzzle_id
    return puzzle_dir.is_dir() and len(list(puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"))) >= NUM_CELLS


def generate_dataset(
    source_dir: Path,
    output_dir: Path,
    n_puzzles: int = 500,
    seed: int = 42,
    padding: int = 20,
    points_per_curve: int = 20,
    workers: int = 1,
    skip_existing: bool = False,
) -> None:
    """Generate the RGBA piece dataset.

    Args:
        source_dir: Directory of source ``puzzle_*.jpg`` images.
        output_dir: Directory to write generated pieces.
        n_puzzles: Number of puzzles to process (from the sorted list).
        seed: Base random seed (per-puzzle seed = seed + index), matching
            exp20/exp26 so piece geometry is reproducible and identical.
        padding: Padding around pieces.
        points_per_curve: Points per Bezier curve.
        workers: Parallel worker processes (per-puzzle generation is
            embarrassingly parallel; use ~CPU count on RunPod).
        skip_existing: Skip puzzles that already have all pieces on disk
            (lets an interrupted RunPod generation resume).
    """
    puzzle_files = sorted(source_dir.glob("puzzle_*.jpg"))
    if not puzzle_files:
        print(f"Error: No puzzle images found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    if n_puzzles < len(puzzle_files):
        puzzle_files = puzzle_files[:n_puzzles]
    else:
        print(f"Warning: Only {len(puzzle_files)} puzzles available (requested {n_puzzles})")

    print(f"Generating RGBA pieces for {len(puzzle_files)} puzzles -> {output_dir} (workers={workers})")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-puzzle seed = seed + index, matching exp20/exp26 exactly (the index
    # is over the full sorted list, independent of skip/parallelism).
    jobs = [
        (puzzle_path, output_dir, seed + i, padding, points_per_curve)
        for i, puzzle_path in enumerate(puzzle_files)
        if not (skip_existing and _is_complete(output_dir, puzzle_path.stem))
    ]
    skipped = len(puzzle_files) - len(jobs)
    if skipped:
        print(f"Skipping {skipped} already-complete puzzles")

    total = len(jobs) * NUM_CELLS
    done = 0
    if workers <= 1:
        for job in jobs:
            done += _generate_one(job)
            print(f"  [{done}/{total}] {job[0].name}", end="\r")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_generate_one, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), start=1):
                done += future.result()
                if i % 50 == 0 or i == len(futures):
                    print(f"  [{i}/{len(jobs)} puzzles, {done}/{total} pieces]", end="\r")

    print(f"\nGenerated {done} RGBA pieces from {len(jobs)} puzzles")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate RGBA realistic pieces for exp30 (lossless rotation)")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Source puzzle images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--n-puzzles", type=int, default=500, help="Number of puzzles to process")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (matches exp20/exp26)")
    parser.add_argument("--padding", type=int, default=20, help="Padding around pieces in pixels")
    parser.add_argument("--points-per-curve", type=int, default=20, help="Points per Bezier curve")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes (use ~CPU count on RunPod)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip puzzles already fully generated (resume an interrupted run)",
    )
    args = parser.parse_args()

    generate_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        n_puzzles=args.n_puzzles,
        seed=args.seed,
        padding=args.padding,
        points_per_curve=args.points_per_curve,
        workers=args.workers,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
