#!/usr/bin/env python3
"""Regenerate the exp20 test-split pieces for re-evaluation.

The original `realistic_4x4_20k` dataset (generated December 2025, deleted
after the RunPod run) is reproducible: `generate_dataset.py` seeds each
puzzle's edge grid with `seed + source_index`, and `generate_edge_grid`
reseeds the global RNG, so every piece (shape AND baked-in rotation) is a
pure function of the per-puzzle seed.

This script reproduces the original train/test split (seed-42 shuffle over
all generated puzzle IDs, as in `create_datasets`) and regenerates pieces
for the test puzzles only.

IMPORTANT: `puzzle_shapes` received edge-geometry fixes in January 2026
(commits 532224c, d23f988) AFTER the exp20 dataset was generated. To
reproduce the shapes the checkpoint was trained on, pass the historical
package via --puzzle-shapes-path (extracted from commit 6b61eb7).

Usage:
    python regenerate_test_split.py \
        --source-dir /path/to/datasets/puzzles \
        --output-dir /path/to/datasets/realistic_4x4_20k_test \
        --puzzle-shapes-path /path/to/hist_lib
"""

import argparse
import random
import sys
import time
from pathlib import Path

# Original run configuration (setup_and_train.sh / train_cuda.py args)
N_TRAIN_PUZZLES_REQUESTED = 10800
N_TEST_PUZZLES_REQUESTED = 1200
GENERATION_SEED = 42
SPLIT_SEED = 42


def reproduce_split(source_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Reproduce the original test split and per-puzzle generation seeds.

    Mirrors generate_dataset.generate_dataset (sorted glob, index-based
    seeds) and dataset.create_datasets (seed-42 shuffle, min() clamping).

    Args:
        source_dir: Directory with the source puzzle JPEGs.

    Returns:
        Tuple of (test puzzle IDs sorted, mapping of puzzle ID to its
        generation seed).
    """
    puzzle_files = sorted(source_dir.glob("puzzle_*.jpg"))
    # generate_dataset was invoked requesting >= all available puzzles,
    # so every source image was processed, in sorted order.
    generation_seeds = {p.stem: GENERATION_SEED + i for i, p in enumerate(puzzle_files)}

    # create_datasets: get_puzzle_ids returns sorted dir names == sorted stems
    all_ids = sorted(generation_seeds)
    rng = random.Random(SPLIT_SEED)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    n_train = min(N_TRAIN_PUZZLES_REQUESTED, len(shuffled) - N_TEST_PUZZLES_REQUESTED)
    n_test = min(N_TEST_PUZZLES_REQUESTED, len(shuffled) - n_train)
    test_ids = sorted(shuffled[n_train : n_train + n_test])

    print(f"Source puzzles: {len(puzzle_files)}")
    print(f"Split: {n_train} train / {n_test} test puzzles")
    return test_ids, generation_seeds


def main() -> None:
    """Regenerate test-split pieces with the original per-puzzle seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--puzzle-shapes-path",
        type=Path,
        default=None,
        help="Directory containing the (historical) puzzle_shapes package to import",
    )
    args = parser.parse_args()

    if args.puzzle_shapes_path is not None:
        sys.path.insert(0, str(args.puzzle_shapes_path))
    # Import after path setup so generate_dataset picks up the right library
    sys.path.insert(0, str(Path(__file__).parent))
    import puzzle_shapes

    print(f"puzzle_shapes loaded from: {puzzle_shapes.__file__}")
    from generate_dataset import generate_pieces_for_puzzle

    test_ids, generation_seeds = reproduce_split(args.source_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n")

    start = time.time()
    for n, puzzle_id in enumerate(test_ids, 1):
        puzzle_dir = args.output_dir / puzzle_id
        if puzzle_dir.exists() and len(list(puzzle_dir.glob("*.png"))) == 16:
            continue  # already generated
        generate_pieces_for_puzzle(
            args.source_dir / f"{puzzle_id}.jpg",
            args.output_dir,
            seed=generation_seeds[puzzle_id],
            padding=20,
            points_per_curve=20,
        )
        if n % 50 == 0 or n == len(test_ids):
            elapsed = time.time() - start
            print(f"  [{n}/{len(test_ids)}] {elapsed:.0f}s elapsed", flush=True)

    print(f"\nDone: {len(test_ids)} test puzzles in {args.output_dir}")


if __name__ == "__main__":
    main()
