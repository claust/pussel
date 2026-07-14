"""Frozen train/val/test split for the realistic 4x4 benchmark.

This module fixes methodology issue #3 from CRITICAL_REVIEW.md ("No
validation set anywhere; the test set steered development"). It defines a
single frozen split, checked into the repo as
``splits/realistic_4x4_v1.json``, that all experiments on the realistic
4x4 benchmark must load instead of re-deriving their own split:

- **test** (1,200 puzzles): byte-identical to the original exp20 test
  split (seed-42 shuffle of all source puzzles, indices 10798+), so every
  number stays comparable to the exp20 re-evaluation and the exp23
  classical baselines. Test is evaluated ONCE per experiment, on the
  final (val-selected) checkpoint only.
- **val** (600 puzzles): the last 600 puzzles of the original *train*
  portion (shuffle indices 10198-10797). Used every epoch for checkpoint
  selection and any hyperparameter decisions. Note: the December 2025
  exp20 checkpoint saw these puzzles during training, so val metrics are
  only meaningful for models trained on this split.
- **train** (remaining puzzles): shuffle indices 0-10197.
- **train_eval** (600 puzzles): a frozen random subset of train (the
  first 600 shuffle indices). Per-epoch train metrics are measured on
  this subset in eval mode with the same deterministic all-4-rotations
  protocol as val/test, so the train/val gap is apples-to-apples.

The split is a pure function of the sorted source-puzzle list and the
original seed-42 shuffle; ``freeze_split`` regenerates it and the CLI
verifies the test portion against the regenerated test set's
``test_ids.txt`` before writing.
"""

import argparse
import json
import random
from pathlib import Path

# Original exp20 run configuration (see regenerate_test_split.py); the
# frozen split MUST keep these so the test portion matches the original.
N_TRAIN_PUZZLES_REQUESTED = 10800
N_TEST_PUZZLES_REQUESTED = 1200
SPLIT_SEED = 42

# New (July 2026) validation carve-out from the original train portion.
N_VAL_PUZZLES = 600
N_TRAIN_EVAL_PUZZLES = 600

SPLIT_VERSION = 1
DEFAULT_SPLIT_PATH = Path(__file__).parent / "splits" / "realistic_4x4_v1.json"

SPLIT_KEYS = ("train", "train_eval", "val", "test")


def freeze_split(source_dir: Path) -> dict[str, object]:
    """Derive the frozen split from the source puzzle images.

    Reproduces the original seed-42 shuffle over all source puzzle IDs
    (as in ``dataset.create_datasets``), keeps the original test portion
    unchanged, and carves val/train_eval out of the original train
    portion deterministically.

    Args:
        source_dir: Directory containing the source ``puzzle_*.jpg`` images.

    Returns:
        Split dictionary ready to be serialized to JSON.
    """
    all_ids = sorted(p.stem for p in source_dir.glob("puzzle_*.jpg"))
    if not all_ids:
        raise ValueError(f"No puzzle_*.jpg files found in {source_dir}")

    rng = random.Random(SPLIT_SEED)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    # Original exp20 partition (min() clamping as in create_datasets).
    n_train_orig = min(N_TRAIN_PUZZLES_REQUESTED, len(shuffled) - N_TEST_PUZZLES_REQUESTED)
    n_test = min(N_TEST_PUZZLES_REQUESTED, len(shuffled) - n_train_orig)
    test_ids = shuffled[n_train_orig : n_train_orig + n_test]

    # Carve val from the END of the original train portion; train_eval is
    # the START of the remaining train portion (shuffle order is random,
    # so both are random samples).
    if n_train_orig <= N_VAL_PUZZLES + N_TRAIN_EVAL_PUZZLES:
        raise ValueError(f"Train portion too small ({n_train_orig}) to carve val/train_eval")
    val_ids = shuffled[n_train_orig - N_VAL_PUZZLES : n_train_orig]
    train_ids = shuffled[: n_train_orig - N_VAL_PUZZLES]
    train_eval_ids = train_ids[:N_TRAIN_EVAL_PUZZLES]

    return {
        "version": SPLIT_VERSION,
        "dataset": "realistic_4x4",
        "split_seed": SPLIT_SEED,
        "n_source_puzzles": len(all_ids),
        "n_train_requested_original": N_TRAIN_PUZZLES_REQUESTED,
        "n_test_requested_original": N_TEST_PUZZLES_REQUESTED,
        "notes": (
            "Frozen split for the realistic 4x4 benchmark. 'test' is identical to the "
            "original exp20 seed-42 test split (comparable to the exp20 re-evaluation and "
            "exp23 baselines). 'val' (checkpoint selection) and 'train_eval' (eval-mode "
            "train metrics) are carved from the original train portion. Do not regenerate "
            "with different parameters; bump the version instead."
        ),
        "train": sorted(train_ids),
        "train_eval": sorted(train_eval_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def load_split(split_path: Path | str = DEFAULT_SPLIT_PATH) -> dict[str, list[str]]:
    """Load the frozen split and validate its basic invariants.

    Args:
        split_path: Path to the frozen split JSON.

    Returns:
        Mapping of split name (train, train_eval, val, test) to sorted
        puzzle ID lists.

    Raises:
        FileNotFoundError: If the split file does not exist.
        ValueError: If the splits overlap or train_eval is not a subset
            of train.
    """
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Frozen split not found at {split_path}. Generate it once with "
            "'python -m experiments.exp20_realistic_pieces.splits --source-dir <puzzles dir>' "
            "and check it into the repo."
        )
    with open(split_path) as f:
        data = json.load(f)

    split = {key: list(data[key]) for key in SPLIT_KEYS}

    train, train_eval, val, test = (set(split[key]) for key in SPLIT_KEYS)
    if not train_eval <= train:
        raise ValueError("Invalid split: train_eval must be a subset of train")
    for name_a, ids_a, name_b, ids_b in (
        ("train", train, "val", val),
        ("train", train, "test", test),
        ("val", val, "test", test),
    ):
        overlap = ids_a & ids_b
        if overlap:
            raise ValueError(f"Invalid split: {name_a} and {name_b} overlap ({len(overlap)} puzzles)")

    return split


def main() -> None:
    """Freeze the split, verify it, and write the JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory with source puzzle JPEGs")
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_PATH, help="Where to write the split JSON")
    parser.add_argument(
        "--verify-test-ids",
        type=Path,
        default=None,
        help="Optional test_ids.txt (from the regenerated test set) that the test portion must match exactly",
    )
    args = parser.parse_args()

    split = freeze_split(args.source_dir)
    train, train_eval, val, test = (split[key] for key in SPLIT_KEYS)
    assert isinstance(train, list) and isinstance(val, list) and isinstance(test, list)
    assert isinstance(train_eval, list)
    print(f"Split sizes: train={len(train)}, train_eval={len(train_eval)}, val={len(val)}, test={len(test)}")

    if args.verify_test_ids is not None:
        expected = sorted(args.verify_test_ids.read_text().split())
        if expected != test:
            raise SystemExit(
                f"Test portion MISMATCH vs {args.verify_test_ids}: "
                f"{len(expected)} expected vs {len(test)} derived. Not writing."
            )
        print(f"Test portion verified against {args.verify_test_ids} ({len(expected)} puzzles)")

    if args.output.exists():
        raise SystemExit(f"{args.output} already exists — the split is frozen. Bump the version to change it.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(split, f, indent=1)
        f.write("\n")
    print(f"Wrote frozen split to {args.output}")

    # Round-trip through the validating loader as a sanity check.
    load_split(args.output)
    print("Reloaded and validated OK")


if __name__ == "__main__":
    main()
