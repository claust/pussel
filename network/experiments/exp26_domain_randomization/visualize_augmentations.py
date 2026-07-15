#!/usr/bin/env python3
r"""Visualize the exp26 domain-randomization augmentations.

Dumps a grid of augmented (piece, puzzle) pairs so the pipeline can be
eyeballed locally before committing GPU time on RunPod. Each row is one
source piece rendered under several independent augmentation draws next to
its (independently jittered) puzzle, so the piece/puzzle photometric
independence is visible.

Run from the network/ directory:
    uv run python -m experiments.exp26_domain_randomization.visualize_augmentations \\
        --dataset-root datasets/realistic_4x4_rgba --n-pieces 6 --n-draws 5
"""

import argparse
from pathlib import Path

from PIL import Image

from ..exp20_realistic_pieces.dataset import parse_piece_filename
from .aug_dataset import DEFAULT_DATASET_ROOT, DEFAULT_PUZZLE_ROOT
from .augment import AUG_PRESETS, AugmentConfig, BackgroundSampler, augment_piece, augment_puzzle, seed_everything


def _find_pieces(dataset_root: Path, n_pieces: int) -> list[Path]:
    """Return up to ``n_pieces`` RGBA piece paths from the dataset root."""
    pieces: list[Path] = []
    for puzzle_dir in sorted(dataset_root.iterdir()):
        if not puzzle_dir.is_dir():
            continue
        found = sorted(puzzle_dir.glob(f"{puzzle_dir.name}_x*_y*_rot*.png"))
        if found:
            pieces.append(found[0])
        if len(pieces) >= n_pieces:
            break
    return pieces


def build_grid(
    dataset_root: Path,
    puzzle_root: Path,
    config: AugmentConfig,
    n_pieces: int,
    n_draws: int,
    cell: int = 128,
) -> Image.Image:
    """Build a contact sheet of augmented pieces and puzzles.

    Args:
        dataset_root: RGBA piece dataset root.
        puzzle_root: Source puzzle JPEG root.
        config: Augmentation config to visualize.
        n_pieces: Number of source pieces (rows).
        n_draws: Number of independent augmentation draws per piece.
        cell: Pixel size of each thumbnail.

    Returns:
        The assembled contact-sheet image. Column 0 is the black-composited
        original piece; columns 1..n_draws are augmented draws; the last
        column is the (jittered) puzzle.
    """
    piece_paths = _find_pieces(dataset_root, n_pieces)
    if not piece_paths:
        raise SystemExit(f"No RGBA pieces found under {dataset_root}. Run generate_dataset.py first.")

    sampler = BackgroundSampler(texture_paths=sorted(puzzle_root.glob("puzzle_*.jpg"))[:64])
    cols = 1 + n_draws + 1
    sheet = Image.new("RGB", (cols * cell, len(piece_paths) * cell), (32, 32, 32))

    for row, piece_path in enumerate(piece_paths):
        parsed = parse_piece_filename(piece_path.name)
        puzzle_id = parsed[0] if parsed else piece_path.parent.name
        with Image.open(piece_path) as raw:
            piece_rgba = raw.convert("RGBA")

        original = Image.new("RGB", piece_rgba.size, (0, 0, 0))
        original.paste(piece_rgba, mask=piece_rgba.getchannel("A"))
        sheet.paste(original.resize((cell, cell)), (0, row * cell))

        for draw in range(n_draws):
            aug = augment_piece(piece_rgba, config, sampler)
            sheet.paste(aug.resize((cell, cell)), ((1 + draw) * cell, row * cell))

        puzzle_path = puzzle_root / f"{puzzle_id}.jpg"
        if puzzle_path.exists():
            with Image.open(puzzle_path) as pim:
                puzzle_rgb = augment_puzzle(pim.convert("RGB"), config)
            sheet.paste(puzzle_rgb.resize((cell, cell)), ((cols - 1) * cell, row * cell))

    return sheet


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualize exp26 augmentations")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="RGBA pieces root")
    parser.add_argument("--puzzle-root", type=Path, default=DEFAULT_PUZZLE_ROOT, help="Source puzzles root")
    parser.add_argument("--aug-preset", choices=sorted(AUG_PRESETS.keys()), default="full", help="Preset to show")
    parser.add_argument("--n-pieces", type=int, default=6, help="Number of source pieces (rows)")
    parser.add_argument("--n-draws", type=int, default=6, help="Augmentation draws per piece")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "outputs" / "augmentation_samples.jpg",
        help="Output image path (JPEG keeps the committed artifact under the pre-commit size limit)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    config = AUG_PRESETS[args.aug_preset]
    sheet = build_grid(
        dataset_root=args.dataset_root,
        puzzle_root=args.puzzle_root,
        config=config,
        n_pieces=args.n_pieces,
        n_draws=args.n_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Saved {args.output} (preset={args.aug_preset}, {args.n_pieces} pieces x {args.n_draws} draws)")


if __name__ == "__main__":
    main()
