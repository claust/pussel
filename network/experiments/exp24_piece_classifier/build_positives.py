"""Build positive (puzzle piece) crops for the piece classifier.

Two sources:

1. Synthetic pieces from the exp20 realistic-piece generator (pieces with
   Bezier tab/blank edges on a black canvas). Point ``--synthetic-root`` at a
   generator output directory (puzzle subdirectories with PNGs).
2. Real piece photos from the north-star capture session (HEIC photos of
   pieces on four background types). Point ``--real-root`` at the photo
   directory. Photos are converted with macOS ``sips`` and segmented with
   rembg, exactly like the backend preview pipeline. The photos themselves
   must never be committed; only local, gitignored crops are produced.

Usage (from network/):
    uv run python -m experiments.exp24_piece_classifier.build_positives \
        --synthetic-root datasets/piece_classifier/synthetic_raw \
        --real-root ~/Pictures/puzzles \
        --output-root datasets/piece_classifier
"""

import argparse
import io
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .data_prep import black_canvas_to_classifier_input, rgba_to_classifier_input

# Crops are stored at most this large; training resizes to 128 anyway.
MAX_CROP_SIZE = 256

# HEIC photos are downscaled to this max dimension before segmentation.
REAL_PHOTO_MAX_DIM = 640

# Overview (full puzzle) shots in the north-star capture session; every other
# IMG_ number in the range is a single-piece shot. IMG_2052 is the retaken
# overview for puzzle 11. See the north-star capture notes.
OVERVIEW_ANCHORS = [1093, 1142, 1191, 1240, 1305, 1406, 1487, 1512, 1561, 1598, 1663, 1760, 1857, 1954]
EXTRA_OVERVIEWS = {2052}


def save_crop(crop: Image.Image, path: Path) -> None:
    """Save a crop, downscaling to MAX_CROP_SIZE when larger.

    Args:
        crop: Square RGB crop.
        path: Destination PNG path.
    """
    if crop.width > MAX_CROP_SIZE:
        crop = crop.resize((MAX_CROP_SIZE, MAX_CROP_SIZE), Image.Resampling.LANCZOS)
    crop.save(path, "PNG")


def build_synthetic(synthetic_root: Path, output_dir: Path) -> int:
    """Process generator output into classifier positives.

    Args:
        synthetic_root: Generator output root (puzzle subdirectories of PNGs).
        output_dir: Directory for the processed crops.

    Returns:
        Number of crops written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    piece_files = sorted(synthetic_root.glob("puzzle_*/*.png"))
    print(f"Processing {len(piece_files)} synthetic pieces from {synthetic_root}")
    for piece_file in piece_files:
        crop = black_canvas_to_classifier_input(Image.open(piece_file))
        if crop is None:
            continue
        # Keep the puzzle id in the name so splits can group by puzzle
        save_crop(crop, output_dir / f"{piece_file.parent.name}__{piece_file.stem}.png")
        count += 1
        if count % 500 == 0:
            print(f"  {count} crops...", end="\r")
    print(f"\nWrote {count} synthetic positives to {output_dir}")
    return count


def puzzle_index_for(img_number: int) -> int:
    """Map a north-star IMG number to its puzzle index (0-based).

    Args:
        img_number: The numeric part of the IMG_ filename.

    Returns:
        Index of the puzzle the shot belongs to.
    """
    idx = 0
    for i, anchor in enumerate(OVERVIEW_ANCHORS):
        if img_number >= anchor:
            idx = i
    return idx


def convert_heic(heic_path: Path, jpeg_path: Path) -> bool:
    """Convert a HEIC photo to a downscaled JPEG using macOS sips.

    Args:
        heic_path: Source HEIC file.
        jpeg_path: Destination JPEG file.

    Returns:
        True on success.
    """
    result = subprocess.run(
        ["sips", "-s", "format", "jpeg", "-Z", str(REAL_PHOTO_MAX_DIM), str(heic_path), "--out", str(jpeg_path)],
        capture_output=True,
    )
    return result.returncode == 0 and jpeg_path.exists()


def build_real(real_root: Path, output_dir: Path, every_nth: int = 1) -> int:
    """Segment real piece photos with rembg and save classifier positives.

    Args:
        real_root: Directory of IMG_*.HEIC photos.
        output_dir: Directory for the processed crops.
        every_nth: Keep only every n-th piece shot (subsampling).

    Returns:
        Number of crops written.
    """
    from rembg import new_session, remove  # local import: heavy, pulls onnxruntime

    session = new_session("u2net")
    output_dir.mkdir(parents=True, exist_ok=True)

    heics = sorted(real_root.glob("IMG_*.HEIC"))
    piece_shots = []
    for heic in heics:
        number = int(heic.stem.split("_")[1])
        if number in OVERVIEW_ANCHORS or number in EXTRA_OVERVIEWS:
            continue
        piece_shots.append((number, heic))
    piece_shots = piece_shots[::every_nth]
    print(f"Segmenting {len(piece_shots)} real piece photos from {real_root}")

    count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for number, heic in piece_shots:
            jpeg_path = Path(tmp) / f"{heic.stem}.jpg"
            if not convert_heic(heic, jpeg_path):
                print(f"  sips failed on {heic.name}, skipping")
                continue
            with open(jpeg_path, "rb") as f:
                rgba = remove(Image.open(io.BytesIO(f.read())).convert("RGB"), session=session)
            crop = rgba_to_classifier_input(rgba)
            if crop is None:
                print(f"  no subject found in {heic.name}, skipping")
                continue
            puzzle_idx = puzzle_index_for(number)
            save_crop(crop, output_dir / f"puzzle{puzzle_idx:02d}__{heic.stem}.png")
            count += 1
            if count % 25 == 0:
                print(f"  {count}/{len(piece_shots)} crops...", end="\r")
    print(f"\nWrote {count} real positives to {output_dir}")
    return count


def main() -> None:
    """Build positive crops from the requested sources."""
    parser = argparse.ArgumentParser(description="Build piece-classifier positives")
    parser.add_argument("--synthetic-root", type=Path, default=None, help="exp20 generator output root")
    parser.add_argument("--real-root", type=Path, default=None, help="Directory of real piece HEIC photos")
    parser.add_argument("--every-nth", type=int, default=1, help="Subsample real photos (keep every n-th)")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dataset root; crops land in <root>/positives/{synthetic,real}",
    )
    args = parser.parse_args()

    if args.synthetic_root is None and args.real_root is None:
        parser.error("Provide --synthetic-root and/or --real-root")
    if args.synthetic_root is not None:
        build_synthetic(args.synthetic_root.expanduser(), args.output_root / "positives" / "synthetic")
    if args.real_root is not None:
        build_real(args.real_root.expanduser(), args.output_root / "positives" / "real", args.every_nth)


if __name__ == "__main__":
    main()
