"""Build negative (not-a-piece) crops for the piece classifier.

Negatives must look like real pipeline inputs — things rembg actually
segments out of a camera frame. Two failure modes matter:

- Webcam case: faces and people (Caltech-101 ``Faces``/``Faces_easy``,
  COCO128 person images).
- Mobile case: household/table objects (cups, phones, watches, scissors...).

Every source image goes through the same pipeline as the backend preview:
downscale to 320px, rembg segmentation, largest component, crop composited
on black and padded square.

Sources (downloaded automatically when missing, both modest in size):
- Caltech-101 (~137 MB): faces + a curated list of object categories.
- COCO128 (~7 MB): 128 real COCO scenes (people, cups, phones on tables).

Usage (from network/):
    uv run python -m experiments.exp24_piece_classifier.build_negatives \
        --output-root datasets/piece_classifier
"""

import argparse
import tarfile
import zipfile
from pathlib import Path

from PIL import Image

from .build_positives import save_crop
from .data_prep import rgba_to_classifier_input

CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

# Caltech-101 categories to use as negatives. Faces cover the webcam false
# positive; the rest are hand-held or table objects a phone camera would see.
CALTECH_CATEGORIES = [
    "Faces",
    "Faces_easy",
    "camera",
    "cellphone",
    "chair",
    "cup",
    "headphone",
    "lamp",
    "laptop",
    "pizza",
    "scissors",
    "soccer_ball",
    "stapler",
    "umbrella",
    "watch",
    "wrench",
]
MAX_PER_CATEGORY = 80

# Mirror of the backend preview pipeline: frames are downscaled before rembg.
PREVIEW_MAX_DIM = 320

# Skip segmentations covering almost nothing (mirror of the backend area gate;
# no upper limit here because close-up faces/objects are exactly the hard case).
MIN_AREA_FRACTION = 0.005


def download(url: str, dest: Path) -> None:
    """Download a file with a progress line unless it already exists.

    Args:
        url: Source URL.
        dest: Destination file path.
    """
    import requests

    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        done = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"  {done / 1e6:.0f} MB", end="\r")
    print(f"\nDownloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def safe_extract_zip(archive_path: Path, dest: Path) -> None:
    """Extract a zip archive, refusing entries that escape the destination (Zip Slip).

    Args:
        archive_path: The zip file to extract.
        dest: Destination directory.

    Raises:
        ValueError: When an archive entry would land outside dest.
    """
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            target = (dest_resolved / member).resolve()
            if not target.is_relative_to(dest_resolved):
                raise ValueError(f"Unsafe zip entry: {member}")
        zf.extractall(dest_resolved)


def ensure_caltech(raw_dir: Path) -> Path:
    """Download and extract Caltech-101, returning the categories directory.

    Args:
        raw_dir: Directory for raw downloads.

    Returns:
        Path to 101_ObjectCategories.
    """
    categories_dir = raw_dir / "101_ObjectCategories"
    if categories_dir.exists():
        return categories_dir
    archive = raw_dir / "caltech-101.zip"
    download(CALTECH_URL, archive)
    safe_extract_zip(archive, raw_dir)
    inner = raw_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
    with tarfile.open(inner) as tf:
        tf.extractall(raw_dir, filter="data")
    return categories_dir


def ensure_coco128(raw_dir: Path) -> Path:
    """Download and extract COCO128, returning the images directory.

    Args:
        raw_dir: Directory for raw downloads.

    Returns:
        Path to the COCO128 images.
    """
    images_dir = raw_dir / "coco128" / "images" / "train2017"
    if images_dir.exists():
        return images_dir
    archive = raw_dir / "coco128.zip"
    download(COCO128_URL, archive)
    safe_extract_zip(archive, raw_dir)
    return images_dir


def segment_to_crop(image_path: Path, session: object) -> Image.Image | None:
    """Run one image through the preview-style rembg pipeline.

    Args:
        image_path: Source image path.
        session: rembg session.

    Returns:
        The classifier-input crop, or None when nothing usable was segmented.
    """
    from rembg import remove

    image = Image.open(image_path).convert("RGB")
    scale = min(1.0, PREVIEW_MAX_DIM / max(image.size))
    if scale < 1.0:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
    rgba = remove(image, session=session)
    if not isinstance(rgba, Image.Image):
        return None
    crop = rgba_to_classifier_input(rgba)
    if crop is None:
        return None
    if crop.width * crop.height < MIN_AREA_FRACTION * image.width * image.height:
        return None
    return crop


def build_source(name: str, images: list[Path], output_dir: Path, session: object) -> int:
    """Segment a list of images and save the crops.

    Args:
        name: Source name for logging.
        images: Image files to process.
        output_dir: Destination directory.
        session: rembg session.

    Returns:
        Number of crops written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in images:
        try:
            crop = segment_to_crop(image_path, session)
        except Exception as exc:  # noqa: B902 - corrupt images should not kill the build
            print(f"  {image_path.name}: {exc}")
            continue
        if crop is None:
            continue
        save_crop(crop, output_dir / f"{image_path.stem}.png")
        count += 1
        if count % 50 == 0:
            print(f"  [{name}] {count} crops...", end="\r")
    print(f"\n[{name}] wrote {count} crops to {output_dir}")
    return count


def main() -> None:
    """Build negative crops from Caltech-101 and COCO128."""
    parser = argparse.ArgumentParser(description="Build piece-classifier negatives")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dataset root; crops land in <root>/negatives/<source>/<category>",
    )
    parser.add_argument("--max-per-category", type=int, default=MAX_PER_CATEGORY)
    args = parser.parse_args()

    from rembg import new_session

    session = new_session("u2net")
    raw_dir = args.output_root / "raw_downloads"
    negatives_root = args.output_root / "negatives"

    categories_dir = ensure_caltech(raw_dir)
    for category in CALTECH_CATEGORIES:
        category_dir = categories_dir / category
        if not category_dir.exists():
            print(f"[caltech101/{category}] missing, skipping")
            continue
        images = sorted(category_dir.glob("*.jpg"))[: args.max_per_category]
        build_source(f"caltech101/{category}", images, negatives_root / "caltech101" / category, session)

    coco_dir = ensure_coco128(raw_dir)
    build_source("coco128", sorted(coco_dir.glob("*.jpg")), negatives_root / "coco128" / "scenes", session)


if __name__ == "__main__":
    main()
