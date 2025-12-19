#!/usr/bin/env python3
"""Download images from the Unsplash dataset CSV file."""

import argparse
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm  # type: ignore[import-untyped]


def download_images(
    csv_file: Path,
    output_dir: Path,
    count: int,
) -> None:
    """Download images from CSV file.

    Args:
        csv_file: Path to the CSV file with image URLs.
        output_dir: Directory to save downloaded images.
        count: Number of images to download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file, sep="\t", usecols=["photo_image_url"])
    urls = df["photo_image_url"].head(count).tolist()

    print(f"Downloading {len(urls)} images to {output_dir}")

    failed = 0
    skipped = 0
    for i, url in enumerate(tqdm(urls, desc="Downloading"), start=1):
        output_path = output_dir / f"puzzle_{i:03d}.jpg"
        if output_path.exists():
            skipped += 1
            continue
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            output_path.write_bytes(response.content)
        except requests.RequestException as e:
            failed += 1
            tqdm.write(f"Failed to download {url}: {e}")

    downloaded = count - failed - skipped
    print(f"Done. Downloaded {downloaded}, skipped {skipped} existing, {failed} failed.")


def main() -> None:
    """Parse arguments and run download."""
    parser = argparse.ArgumentParser(description="Download images from Unsplash CSV")
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=Path(__file__).parent / "dataset.csv000",
        help="Path to CSV file (default: dataset.csv000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "datasets" / "example" / "raw",
        help="Output directory (default: datasets/example/raw)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of images to download (default: 1000)",
    )
    args = parser.parse_args()

    download_images(args.csv_file, args.output_dir, args.count)


if __name__ == "__main__":
    main()
