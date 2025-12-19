#!/usr/bin/env python
"""Puzzle Image Resizer.

This utility takes a dataset of puzzle images and resizes them to 512x512
pixels. All puzzle images matching the pattern puzzle_*.jpg will be processed.
"""

import argparse
import os

from PIL import Image
from PIL.Image import Resampling


def resize_image(input_path: str, output_path: str, size: tuple = (512, 512)):
    """Directly resize an image to the specified size without preserving ratio.

    Args:
        input_path: Path to the input image
        output_path: Path to save the resized image
        size: Target size (width, height)
    """
    with Image.open(input_path) as img:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Directly resize to target dimensions
        resized_img = img.resize(size, Resampling.LANCZOS)

        # Save the resized image
        resized_img.save(output_path, quality=95)

        print(f"Resized {input_path} → {output_path}")


def resize_dataset(dataset_dir: str, output_dir: str | None = None, size: tuple = (512, 512)):
    """Resize all puzzle images in a dataset directory.

    Args:
        dataset_dir: Directory containing puzzle images
        output_dir: Directory to save resized images (defaults to same dir)
        size: Target size (width, height)
    """
    if output_dir is None:
        output_dir = dataset_dir

    os.makedirs(output_dir, exist_ok=True)

    # Find all puzzle images
    count = 0
    for filename in os.listdir(dataset_dir):
        if filename.lower().startswith("puzzle_") and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            input_path = os.path.join(dataset_dir, filename)

            # Generate output filename
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}{ext}")

            # Resize the image
            resize_image(input_path, output_path, size)
            count += 1

    print(f"Resized {count} puzzle images to {size[0]}x{size[1]}")


def main():
    """Process command-line arguments and run the puzzle image resizer."""
    parser = argparse.ArgumentParser(description="Resize puzzle images to 512x512")
    parser.add_argument("dataset_dir", help="Directory containing puzzle images")
    parser.add_argument("--output-dir", help="Directory to save resized images (default: same as input)")
    parser.add_argument("--width", type=int, default=512, help="Target width")
    parser.add_argument("--height", type=int, default=512, help="Target height")

    args = parser.parse_args()

    # Handle paths
    dataset_dir = os.path.abspath(args.dataset_dir)

    output_dir = args.output_dir
    if output_dir:
        output_dir = os.path.abspath(output_dir)

    resize_dataset(dataset_dir, output_dir, (args.width, args.height))


if __name__ == "__main__":
    main()
