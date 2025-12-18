#!/usr/bin/env python
r"""Puzzle Generator and Preprocessor.

This utility takes complete puzzle images and:
1. Generates puzzle pieces with varying rotations
2. Processes the pieces for machine learning model training
3. Creates metadata for position and rotation prediction tasks
4. Applies data augmentation for more robust training (using Albumentations library)

The generator maintains a single unified pipeline from original puzzles
to model-ready data, avoiding redundant storage of intermediate pieces.

Dependencies:
  - numpy, Pillow, albumentations

Example usage:
  # Basic processing
  python puzzle_generator.py puzzle.jpg --output-dir datasets/processed

  # With augmentation
  python puzzle_generator.py puzzles/ --output-dir datasets/processed --augment \
    --brightness-range 0.8 1.2 --contrast-range 0.8 1.2 \
    --color-shift-range 0.9 1.1 --rotation-range 15 \
    --zoom-range 0.9 1.1 --random-crop 0.1

  # With custom piece and context sizes
  python puzzle_generator.py puzzles/ --output-dir datasets/processed \
    --context-size 256 256 --piece-size 224 224 --validation-split 0.2
"""


import argparse
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

import albumentations as A  # type: ignore[import]
import numpy as np
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

# Type aliases to shorten long return type annotations
BBox = Tuple[int, int, int, int]
PieceData = Tuple[Image.Image, BBox, int]


@dataclass
class ProcessingOptions:
    """Options for processing puzzle pieces."""

    # Piece generation options
    num_pieces: int = 1000
    piece_output_size: Tuple[int, int] = (224, 224)  # Size to save pieces at

    # Puzzle context image options (pieces are cut from original, not this)
    puzzle_context_size: Tuple[int, int] = (256, 256)  # Size for context image

    # Training/validation split
    validation_split: float = 0.2

    # Augmentation options (to be used later)
    use_augmentation: bool = False
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    # Additional augmentation options
    color_shift_range: Tuple[float, float] = (0.9, 1.1)  # RGB channel multipliers
    rotation_range: int = 0  # Max degrees for non-90° rotations
    shear_range: float = 0.0  # Max shear angle in degrees
    zoom_range: Tuple[float, float] = (0.9, 1.1)  # Zoom factor
    random_crop_percent: float = 0.0  # How much to randomly crop and pad

    # Processing options
    normalize_pixels: bool = True

    # Output format options
    output_format: str = "jpeg"  # "jpeg" (fast) or "png" (with transparency)
    jpeg_quality: int = 85  # JPEG quality (1-100)
    save_threads: int = 4  # Number of threads for parallel piece saving

    @classmethod
    def from_json(cls, json_path: str) -> "ProcessingOptions":
        """Create options from a JSON configuration file."""
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(**config)


class MetadataHandler:
    """Handles creation and updating of metadata files for puzzle pieces."""

    def __init__(self, metadata_path: str, create_new: bool = True):
        """Initialize the metadata handler.

        Args:
            metadata_path: Path to the metadata CSV file
            create_new: Whether to create a new file (True) or append (False)
        """
        self.metadata_path = metadata_path
        self.header = [
            "piece_id",
            "puzzle_id",
            "filename",
            "x1",
            "y1",
            "x2",
            "y2",
            "rotation",
            "normalized_x1",
            "normalized_y1",
            "normalized_x2",
            "normalized_y2",
            "split",
        ]

        if create_new:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

            # Create the CSV file with header
            with open(metadata_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def add_piece(
        self,
        piece_id: str,
        puzzle_id: str,
        filename: str,
        bbox: BBox,
        rotation: int,
        original_size: Tuple[int, int],
        split: str = "train",
    ) -> None:
        """Add a puzzle piece to the metadata file.

        Args:
            piece_id: Unique ID for the piece
            puzzle_id: ID of the source puzzle
            filename: Relative path to the piece image
            bbox: Original bounding box (x1, y1, x2, y2)
            rotation: Rotation value (0, 90, 180, 270)
            original_size: Original puzzle dimensions (width, height) for normalization
            split: Dataset split ('train' or 'validation')
        """
        # Calculate normalized coordinates using original puzzle dimensions
        x1, y1, x2, y2 = bbox
        width, height = original_size

        # Calculate normalized coordinates
        normalized_coords = {
            "x1": x1 / width,
            "y1": y1 / height,
            "x2": x2 / width,
            "y2": y2 / height,
        }

        # Create row data
        row = [
            piece_id,
            puzzle_id,
            filename,
            x1,
            y1,
            x2,
            y2,
            rotation,
            normalized_coords["x1"],
            normalized_coords["y1"],
            normalized_coords["x2"],
            normalized_coords["y2"],
            split,
        ]

        # Append to CSV
        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def create_split_files(self, output_dir: str) -> None:
        """Create train.txt and val.txt files from the metadata.

        Args:
            output_dir: Directory to save the split files
        """
        train_pieces = []
        val_pieces = []

        # Read metadata and separate by split
        with open(self.metadata_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == "train":
                    train_pieces.append(row["piece_id"])
                else:
                    val_pieces.append(row["piece_id"])

        # Write train.txt
        train_path = os.path.join(output_dir, "train.txt")
        with open(train_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_pieces))

        # Write val.txt
        val_path = os.path.join(output_dir, "val.txt")
        with open(val_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_pieces))


class PuzzleProcessor:
    """Processes puzzle images for piece generation and model training."""

    def __init__(self, options: ProcessingOptions):
        """Initialize the puzzle processor.

        Args:
            options: Processing options
        """
        self.options = options

        # Initialize augmentation pipeline if enabled
        self.aug_pipeline = None
        if options.use_augmentation:
            self._setup_augmentation_pipeline()

    def _setup_augmentation_pipeline(self):
        """Set up the augmentation pipeline using albumentations."""
        min_bright, max_bright = self.options.brightness_range
        min_contrast, max_contrast = self.options.contrast_range
        min_color, max_color = self.options.color_shift_range
        min_zoom, max_zoom = self.options.zoom_range

        # Create transformations pipeline
        self.aug_pipeline = A.Compose(
            [
                # Basic color adjustments
                A.RandomBrightnessContrast(
                    brightness_limit=(min_bright - 1.0, max_bright - 1.0),
                    contrast_limit=(min_contrast - 1.0, max_contrast - 1.0),
                    p=0.5,
                ),
                # Color shifts (RGB adjustments)
                A.RGBShift(
                    r_shift_limit=(min_color - 1.0, max_color - 1.0),
                    g_shift_limit=(min_color - 1.0, max_color - 1.0),
                    b_shift_limit=(min_color - 1.0, max_color - 1.0),
                    p=0.5,
                ),
                # Affine transformations
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=(min_zoom - 1.0, max_zoom - 1.0),
                    rotate_limit=self.options.rotation_range,
                    p=0.5,
                    border_mode=0,
                ),
                # Shear transformation
                A.IAAAffine(
                    shear=(-self.options.shear_range, self.options.shear_range),
                    p=0.5 if self.options.shear_range > 0 else 0,
                    mode="constant",
                ),
                # Random cropping
                A.RandomSizedCrop(
                    min_height=int((1.0 - self.options.random_crop_percent) * 100),
                    max_height=100,
                    height=100,
                    width=100,
                    p=0.5 if self.options.random_crop_percent > 0 else 0,
                ),
            ]
        )

    def create_context_image(self, image: Image.Image) -> Image.Image:
        """Create a smaller context image for training.

        The context image is used as puzzle context during training,
        while pieces are cut from the original high-resolution image.

        Args:
            image: Input puzzle image (original resolution)

        Returns:
            Context image at puzzle_context_size
        """
        width, height = self.options.puzzle_context_size
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def generate_mask(self, width: int, height: int) -> np.ndarray:
        """Generate a mask for cutting an image into jigsaw puzzle pieces.

        Args:
            width: Width of the image
            height: Height of the image

        Returns:
            A numpy array of labels where each unique value represents a piece
        """
        # Calculate number of pieces based on target count
        piece_size = int(np.sqrt((width * height) / self.options.num_pieces))

        # Calculate number of pieces across and down
        pieces_x = max(2, width // piece_size)
        pieces_y = max(2, height // piece_size)

        # Create a grid of initial piece IDs
        piece_grid = np.zeros((height, width), dtype=np.int32)
        for y in range(pieces_y):
            for x in range(pieces_x):
                # Create boundaries for each piece
                y_start = int(y * height / pieces_y)
                y_end = int((y + 1) * height / pieces_y)
                x_start = int(x * width / pieces_x)
                x_end = int((x + 1) * width / pieces_x)

                # Assign unique ID to each piece
                piece_id = y * pieces_x + x + 1
                piece_grid[y_start:y_end, x_start:x_end] = piece_id

        # Return regular grid without distortion
        return piece_grid

    def process_piece(
        self,
        piece_img: Image.Image,
        # We need to accept these arguments for API consistency
        bbox: Optional[BBox] = None,
        rotation: int = 0,
    ) -> Image.Image:
        """Process a puzzle piece for model input.

        Args:
            piece_img: Original piece image
            bbox: Bounding box coordinates (unused, kept for API consistency)
            rotation: Rotation value (unused, kept for API consistency)

        Returns:
            Processed piece image
        """
        # Apply augmentations if enabled
        if self.options.use_augmentation and self.aug_pipeline is not None:
            # Convert PIL to numpy for albumentations
            img_array = np.array(piece_img)

            # Apply augmentations
            augmented = self.aug_pipeline(image=img_array)

            # Convert back to PIL
            piece_img = Image.fromarray(augmented["image"])

        # Resize the piece to the output size
        target_width, target_height = self.options.piece_output_size
        processed_img = piece_img.resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        )

        # Normalize pixel values if requested
        if self.options.normalize_pixels:
            processed_img = self._normalize_image(processed_img)

        return processed_img

    def _normalize_image(self, img: Image.Image) -> Image.Image:
        """Normalize pixel values of an image.

        Args:
            img: Input image

        Returns:
            Normalized image
        """
        # Convert to numpy array for normalization
        img_array = np.array(img).astype(np.float32)

        # Handle alpha channel properly
        if img_array.shape[-1] == 4:  # RGBA
            # Normalize RGB channels to 0-1, then back to 0-255
            # Keep alpha channel unchanged to preserve transparency
            img_array[..., :3] = img_array[..., :3] / 255.0
            img_array[..., :3] = img_array[..., :3] * 255.0
        else:  # RGB
            img_array = img_array / 255.0
            img_array = img_array * 255.0

        # Convert back to PIL
        return Image.fromarray(img_array.astype(np.uint8))

    def extract_piece(
        self, image: Image.Image, mask: np.ndarray, piece_id: int
    ) -> Optional[PieceData]:
        """Extract and process a single puzzle piece.

        Args:
            image: Source puzzle image
            mask: Piece mask array
            piece_id: ID of the piece to extract

        Returns:
            Tuple of (original piece image, bounding box, rotation) or None
        """
        img_array = np.array(image)
        return self.extract_piece_from_array(img_array, mask, piece_id)

    def extract_piece_from_array(
        self, img_array: np.ndarray, mask: np.ndarray, piece_id: int
    ) -> Optional[PieceData]:
        """Extract and process a single puzzle piece from numpy array.

        Args:
            img_array: Source puzzle image as numpy array
            mask: Piece mask array
            piece_id: ID of the piece to extract

        Returns:
            Tuple of (original piece image, bounding box, rotation) or None
        """
        # Create a binary mask for this piece
        piece_mask = mask == piece_id

        # Find bounding box using numpy (faster than PIL)
        rows = np.any(piece_mask, axis=1)
        cols = np.any(piece_mask, axis=0)
        if not rows.any() or not cols.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        y2 += 1  # Make exclusive
        x2 += 1

        bbox = (x1, y1, x2, y2)

        # Extract region from image and mask using numpy (vectorized)
        region = img_array[y1:y2, x1:x2].copy()
        mask_region = piece_mask[y1:y2, x1:x2]

        # Create RGBA output array
        h, w = region.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Copy RGB where mask is True, set alpha to 255
        rgba[mask_region, :3] = region[mask_region, :3]
        rgba[mask_region, 3] = 255

        piece_img = Image.fromarray(rgba, mode="RGBA")

        # Random rotation for training diversity
        rotation = random.choice([0, 90, 180, 270])
        if rotation:
            piece_img = piece_img.rotate(rotation, expand=True)

        return piece_img, bbox, rotation

    def process_puzzle(
        self, puzzle_path: str, output_dir: str, metadata_handler: MetadataHandler
    ) -> int:
        """Process a puzzle image into pieces for model training.

        Args:
            puzzle_path: Path to the puzzle image
            output_dir: Base directory for saving processed data
            metadata_handler: Handler for piece metadata

        Returns:
            Number of pieces generated
        """
        return process_puzzle_helper(self, puzzle_path, output_dir, metadata_handler)


def _process_single_puzzle_worker(
    args: Tuple[str, str, ProcessingOptions],
) -> List[dict]:
    """Worker function to process a single puzzle image.

    This function is designed to be called by ProcessPoolExecutor.
    It returns metadata entries instead of writing directly to avoid
    concurrent file access issues.

    Args:
        args: Tuple of (puzzle_path, output_dir, options)

    Returns:
        List of metadata entry dictionaries
    """
    puzzle_path, output_dir, options = args
    processor = PuzzleProcessor(options)

    # Create directories
    dirs = _prepare_directories(output_dir)

    # Get puzzle name and ID
    puzzle_name = os.path.splitext(os.path.basename(puzzle_path))[0]
    puzzle_id = puzzle_name

    # Load and process puzzle image
    original_image, original_size = _process_puzzle_image(
        processor, puzzle_path, dirs["puzzles"], puzzle_name
    )

    # Convert image to numpy once (avoid repeated conversion per piece)
    img_array = np.array(original_image)

    # Generate mask
    width, height = original_size
    mask = processor.generate_mask(width, height)

    # Process pieces
    unique_ids = np.unique(mask)
    piece_ids = unique_ids[1:]  # Skip background (0)

    # Create puzzle subdirectory once
    puzzle_subdir = os.path.join(dirs["pieces"], puzzle_id)
    os.makedirs(puzzle_subdir, exist_ok=True)

    # Process pieces in parallel using threads (I/O bound)
    def process_one_piece(pid: int) -> Optional[dict]:
        return _process_piece_for_worker(
            processor,
            img_array,
            mask,
            pid,
            dirs["pieces"],
            puzzle_id,
            original_size,
            options,
        )

    metadata_entries: List[dict] = []

    if options.save_threads > 1:
        # Parallel piece processing with threads
        with ThreadPoolExecutor(max_workers=options.save_threads) as executor:
            results = executor.map(process_one_piece, piece_ids)
            metadata_entries = [e for e in results if e is not None]
    else:
        # Sequential processing
        for piece_id in piece_ids:
            entry = process_one_piece(piece_id)
            if entry:
                metadata_entries.append(entry)

    return metadata_entries


def _process_piece_for_worker(
    processor: PuzzleProcessor,
    img_array: np.ndarray,
    mask: np.ndarray,
    piece_id: int,
    pieces_dir: str,
    puzzle_id: str,
    original_size: Tuple[int, int],
    options: ProcessingOptions,
) -> Optional[dict]:
    """Process a single piece and return metadata entry.

    Args:
        processor: The PuzzleProcessor instance
        img_array: Original high-resolution puzzle image as numpy array
        mask: Piece mask array
        piece_id: ID of the piece to extract
        pieces_dir: Directory to save piece images
        puzzle_id: ID of the source puzzle
        original_size: Original puzzle dimensions (width, height)
        options: Processing options

    Returns:
        Metadata entry dictionary or None
    """
    # Extract the piece
    piece_data = processor.extract_piece_from_array(img_array, mask, piece_id)
    if piece_data is None:
        return None

    original_piece, bbox, rotation = piece_data

    # Determine split
    split = "validation" if random.random() < options.validation_split else "train"

    # Create filename based on output format
    piece_id_str = f"{puzzle_id}_{piece_id:03d}"
    ext = "jpg" if options.output_format == "jpeg" else "png"
    filename = f"{piece_id_str}_r{rotation}.{ext}"

    # Process and save piece
    processed_piece = processor.process_piece(original_piece, bbox, rotation)

    # Save in puzzle-specific subdirectory
    puzzle_subdir = os.path.join(pieces_dir, puzzle_id)
    piece_path = os.path.join(puzzle_subdir, filename)

    # Save with format-specific options
    if options.output_format == "jpeg":
        # Convert RGBA to RGB for JPEG (no alpha channel support)
        if processed_piece.mode == "RGBA":
            rgb_piece = Image.new("RGB", processed_piece.size, (255, 255, 255))
            rgb_piece.paste(processed_piece, mask=processed_piece.split()[3])
            processed_piece = rgb_piece
        processed_piece.save(piece_path, "JPEG", quality=options.jpeg_quality)
    else:
        # PNG with minimal compression for speed
        processed_piece.save(piece_path, "PNG", compress_level=1)

    # Return metadata entry (don't write to file yet)
    x1, y1, x2, y2 = bbox
    width, height = original_size

    return {
        "piece_id": piece_id_str,
        "puzzle_id": puzzle_id,
        "filename": os.path.join("pieces", puzzle_id, filename),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "rotation": rotation,
        "normalized_x1": x1 / width,
        "normalized_y1": y1 / height,
        "normalized_x2": x2 / width,
        "normalized_y2": y2 / height,
        "split": split,
    }


def process_directory(
    input_dir: str, output_dir: str, options: ProcessingOptions, num_workers: int = 1
) -> int:
    """Process all puzzle images in a directory.

    Args:
        input_dir: Input directory containing puzzle images
        output_dir: Output directory for processed data
        options: Processing options
        num_workers: Number of parallel workers (default: 1)

    Returns:
        Total number of pieces generated
    """
    # Collect all puzzle images first
    puzzle_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Prepare worker arguments
    worker_args = [
        (os.path.join(input_dir, filename), output_dir, options)
        for filename in puzzle_files
    ]

    # Collect all metadata entries
    all_metadata: List[dict] = []

    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_puzzle_worker, args): args[0]
                for args in worker_args
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing puzzles",
                unit="puzzle",
            ):
                metadata_entries = future.result()
                all_metadata.extend(metadata_entries)
    else:
        # Sequential processing (original behavior)
        for args in tqdm(worker_args, desc="Processing puzzles", unit="puzzle"):
            metadata_entries = _process_single_puzzle_worker(args)
            all_metadata.extend(metadata_entries)

    # Write all metadata at once
    metadata_path = os.path.join(output_dir, "metadata.csv")
    _write_metadata(metadata_path, all_metadata)

    # Create split files
    _create_split_files_from_metadata(all_metadata, output_dir)

    return len(all_metadata)


def _write_metadata(metadata_path: str, entries: List[dict]) -> None:
    """Write all metadata entries to CSV file.

    Args:
        metadata_path: Path to the metadata CSV file
        entries: List of metadata entry dictionaries
    """
    header = [
        "piece_id",
        "puzzle_id",
        "filename",
        "x1",
        "y1",
        "x2",
        "y2",
        "rotation",
        "normalized_x1",
        "normalized_y1",
        "normalized_x2",
        "normalized_y2",
        "split",
    ]

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(entries)


def _create_split_files_from_metadata(entries: List[dict], output_dir: str) -> None:
    """Create train.txt and val.txt files from metadata entries.

    Args:
        entries: List of metadata entry dictionaries
        output_dir: Directory to save the split files
    """
    train_pieces = [e["piece_id"] for e in entries if e["split"] == "train"]
    val_pieces = [e["piece_id"] for e in entries if e["split"] == "validation"]

    train_path = os.path.join(output_dir, "train.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_pieces))

    val_path = os.path.join(output_dir, "val.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_pieces))


def process_puzzle_helper(
    processor,
    puzzle_path,
    output_dir,
    metadata_handler,
    puzzle_name=None,
    puzzle_id=None,
):
    """Helper function to reduce local variables in process_puzzle.

    Args:
        processor: The PuzzleProcessor instance
        puzzle_path: Path to the puzzle image
        output_dir: Base directory for saving processed data
        metadata_handler: Handler for piece metadata
        puzzle_name: Optional name override
        puzzle_id: Optional ID override

    Returns:
        Number of pieces generated
    """
    # Create directories and prepare paths
    dirs = _prepare_directories(output_dir)

    # Get the puzzle name and ID if not provided
    if puzzle_name is None:
        puzzle_name = os.path.splitext(os.path.basename(puzzle_path))[0]
    if puzzle_id is None:
        puzzle_id = puzzle_name

    # Load original image and create context image
    original_image, original_size = _process_puzzle_image(
        processor, puzzle_path, dirs["puzzles"], puzzle_name
    )

    # Generate mask from original high-resolution image
    width, height = original_size
    mask = processor.generate_mask(width, height)

    # Process pieces with progress bar
    processed_count = 0
    unique_ids = np.unique(mask)
    piece_ids = unique_ids[1:]  # Skip background (0)

    for piece_id in tqdm(
        piece_ids,
        desc=f"Extracting pieces from {puzzle_name}",
        unit="piece",
        leave=False,
    ):
        processed_count += process_single_piece(
            processor,
            original_image,
            mask,
            piece_id,
            dirs["pieces"],
            puzzle_id,
            metadata_handler,
            original_size,
        )

    return processed_count


def _prepare_directories(output_dir):
    """Prepare output directories for puzzle processing.

    Args:
        output_dir: Base output directory

    Returns:
        Dictionary of directory paths
    """
    pieces_dir = os.path.join(output_dir, "pieces")
    puzzles_dir = os.path.join(output_dir, "puzzles")
    os.makedirs(pieces_dir, exist_ok=True)
    os.makedirs(puzzles_dir, exist_ok=True)

    return {"pieces": pieces_dir, "puzzles": puzzles_dir}


def _process_puzzle_image(processor, puzzle_path, puzzles_dir, puzzle_name):
    """Load the puzzle image and create context image for training.

    Pieces are cut from the original high-resolution image to preserve detail.
    A smaller context image is saved for use during training.

    Args:
        processor: The PuzzleProcessor instance
        puzzle_path: Path to the puzzle image
        puzzles_dir: Directory to save context images
        puzzle_name: Name of the puzzle

    Returns:
        Tuple of (original_image, original_size) where original_size is (width, height)
    """
    # Load the image at original resolution
    original_image = Image.open(puzzle_path).convert("RGB")
    original_size = original_image.size  # (width, height)

    # Create and save context image (smaller, for training)
    context_image = processor.create_context_image(original_image)
    context_path = os.path.join(puzzles_dir, f"{puzzle_name}.jpg")
    context_image.save(context_path)

    # Return original for piece extraction, plus dimensions for normalization
    return original_image, original_size


def process_single_piece(
    processor,
    original_image,
    mask,
    piece_id,
    pieces_dir,
    puzzle_id,
    metadata_handler,
    original_size,
):
    """Process a single puzzle piece to reduce complexity.

    Args:
        processor: The PuzzleProcessor instance
        original_image: Original high-resolution puzzle image
        mask: Piece mask array
        piece_id: ID of the piece to extract
        pieces_dir: Directory to save piece images
        puzzle_id: ID of the source puzzle
        metadata_handler: Handler for piece metadata
        original_size: Original puzzle dimensions (width, height) for normalization

    Returns:
        1 if piece was processed, 0 otherwise
    """
    # Extract the piece from original high-resolution image
    piece_data = processor.extract_piece(original_image, mask, piece_id)
    if piece_data is None:
        return 0

    # Process piece and save
    piece_info = _create_piece_info(piece_data, puzzle_id, piece_id, processor.options)
    _save_and_record_piece(
        processor, piece_info, pieces_dir, metadata_handler, original_size
    )

    return 1


def _create_piece_info(piece_data, puzzle_id, piece_id, options):
    """Create piece information dictionary from extracted piece data.

    Args:
        piece_data: Tuple of (original_piece, bbox, rotation)
        puzzle_id: ID of the source puzzle
        piece_id: ID of the piece
        options: Processing options

    Returns:
        Dictionary with piece information
    """
    original_piece, bbox, rotation = piece_data

    # Determine the split (train or validation)
    split = "validation" if random.random() < options.validation_split else "train"

    return {
        "piece": original_piece,
        "bbox": bbox,
        "rotation": rotation,
        "id": f"{puzzle_id}_{piece_id:03d}",
        "puzzle_id": puzzle_id,
        "split": split,
        "filename": f"{puzzle_id}_{piece_id:03d}_r{rotation}.png",
    }


def _save_and_record_piece(
    processor, piece_info, pieces_dir, metadata_handler, original_size
):
    """Save processed piece and record its metadata.

    Args:
        processor: The PuzzleProcessor instance
        piece_info: Dictionary with piece information
        pieces_dir: Directory to save piece images
        metadata_handler: Handler for piece metadata
        original_size: Original puzzle dimensions (width, height) for normalization
    """
    # Process the piece for the model
    processed_piece = processor.process_piece(
        piece_info["piece"], piece_info["bbox"], piece_info["rotation"]
    )

    # Save the processed piece in puzzle-specific subdirectory
    puzzle_id = piece_info["puzzle_id"]
    puzzle_subdir = os.path.join(pieces_dir, puzzle_id)
    os.makedirs(puzzle_subdir, exist_ok=True)

    piece_path = os.path.join(puzzle_subdir, piece_info["filename"])
    processed_piece.save(piece_path)

    # Add to metadata (normalized by original puzzle dimensions)
    metadata_handler.add_piece(
        piece_id=piece_info["id"],
        puzzle_id=puzzle_id,
        filename=os.path.join("pieces", puzzle_id, piece_info["filename"]),
        bbox=piece_info["bbox"],
        rotation=piece_info["rotation"],
        original_size=original_size,
        split=piece_info["split"],
    )


def main():
    """Process command-line arguments and run the puzzle processor."""
    parser = argparse.ArgumentParser(
        description="Generate and process puzzle pieces for model training"
    )
    parser.add_argument(
        "input_path", help="Path to a puzzle image or directory of puzzle images"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the processed data (defaults to input directory)",
    )
    parser.add_argument(
        "--pieces",
        type=int,
        default=1000,
        help="Approximate number of pieces to generate per puzzle",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("WIDTH", "HEIGHT"),
        help="Size for puzzle context image (pieces are cut from original resolution)",
    )
    parser.add_argument(
        "--piece-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("WIDTH", "HEIGHT"),
        help="Size to resize pieces to after extraction",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of pieces to use for validation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers for processing (default: 5)",
    )
    parser.add_argument(
        "--format",
        choices=["jpeg", "png"],
        default="jpeg",
        help="Output format for pieces: jpeg (fast) or png (with transparency)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85)",
    )
    parser.add_argument(
        "--save-threads",
        type=int,
        default=8,
        help="Threads for parallel piece saving per puzzle (default: 8)",
    )
    parser.add_argument("--config", help="Path to JSON configuration file")

    # Add augmentation arguments
    aug_group = parser.add_argument_group("Augmentation options")
    aug_group.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation",
    )
    aug_group.add_argument(
        "--brightness-range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        metavar=("MIN", "MAX"),
        help="Range for brightness adjustment",
    )
    aug_group.add_argument(
        "--contrast-range",
        type=float,
        nargs=2,
        default=[0.8, 1.2],
        metavar=("MIN", "MAX"),
        help="Range for contrast adjustment",
    )
    aug_group.add_argument(
        "--color-shift-range",
        type=float,
        nargs=2,
        default=[0.9, 1.1],
        metavar=("MIN", "MAX"),
        help="Range for RGB channel multipliers",
    )
    aug_group.add_argument(
        "--rotation-range",
        type=int,
        default=0,
        help="Max degrees for small random rotations",
    )
    aug_group.add_argument(
        "--shear-range",
        type=float,
        default=0.0,
        help="Max shear angle in degrees",
    )
    aug_group.add_argument(
        "--zoom-range",
        type=float,
        nargs=2,
        default=[0.9, 1.1],
        metavar=("MIN", "MAX"),
        help="Range for random zoom",
    )
    aug_group.add_argument(
        "--random-crop",
        type=float,
        default=0.0,
        help="Percentage of image to randomly crop and pad",
    )

    # Parse arguments and create options
    args = parser.parse_args()
    options = _create_options(args)

    # Process the input
    _process_input(args.input_path, args.output_dir, options, args.workers)


def _create_options(args):
    """Create processing options from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ProcessingOptions instance
    """
    if args.config:
        return ProcessingOptions.from_json(args.config)

    options = ProcessingOptions(
        num_pieces=args.pieces,
        puzzle_context_size=tuple(args.context_size),
        piece_output_size=tuple(args.piece_size),
        validation_split=args.validation_split,
        output_format=args.format,
        jpeg_quality=args.jpeg_quality,
        save_threads=args.save_threads,
    )

    # Set augmentation options
    if hasattr(args, "augment"):
        options.use_augmentation = args.augment
        options.brightness_range = tuple(args.brightness_range)
        options.contrast_range = tuple(args.contrast_range)
        options.color_shift_range = tuple(args.color_shift_range)
        options.rotation_range = args.rotation_range
        options.shear_range = args.shear_range
        options.zoom_range = tuple(args.zoom_range)
        options.random_crop_percent = args.random_crop

    return options


def _get_default_output_dir(input_path: str) -> str:
    """Determine the default output directory based on input path.

    Goes up 2 levels from the input directory to find the dataset root.
    E.g., datasets/example/raw/puzzles -> datasets/example

    Args:
        input_path: Absolute path to the input directory

    Returns:
        Default output directory path (grandparent of input)
    """
    # Go up 2 levels: input/.. -> parent, input/../.. -> grandparent
    return os.path.dirname(os.path.dirname(input_path))


def _process_input(input_path, output_dir_path, options, num_workers=1):
    """Process the input image or directory.

    Args:
        input_path: Path to the input image or directory
        output_dir_path: Path to the output directory (None to use input directory)
        options: Processing options
        num_workers: Number of parallel workers for processing
    """
    # Process the input path first to determine default output
    input_path = os.path.abspath(input_path)

    # Default output to input directory (or parent directory for single files)
    if output_dir_path is None:
        if os.path.isdir(input_path):
            output_dir = _get_default_output_dir(input_path)
        else:
            output_dir = os.path.dirname(input_path)
    else:
        output_dir = os.path.abspath(output_dir_path)

    os.makedirs(output_dir, exist_ok=True)

    # Process the input
    if os.path.isdir(input_path):
        total_pieces = process_directory(input_path, output_dir, options, num_workers)
        print(f"Total: Generated {total_pieces} pieces in {output_dir}")
    else:
        _process_single_file(input_path, output_dir, options)


def _process_single_file(input_path, output_dir, options):
    """Process a single puzzle image.

    Args:
        input_path: Path to the puzzle image
        output_dir: Path to the output directory
        options: Processing options
    """
    # Setup for single file
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_handler = MetadataHandler(metadata_path, create_new=True)

    # Process the single puzzle
    processor = PuzzleProcessor(options)
    pieces_count = processor.process_puzzle(input_path, output_dir, metadata_handler)

    # Create split files
    metadata_handler.create_split_files(output_dir)

    print(f"Generated {pieces_count} pieces in {output_dir}")


if __name__ == "__main__":
    main()
