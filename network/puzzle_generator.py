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

  # With custom piece and puzzle sizes
  python puzzle_generator.py puzzles/ --output-dir datasets/processed \
    --puzzle-size 512 512 --piece-size 224 224 --validation-split 0.2
"""


import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import albumentations as A  # type: ignore[import]
import numpy as np
from PIL import Image

# Type aliases to shorten long return type annotations
BBox = Tuple[int, int, int, int]
PieceData = Tuple[Image.Image, BBox, int]


@dataclass
class ProcessingOptions:
    """Options for processing puzzle pieces."""

    # Piece generation options
    num_pieces: int = 500
    piece_size: Tuple[int, int] = (224, 224)

    # Puzzle standardization options
    puzzle_size: Tuple[int, int] = (512, 512)

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
        puzzle_size: Tuple[int, int],
        split: str = "train",
    ) -> None:
        """Add a puzzle piece to the metadata file.

        Args:
            piece_id: Unique ID for the piece
            puzzle_id: ID of the source puzzle
            filename: Relative path to the piece image
            bbox: Original bounding box (x1, y1, x2, y2)
            rotation: Rotation value (0, 90, 180, 270)
            puzzle_size: Size of the standardized puzzle (width, height)
            split: Dataset split ('train' or 'validation')
        """
        # Calculate normalized coordinates
        x1, y1, x2, y2 = bbox
        width, height = puzzle_size

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

    def standardize_puzzle(self, image: Image.Image) -> Image.Image:
        """Resize a puzzle image to standard dimensions.

        Args:
            image: Input puzzle image

        Returns:
            Standardized puzzle image
        """
        width, height = self.options.puzzle_size
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

        # Resize the piece to the target size
        target_width, target_height = self.options.piece_size
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
        # Create a binary mask for this piece
        piece_mask = (mask == piece_id).astype(np.uint8) * 255
        piece_mask_img = Image.fromarray(piece_mask)

        # Find bounding box
        bbox = piece_mask_img.getbbox()
        if bbox is None:
            return None

        # Extract piece
        x1, y1, x2, y2 = bbox
        piece_img = Image.new("RGBA", (x2 - x1, y2 - y1), (0, 0, 0, 0))

        # Extract pixel data with proper alpha channel
        for y in range(y1, y2):
            for x in range(x1, x2):
                if mask[y, x] == piece_id:
                    pixel = image.getpixel((x, y))
                    if isinstance(pixel, tuple):
                        r, g, b = pixel[:3]
                    elif pixel is not None:
                        r = g = b = int(pixel)
                    else:
                        r = g = b = 0
                    piece_img.putpixel((x - x1, y - y1), (r, g, b, 255))
                else:
                    piece_img.putpixel((x - x1, y - y1), (0, 0, 0, 0))

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


def process_directory(
    input_dir: str, output_dir: str, options: ProcessingOptions
) -> int:
    """Process all puzzle images in a directory.

    Args:
        input_dir: Input directory containing puzzle images
        output_dir: Output directory for processed data
        options: Processing options

    Returns:
        Total number of pieces generated
    """
    # Setup the metadata handler
    metadata_path = os.path.join(output_dir, "metadata.csv")
    metadata_handler = MetadataHandler(metadata_path, create_new=True)

    # Initialize the processor
    processor = PuzzleProcessor(options)

    # Keep track of total pieces
    total_pieces = 0

    # Process each puzzle image
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        puzzle_path = os.path.join(input_dir, filename)
        pieces_count = processor.process_puzzle(
            puzzle_path, output_dir, metadata_handler
        )

        print(f"Generated {pieces_count} pieces from {filename}")
        total_pieces += pieces_count

    # Create the train/val split files
    metadata_handler.create_split_files(output_dir)

    return total_pieces


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

    # Load and process image
    std_image = _process_puzzle_image(
        processor, puzzle_path, dirs["puzzles"], puzzle_name
    )

    # Generate mask
    width, height = std_image.size
    mask = processor.generate_mask(width, height)

    # Process pieces
    processed_count = 0
    unique_ids = np.unique(mask)

    for piece_id in unique_ids[1:]:  # Skip background (0)
        processed_count += process_single_piece(
            processor,
            std_image,
            mask,
            piece_id,
            dirs["pieces"],
            puzzle_id,
            metadata_handler,
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
    """Load and process the puzzle image.

    Args:
        processor: The PuzzleProcessor instance
        puzzle_path: Path to the puzzle image
        puzzles_dir: Directory to save processed puzzles
        puzzle_name: Name of the puzzle

    Returns:
        Standardized puzzle image
    """
    # Load the image
    original_image = Image.open(puzzle_path).convert("RGB")
    std_image = processor.standardize_puzzle(original_image)

    # Save the standardized puzzle
    std_puzzle_path = os.path.join(puzzles_dir, f"{puzzle_name}.jpg")
    std_image.save(std_puzzle_path)

    return std_image


def process_single_piece(
    processor, std_image, mask, piece_id, pieces_dir, puzzle_id, metadata_handler
):
    """Process a single puzzle piece to reduce complexity.

    Args:
        processor: The PuzzleProcessor instance
        std_image: Standardized puzzle image
        mask: Piece mask array
        piece_id: ID of the piece to extract
        pieces_dir: Directory to save piece images
        puzzle_id: ID of the source puzzle
        metadata_handler: Handler for piece metadata

    Returns:
        1 if piece was processed, 0 otherwise
    """
    # Extract the piece
    piece_data = processor.extract_piece(std_image, mask, piece_id)
    if piece_data is None:
        return 0

    # Process piece and save
    piece_info = _create_piece_info(piece_data, puzzle_id, piece_id, processor.options)
    _save_and_record_piece(processor, piece_info, pieces_dir, metadata_handler)

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
        "split": split,
        "filename": f"{puzzle_id}_{piece_id:03d}_r{rotation}.png",
    }


def _save_and_record_piece(processor, piece_info, pieces_dir, metadata_handler):
    """Save processed piece and record its metadata.

    Args:
        processor: The PuzzleProcessor instance
        piece_info: Dictionary with piece information
        pieces_dir: Directory to save piece images
        metadata_handler: Handler for piece metadata
    """
    # Process the piece for the model
    processed_piece = processor.process_piece(
        piece_info["piece"], piece_info["bbox"], piece_info["rotation"]
    )

    # Save the processed piece
    piece_path = os.path.join(pieces_dir, piece_info["filename"])
    processed_piece.save(piece_path)

    # Add to metadata
    metadata_handler.add_piece(
        piece_id=piece_info["id"],
        puzzle_id=piece_info["id"].split("_")[0],
        filename=os.path.join("pieces", piece_info["filename"]),
        bbox=piece_info["bbox"],
        rotation=piece_info["rotation"],
        puzzle_size=processor.options.puzzle_size,
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
        default=500,
        help="Approximate number of pieces to generate per puzzle",
    )
    parser.add_argument(
        "--puzzle-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="Size to resize puzzles to",
    )
    parser.add_argument(
        "--piece-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("WIDTH", "HEIGHT"),
        help="Size to resize pieces to",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of pieces to use for validation",
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
    _process_input(args.input_path, args.output_dir, options)


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
        puzzle_size=tuple(args.puzzle_size),
        piece_size=tuple(args.piece_size),
        validation_split=args.validation_split,
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


def _process_input(input_path, output_dir_path, options):
    """Process the input image or directory.

    Args:
        input_path: Path to the input image or directory
        output_dir_path: Path to the output directory (None to use input directory)
        options: Processing options
    """
    # Process the input path first to determine default output
    input_path = os.path.abspath(input_path)

    # Default output to input directory (or parent directory for single files)
    if output_dir_path is None:
        if os.path.isdir(input_path):
            output_dir = input_path
        else:
            output_dir = os.path.dirname(input_path)
    else:
        output_dir = os.path.abspath(output_dir_path)

    os.makedirs(output_dir, exist_ok=True)

    # Process the input
    if os.path.isdir(input_path):
        total_pieces = process_directory(input_path, output_dir, options)
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
