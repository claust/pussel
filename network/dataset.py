#!/usr/bin/env python
"""Dataset handling for puzzle piece prediction model."""

import logging
import os
from typing import Dict, Optional, Set, Tuple

# Type ignore for missing stubs
import albumentations as A  # type: ignore
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class PuzzleDataset(Dataset):
    """Dataset for puzzle piece and full puzzle image pairs."""

    def __init__(
        self,
        root_dir: str,
        metadata_path: str,
        split: str = "train",
        piece_transform: Optional[A.Compose] = None,
        puzzle_transform: Optional[A.Compose] = None,
        prepare_puzzles: bool = True,
    ):
        """Initialize the dual-input puzzle dataset.

        Args:
            root_dir: Root directory containing the puzzle data
            metadata_path: Path to metadata CSV file
            split: Dataset split ('train' or 'validation')
            piece_transform: Optional transforms for puzzle pieces
            puzzle_transform: Optional transforms for full puzzle images
            prepare_puzzles: Whether to create placeholder puzzle images if missing
        """
        self.root_dir = root_dir
        self.piece_transform = piece_transform
        self.puzzle_transform = puzzle_transform

        # Load metadata from CSV file
        self.metadata = pd.read_csv(metadata_path)

        # Filter by split
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.metadata.reset_index(drop=True, inplace=True)

        # Prepare puzzle directory
        self.puzzles_dir = os.path.join(root_dir, "puzzles")
        if not os.path.exists(self.puzzles_dir):
            os.makedirs(self.puzzles_dir, exist_ok=True)

        # Get unique puzzle IDs
        self.puzzle_ids = set(self.metadata["puzzle_id"].unique())

        # Cache puzzle paths for faster loading
        self.puzzles = {}
        missing_puzzles = set()

        for puzzle_id in self.puzzle_ids:
            puzzle_path = os.path.join(self.puzzles_dir, f"{puzzle_id}.jpg")

            if os.path.exists(puzzle_path):
                self.puzzles[puzzle_id] = puzzle_path
            else:
                # Check alternate locations
                alt_path = os.path.join(root_dir, f"puzzle_{puzzle_id}.jpg")
                if os.path.exists(alt_path):
                    self.puzzles[puzzle_id] = alt_path
                else:
                    missing_puzzles.add(puzzle_id)

        # Extract full puzzles if needed and requested
        if missing_puzzles and prepare_puzzles:
            print(f"Creating {len(missing_puzzles)} missing puzzle images...")
            self._create_missing_puzzles(missing_puzzles)

        # Final check of missing puzzles
        self.missing_puzzles = set()
        for puzzle_id in self.puzzle_ids:
            if puzzle_id not in self.puzzles:
                self.missing_puzzles.add(puzzle_id)

        if self.missing_puzzles:
            count = len(self.missing_puzzles)
            print(f"Warning: {count} puzzles still missing. Using placeholders.")
        else:
            print(f"All {len(self.puzzle_ids)} puzzles found or created!")

        # Map to track which placeholders we've already warned about
        self.warned_placeholders: Set[str] = set()

    def _create_missing_puzzles(self, missing_puzzles: Set[str]) -> None:
        """Create placeholder puzzle images for missing puzzles.

        Args:
            missing_puzzles: Set of puzzle IDs that need placeholders
        """
        for puzzle_id in missing_puzzles:
            # Get all pieces for this puzzle
            pieces_data = self.metadata[self.metadata["puzzle_id"] == puzzle_id]

            if len(pieces_data) == 0:
                continue

            # Get the first piece to determine size
            first_piece = pieces_data.iloc[0]
            piece_path = os.path.join(self.root_dir, first_piece["filename"])

            try:
                piece_img = Image.open(piece_path)
                # Create a blank puzzle image
                puzzle_size = (
                    piece_img.width * 4,
                    piece_img.height * 4,
                )  # Estimate size
                puzzle_img = Image.new("RGB", puzzle_size, color=(240, 240, 240))

                # Save the puzzle image
                output_path = os.path.join(self.puzzles_dir, f"{puzzle_id}.jpg")
                puzzle_img.save(output_path)

                # Add to the puzzles dictionary
                self.puzzles[puzzle_id] = output_path
                print(f"Created placeholder for puzzle {puzzle_id}")
            except Exception as e:
                print(f"Error creating placeholder for puzzle {puzzle_id}: {e}")

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset item with both piece and puzzle.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dictionary containing:
                - piece: Puzzle piece image tensor (C, H, W)
                - puzzle: Full puzzle image tensor (C, H, W)
                - position: Position target tensor (x1, y1, x2, y2) normalized
                - rotation: Rotation class (0, 1, 2, 3) for 0°, 90°, 180°, 270°
                - puzzle_id: ID of the source puzzle
                - piece_id: ID of the piece
        """
        # Get metadata for this piece
        piece_data = self.metadata.iloc[idx]

        # Load the piece image
        piece_path = os.path.join(self.root_dir, piece_data["filename"])
        piece_image = Image.open(piece_path).convert("RGB")
        piece_array = np.array(piece_image)

        # Load the puzzle image
        puzzle_id = piece_data["puzzle_id"]
        if puzzle_id in self.puzzles:
            puzzle_path = self.puzzles[puzzle_id]
            puzzle_image = Image.open(puzzle_path).convert("RGB")
            puzzle_array = np.array(puzzle_image)
        else:
            # If puzzle image not found, use a placeholder with same size as piece
            puzzle_array = np.ones_like(piece_array) * 240  # Light gray background

            # Only warn once per puzzle ID to reduce console spam
            if puzzle_id not in self.warned_placeholders:
                self.warned_placeholders.add(puzzle_id)
                print(f"Using placeholder for puzzle {puzzle_id}")

        # Apply transforms if provided
        if self.piece_transform:
            transformed_piece = self.piece_transform(image=piece_array)
            piece_array = transformed_piece["image"]

        if self.puzzle_transform:
            transformed_puzzle = self.puzzle_transform(image=puzzle_array)
            puzzle_array = transformed_puzzle["image"]

        # Convert to PyTorch tensors
        piece_tensor = torch.from_numpy(piece_array).permute(2, 0, 1).float() / 255.0
        puzzle_tensor = torch.from_numpy(puzzle_array).permute(2, 0, 1).float() / 255.0

        # Get position target (normalized coordinates)
        position = torch.tensor(
            [
                piece_data["normalized_x1"],
                piece_data["normalized_y1"],
                piece_data["normalized_x2"],
                piece_data["normalized_y2"],
            ],
            dtype=torch.float32,
        )

        # Get rotation target (convert to class index)
        rotation_degrees = piece_data["rotation"]
        rotation_class = torch.tensor(rotation_degrees // 90, dtype=torch.long)

        return {
            "piece": piece_tensor,
            "puzzle": puzzle_tensor,
            "position": position,
            "rotation": rotation_class,
            "puzzle_id": piece_data["puzzle_id"],
            "piece_id": piece_data["piece_id"],
        }


class PuzzleDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for dual-input puzzle dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        piece_size: Tuple[int, int] = (224, 224),
        puzzle_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize the puzzle data module.

        Args:
            data_dir: Directory containing puzzle data
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            piece_size: Target size for piece images
            puzzle_size: Target size for full puzzle images
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size

        # Set file paths
        self.metadata_path = os.path.join(data_dir, "metadata.csv")

        # Define transforms
        self.train_piece_transform = self._get_train_piece_transform()
        self.train_puzzle_transform = self._get_train_puzzle_transform()
        self.val_piece_transform = self._get_val_piece_transform()
        self.val_puzzle_transform = self._get_val_puzzle_transform()

        # Initialize dataset attributes with Optional type annotation
        self.train_dataset: Optional[PuzzleDataset] = None
        self.val_dataset: Optional[PuzzleDataset] = None

    def _get_train_piece_transform(self) -> A.Compose:
        """Get training data augmentation transforms for pieces.

        Returns:
            Albumentations composition of transforms
        """
        return A.Compose(
            [
                A.Resize(height=self.piece_size[0], width=self.piece_size[1]),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_train_puzzle_transform(self) -> A.Compose:
        """Get training data augmentation transforms for puzzles.

        Returns:
            Albumentations composition of transforms
        """
        return A.Compose(
            [
                A.Resize(height=self.puzzle_size[0], width=self.puzzle_size[1]),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_val_piece_transform(self) -> A.Compose:
        """Get validation data transforms for pieces.

        Returns:
            Albumentations composition of transforms
        """
        return A.Compose(
            [
                A.Resize(height=self.piece_size[0], width=self.piece_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_val_puzzle_transform(self) -> A.Compose:
        """Get validation data transforms for puzzles.

        Returns:
            Albumentations composition of transforms
        """
        return A.Compose(
            [
                A.Resize(height=self.puzzle_size[0], width=self.puzzle_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets based on stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PuzzleDataset(
                self.data_dir,
                self.metadata_path,
                split="train",
                piece_transform=self.train_piece_transform,
                puzzle_transform=self.train_puzzle_transform,
                prepare_puzzles=False,  # Never prepare puzzles in training
            )
            self.val_dataset = PuzzleDataset(
                self.data_dir,
                self.metadata_path,
                split="validation",
                piece_transform=self.val_piece_transform,
                puzzle_transform=self.val_puzzle_transform,
                prepare_puzzles=False,  # Never prepare puzzles in training
            )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            Training dataloader
        """
        if self.train_dataset is None:
            raise ValueError("train_dataset is not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            Validation dataloader
        """
        if self.val_dataset is None:
            raise ValueError("val_dataset is not initialized. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == "__main__":
    # Test dataset functionality
    import matplotlib.pyplot as plt

    # Set up test dataset
    test_data_dir = "datasets/example/processed"
    dataset = PuzzleDataset(
        test_data_dir, os.path.join(test_data_dir, "metadata.csv"), split="train"
    )

    # Display sample
    sample = dataset[0]

    # Convert tensor to numpy for display
    display_img = sample["piece"].permute(1, 2, 0).numpy()

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(display_img)
    plt.title(f"Piece: {sample['piece_id']}, Rotation: {sample['rotation'] * 90}°")
    plt.xlabel(f"Position: {sample['position'].numpy()}")
    plt.show()
