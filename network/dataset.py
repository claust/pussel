#!/usr/bin/env python
"""Dataset handling for puzzle piece prediction model."""

import os
from typing import Dict, Optional, Tuple

# Type ignore for missing stubs
import albumentations as A  # type: ignore
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class PuzzleDataset(Dataset):
    """Dataset for puzzle piece position and rotation prediction."""

    def __init__(
        self,
        root_dir: str,
        metadata_path: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
    ):
        """Initialize the puzzle dataset.

        Args:
            root_dir: Root directory containing the puzzle data
            metadata_path: Path to metadata CSV file
            split: Dataset split ('train' or 'validation')
            transform: Optional albumentations transforms
        """
        self.root_dir = root_dir
        self.transform = transform

        # Load metadata from CSV file
        self.metadata = pd.read_csv(metadata_path)

        # Filter by split
        self.metadata = self.metadata[self.metadata["split"] == split]
        self.metadata.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        """Return the dataset size."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single dataset item.

        Args:
            idx: Index of the item to retrieve

        Returns:
            Dictionary containing:
                - piece: Puzzle piece image tensor (C, H, W)
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

        # Convert PIL image to numpy for albumentations
        piece_array = np.array(piece_image)

        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=piece_array)
            piece_array = transformed["image"]

        # Convert to PyTorch tensor and normalize to [0, 1]
        piece_tensor = torch.from_numpy(piece_array).permute(2, 0, 1).float() / 255.0

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
            "position": position,
            "rotation": rotation_class,
            "puzzle_id": piece_data["puzzle_id"],
            "piece_id": piece_data["piece_id"],
        }


class PuzzleDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for puzzle piece dataset."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        input_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize the puzzle data module.

        Args:
            data_dir: Directory containing puzzle data
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            input_size: Target size for piece images
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        # Set file paths
        self.metadata_path = os.path.join(data_dir, "metadata.csv")

        # Define transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()

        # Initialize dataset attributes with Optional type annotation
        self.train_dataset: Optional[PuzzleDataset] = None
        self.val_dataset: Optional[PuzzleDataset] = None

    def _get_train_transform(self) -> A.Compose:
        """Get training data augmentation transforms.

        Returns:
            Albumentations composition of transforms
        """
        return A.Compose(
            [
                A.Resize(height=self.input_size[0], width=self.input_size[1]),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _get_val_transform(self) -> A.Compose:
        """Get validation data transforms.

        Returns:
            Albumentations composition of transforms
        """
        return A.Compose(
            [
                A.Resize(height=self.input_size[0], width=self.input_size[1]),
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
                transform=self.train_transform,
            )
            self.val_dataset = PuzzleDataset(
                self.data_dir,
                self.metadata_path,
                split="validation",
                transform=self.val_transform,
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
