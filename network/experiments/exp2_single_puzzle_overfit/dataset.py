"""Dataset loader for single puzzle overfit experiment.

Loads all pieces from puzzle_001 and their target center coordinates.
This is a pure overfit test - we train and evaluate on the same data.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths relative to network/ directory
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets"
DEFAULT_PUZZLE_ID = "puzzle_001"


class SinglePuzzleDataset(Dataset):
    """Dataset for single puzzle overfit experiment.

    Loads all pieces from a single puzzle and returns:
    - piece_image: Resized piece tensor (C, H, W)
    - target: Normalized center coordinates (cx, cy) in [0, 1]

    Optionally also returns the puzzle image for dual-encoder architectures.
    """

    def __init__(
        self,
        puzzle_id: str = DEFAULT_PUZZLE_ID,
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        piece_size: int = 64,
        puzzle_size: int = 256,
        unrotate_pieces: bool = True,
    ):
        """Initialize the dataset.

        Args:
            puzzle_id: ID of the puzzle to load (e.g., "puzzle_001").
            dataset_root: Root directory containing puzzles/, pieces/, metadata.csv.
            piece_size: Size to resize piece images to (square).
            puzzle_size: Size to resize puzzle image to (square).
            unrotate_pieces: If True, rotate pieces back to original orientation.
        """
        self.puzzle_id = puzzle_id
        self.dataset_root = Path(dataset_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.unrotate_pieces = unrotate_pieces

        # Load metadata for this puzzle
        self.samples = self._load_metadata()
        print(f"Loaded {len(self.samples)} pieces for {puzzle_id}")

        # Puzzle tensor loaded lazily via get_puzzle_image() or get_with_puzzle()
        self.puzzle_tensor: torch.Tensor | None = None

        # Define transforms
        self.piece_transform = transforms.Compose(
            [
                transforms.Resize((piece_size, piece_size)),
                transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
            ]
        )

    def _load_metadata(self) -> list[dict]:
        """Load metadata entries for the specified puzzle."""
        metadata_path = self.dataset_root / "metadata.csv"
        samples = []

        with open(metadata_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["puzzle_id"] == self.puzzle_id:
                    # Compute center from normalized coordinates
                    nx1 = float(row["normalized_x1"])
                    ny1 = float(row["normalized_y1"])
                    nx2 = float(row["normalized_x2"])
                    ny2 = float(row["normalized_y2"])

                    cx = (nx1 + nx2) / 2
                    cy = (ny1 + ny2) / 2

                    samples.append(
                        {
                            "piece_id": row["piece_id"],
                            "filename": row["filename"],
                            "rotation": int(row["rotation"]),
                            "cx": cx,
                            "cy": cy,
                            # Store bbox too for visualization
                            "nx1": nx1,
                            "ny1": ny1,
                            "nx2": nx2,
                            "ny2": ny2,
                        }
                    )

        return samples

    def _load_puzzle_image(self) -> torch.Tensor:
        """Load and preprocess the puzzle image."""
        puzzle_path = self.dataset_root / "puzzles" / f"{self.puzzle_id}.jpg"
        puzzle_img = Image.open(puzzle_path).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((self.puzzle_size, self.puzzle_size)),
                transforms.ToTensor(),
            ]
        )

        result = transform(puzzle_img)
        assert isinstance(result, torch.Tensor)
        return result

    def _unrotate_image(self, img: Image.Image, rotation: int) -> Image.Image:
        """Rotate image back to original orientation.

        The pieces are stored rotated by `rotation` degrees.
        To get the original orientation, we rotate by -rotation.
        PIL's rotate is counter-clockwise, so we use positive rotation value.
        """
        if rotation == 0:
            return img
        # PIL rotate is counter-clockwise, pieces were rotated clockwise
        # So to undo: rotate counter-clockwise by the same amount
        return img.rotate(rotation, expand=False)

    def __len__(self) -> int:
        """Return the number of pieces in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return piece image and target coordinates.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (piece_tensor, target_tensor).
            Use get_with_puzzle() if you need the puzzle image too.
        """
        sample = self.samples[idx]

        # Load piece image
        piece_path = self.dataset_root / sample["filename"]
        piece_img = Image.open(piece_path).convert("RGB")

        # Optionally unrotate
        if self.unrotate_pieces:
            piece_img = self._unrotate_image(piece_img, sample["rotation"])

        # Apply transforms
        piece_result = self.piece_transform(piece_img)
        assert isinstance(piece_result, torch.Tensor)
        piece_tensor: torch.Tensor = piece_result

        # Target coordinates
        target_tensor = torch.tensor([sample["cx"], sample["cy"]], dtype=torch.float32)

        return piece_tensor, target_tensor

    def get_with_puzzle(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return piece image, puzzle image, and target coordinates.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (piece_tensor, puzzle_tensor, target_tensor).
        """
        piece_tensor, target_tensor = self.__getitem__(idx)

        if self.puzzle_tensor is None:
            self.puzzle_tensor = self._load_puzzle_image()

        return piece_tensor, self.puzzle_tensor.clone(), target_tensor

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample (useful for visualization)."""
        return self.samples[idx]

    def get_puzzle_image(self) -> torch.Tensor:
        """Get the puzzle image tensor (loads if not cached)."""
        if self.puzzle_tensor is None:
            self.puzzle_tensor = self._load_puzzle_image()
        return self.puzzle_tensor.clone()


class SubsetSinglePuzzleDataset(Dataset):
    """Wrapper to create a subset of SinglePuzzleDataset for overfit tests."""

    def __init__(self, dataset: SinglePuzzleDataset, indices: list[int]):
        """Initialize with a subset of indices.

        Args:
            dataset: The full SinglePuzzleDataset.
            indices: List of indices to include in this subset.
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sample at the given index."""
        return self.dataset[self.indices[idx]]


def compute_dataset_statistics(dataset: SinglePuzzleDataset) -> dict:
    """Compute statistics about the dataset for debugging.

    Args:
        dataset: The dataset to analyze.

    Returns:
        Dictionary with statistics about target distribution.
    """
    cx_values = [s["cx"] for s in dataset.samples]
    cy_values = [s["cy"] for s in dataset.samples]

    return {
        "num_samples": len(dataset),
        "cx_min": min(cx_values),
        "cx_max": max(cx_values),
        "cx_mean": np.mean(cx_values),
        "cx_std": np.std(cx_values),
        "cy_min": min(cy_values),
        "cy_max": max(cy_values),
        "cy_mean": np.mean(cy_values),
        "cy_std": np.std(cy_values),
    }


if __name__ == "__main__":
    # Quick test
    print("Testing SinglePuzzleDataset...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")

    dataset = SinglePuzzleDataset(
        puzzle_id="puzzle_001",
        piece_size=64,
        puzzle_size=256,
        unrotate_pieces=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Print statistics
    stats = compute_dataset_statistics(dataset)
    print("\nDataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Test a few samples
    print("\nSample outputs:")
    for i in [0, 100, 500]:
        if i < len(dataset):
            piece, target = dataset[i]
            info = dataset.get_sample_info(i)
            print(f"  [{i}] piece shape: {piece.shape}, target: ({target[0]:.4f}, {target[1]:.4f})")
            print(f"       piece_id: {info['piece_id']}, rotation: {info['rotation']}")

    # Test get_with_puzzle method
    print("\nTesting get_with_puzzle method...")
    piece, puzzle, target = dataset.get_with_puzzle(0)
    print(f"  piece shape: {piece.shape}")
    print(f"  puzzle shape: {puzzle.shape}")
    print(f"  target: ({target[0]:.4f}, {target[1]:.4f})")
