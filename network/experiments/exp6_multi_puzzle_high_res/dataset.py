"""Dataset loader for multi-puzzle high-resolution experiment.

This module provides datasets for training on multiple puzzles simultaneously
with higher resolution puzzle images to enable cross-puzzle generalization.

Key differences from exp5:
1. Supports loading multiple puzzles into a single dataset
2. Higher default puzzle resolution (512x512 vs 256x256)
3. Each sample includes which puzzle it came from for proper pairing
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths relative to network/ directory
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets"

# Grid dimensions (same for all puzzles in our dataset)
NUM_COLS = 38
NUM_ROWS = 25
NUM_CELLS = NUM_COLS * NUM_ROWS  # 950


class SinglePuzzleDataset(Dataset):
    """Dataset for a single puzzle's pieces with puzzle context.

    Loads all pieces from one puzzle and returns:
    - piece_image: Resized piece tensor (C, H, W)
    - puzzle_image: Resized puzzle tensor (C, H, W)
    - cell_index: Integer cell index (0 to NUM_CELLS-1)
    """

    def __init__(
        self,
        puzzle_id: str,
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        piece_size: int = 64,
        puzzle_size: int = 512,
        unrotate_pieces: bool = True,
        num_cols: int = NUM_COLS,
        num_rows: int = NUM_ROWS,
    ):
        """Initialize the dataset.

        Args:
            puzzle_id: ID of the puzzle to load (e.g., "puzzle_001").
            dataset_root: Root directory containing puzzles/, pieces/, metadata.csv.
            piece_size: Size to resize piece images to (square).
            puzzle_size: Size to resize puzzle image to (square).
            unrotate_pieces: If True, rotate pieces back to original orientation.
            num_cols: Number of columns in the puzzle grid.
            num_rows: Number of rows in the puzzle grid.
        """
        self.puzzle_id = puzzle_id
        self.dataset_root = Path(dataset_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.unrotate_pieces = unrotate_pieces
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_cells = num_cols * num_rows

        # Load metadata for this puzzle
        self.samples = self._load_metadata()

        # Pre-load and cache puzzle tensor for efficiency
        self.puzzle_tensor = self._load_puzzle_image()

        # Define transforms
        self.piece_transform = transforms.Compose(
            [
                transforms.Resize((piece_size, piece_size)),
                transforms.ToTensor(),
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

                    # Compute cell index
                    col = min(int(cx * self.num_cols), self.num_cols - 1)
                    row_idx = min(int(cy * self.num_rows), self.num_rows - 1)
                    cell_index = row_idx * self.num_cols + col

                    samples.append(
                        {
                            "piece_id": row["piece_id"],
                            "filename": row["filename"],
                            "rotation": int(row["rotation"]),
                            "cx": cx,
                            "cy": cy,
                            "col": col,
                            "row": row_idx,
                            "cell_index": cell_index,
                            "puzzle_id": self.puzzle_id,
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
        """Rotate image back to original orientation."""
        if rotation == 0:
            return img
        return img.rotate(rotation, expand=False)

    def __len__(self) -> int:
        """Return the number of pieces in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return piece image, puzzle image, and cell index.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (piece_tensor, puzzle_tensor, cell_index_tensor).
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

        # Target is the cell index
        cell_index = torch.tensor(sample["cell_index"], dtype=torch.long)

        # Return piece, puzzle (cached), and target
        return piece_tensor, self.puzzle_tensor.clone(), cell_index

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample (useful for visualization)."""
        return self.samples[idx]

    def get_puzzle_image(self) -> torch.Tensor:
        """Get the puzzle image tensor."""
        return self.puzzle_tensor.clone()

    def cell_index_to_coords(self, cell_index: int) -> tuple[int, int]:
        """Convert cell index to (col, row) coordinates."""
        row = cell_index // self.num_cols
        col = cell_index % self.num_cols
        return col, row


class MultiPuzzleDataset(Dataset):
    """Dataset that combines pieces from multiple puzzles.

    Each sample includes the piece image paired with its corresponding puzzle image,
    enabling the model to learn a matching function across different puzzles.
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        piece_size: int = 64,
        puzzle_size: int = 512,
        unrotate_pieces: bool = True,
        num_cols: int = NUM_COLS,
        num_rows: int = NUM_ROWS,
    ):
        """Initialize the multi-puzzle dataset.

        Args:
            puzzle_ids: List of puzzle IDs to include (e.g., ["puzzle_001", "puzzle_002"]).
            dataset_root: Root directory containing puzzles/, pieces/, metadata.csv.
            piece_size: Size to resize piece images to (square).
            puzzle_size: Size to resize puzzle image to (square).
            unrotate_pieces: If True, rotate pieces back to original orientation.
            num_cols: Number of columns in the puzzle grid.
            num_rows: Number of rows in the puzzle grid.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.num_cells = num_cols * num_rows

        # Create individual datasets for each puzzle
        self.puzzle_datasets: list[SinglePuzzleDataset] = []
        self.puzzle_tensors: dict[str, torch.Tensor] = {}

        print(f"Loading {len(puzzle_ids)} puzzles with {puzzle_size}x{puzzle_size} resolution...")

        for puzzle_id in puzzle_ids:
            dataset = SinglePuzzleDataset(
                puzzle_id=puzzle_id,
                dataset_root=dataset_root,
                piece_size=piece_size,
                puzzle_size=puzzle_size,
                unrotate_pieces=unrotate_pieces,
                num_cols=num_cols,
                num_rows=num_rows,
            )
            self.puzzle_datasets.append(dataset)
            self.puzzle_tensors[puzzle_id] = dataset.get_puzzle_image()
            print(f"  {puzzle_id}: {len(dataset)} pieces")

        # Build combined sample list with puzzle references
        self.samples: list[tuple[int, int]] = []  # (puzzle_idx, sample_idx)
        for puzzle_idx, dataset in enumerate(self.puzzle_datasets):
            for sample_idx in range(len(dataset)):
                self.samples.append((puzzle_idx, sample_idx))

        print(f"Total: {len(self.samples)} pieces from {len(puzzle_ids)} puzzles")

    def __len__(self) -> int:
        """Return total number of pieces across all puzzles."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return piece image, its corresponding puzzle image, and cell index.

        Args:
            idx: Index into the combined sample list.

        Returns:
            Tuple of (piece_tensor, puzzle_tensor, cell_index_tensor).
        """
        puzzle_idx, sample_idx = self.samples[idx]
        return self.puzzle_datasets[puzzle_idx][sample_idx]

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample."""
        puzzle_idx, sample_idx = self.samples[idx]
        return self.puzzle_datasets[puzzle_idx].get_sample_info(sample_idx)

    def get_puzzle_image(self, puzzle_id: str) -> torch.Tensor:
        """Get a specific puzzle's image tensor."""
        return self.puzzle_tensors[puzzle_id].clone()


def create_train_test_datasets(
    train_puzzle_ids: list[str],
    test_puzzle_id: str,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    piece_size: int = 64,
    puzzle_size: int = 512,
) -> tuple[MultiPuzzleDataset, SinglePuzzleDataset]:
    """Create training and test datasets.

    Args:
        train_puzzle_ids: List of puzzle IDs for training.
        test_puzzle_id: Puzzle ID for testing (held out).
        dataset_root: Root directory of the dataset.
        piece_size: Size to resize piece images.
        puzzle_size: Size to resize puzzle images.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    train_dataset = MultiPuzzleDataset(
        puzzle_ids=train_puzzle_ids,
        dataset_root=dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    test_dataset = SinglePuzzleDataset(
        puzzle_id=test_puzzle_id,
        dataset_root=dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    print("Testing MultiPuzzleDataset...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")

    # Test multi-puzzle loading
    train_puzzles = ["puzzle_001", "puzzle_002", "puzzle_003", "puzzle_004", "puzzle_005"]
    test_puzzle = "puzzle_006"

    print(f"\nTraining puzzles: {train_puzzles}")
    print(f"Test puzzle: {test_puzzle}")

    train_dataset, test_dataset = create_train_test_datasets(
        train_puzzle_ids=train_puzzles,
        test_puzzle_id=test_puzzle,
        piece_size=64,
        puzzle_size=512,
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Test a sample
    piece, puzzle, cell_index = train_dataset[0]
    info = train_dataset.get_sample_info(0)
    print(f"\nSample 0 from {info['puzzle_id']}:")
    print(f"  Piece shape: {piece.shape}")
    print(f"  Puzzle shape: {puzzle.shape}")
    print(f"  Cell index: {cell_index.item()}")

    # Test from different puzzle
    piece2, puzzle2, cell_index2 = train_dataset[1000]
    info2 = train_dataset.get_sample_info(1000)
    print(f"\nSample 1000 from {info2['puzzle_id']}:")
    print(f"  Piece shape: {piece2.shape}")
    print(f"  Puzzle shape: {puzzle2.shape}")
    print(f"  Cell index: {cell_index2.item()}")

    # Verify puzzles are different
    print(f"\nPuzzle tensors are different: {not torch.equal(puzzle, puzzle2)}")
