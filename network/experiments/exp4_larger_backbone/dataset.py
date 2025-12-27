"""Dataset loader for larger backbone cell classification experiment.

Loads all pieces from puzzle_001 and returns cell indices as targets.
Cell index = row * num_cols + col, where row and col are computed from
the piece's normalized center coordinates.

This is identical to exp3's dataset - reused for consistency.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths relative to network/ directory
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets"
DEFAULT_PUZZLE_ID = "puzzle_001"

# Grid dimensions for puzzle_001
NUM_COLS = 38
NUM_ROWS = 25
NUM_CELLS = NUM_COLS * NUM_ROWS  # 950


class CellClassificationDataset(Dataset):
    """Dataset for cell classification experiment.

    Loads all pieces from a single puzzle and returns:
    - piece_image: Resized piece tensor (C, H, W)
    - cell_index: Integer cell index (0 to NUM_CELLS-1)

    The cell index is computed from the piece's center coordinates:
    col = int(cx * num_cols), row = int(cy * num_rows)
    cell_index = row * num_cols + col
    """

    def __init__(
        self,
        puzzle_id: str = DEFAULT_PUZZLE_ID,
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        piece_size: int = 64,
        puzzle_size: int = 256,
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
        print(f"Loaded {len(self.samples)} pieces for {puzzle_id}")
        print(f"Grid: {num_cols} cols x {num_rows} rows = {self.num_cells} cells")

        # Puzzle tensor loaded lazily
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

                    # Compute cell index
                    # Clamp to valid range to handle edge cases
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return piece image and cell index.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (piece_tensor, cell_index_tensor).
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

        return piece_tensor, cell_index

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample (useful for visualization)."""
        return self.samples[idx]

    def get_puzzle_image(self) -> torch.Tensor:
        """Get the puzzle image tensor (loads if not cached)."""
        if self.puzzle_tensor is None:
            self.puzzle_tensor = self._load_puzzle_image()
        return self.puzzle_tensor.clone()

    def cell_index_to_coords(self, cell_index: int) -> tuple[int, int]:
        """Convert cell index to (col, row) coordinates."""
        row = cell_index // self.num_cols
        col = cell_index % self.num_cols
        return col, row

    def cell_index_to_center(self, cell_index: int) -> tuple[float, float]:
        """Convert cell index to normalized center (cx, cy)."""
        col, row = self.cell_index_to_coords(cell_index)
        cx = (col + 0.5) / self.num_cols
        cy = (row + 0.5) / self.num_rows
        return cx, cy


def verify_cell_indices(dataset: CellClassificationDataset) -> dict:
    """Verify that cell indices are unique and cover all cells.

    Args:
        dataset: The dataset to verify.

    Returns:
        Dictionary with verification results.
    """
    cell_indices = [s["cell_index"] for s in dataset.samples]
    unique_indices = set(cell_indices)

    return {
        "num_samples": len(dataset),
        "unique_cells": len(unique_indices),
        "expected_cells": dataset.num_cells,
        "all_unique": len(cell_indices) == len(unique_indices),
        "all_covered": len(unique_indices) == dataset.num_cells,
        "min_index": min(cell_indices),
        "max_index": max(cell_indices),
    }


if __name__ == "__main__":
    # Quick test
    print("Testing CellClassificationDataset...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")

    dataset = CellClassificationDataset(
        puzzle_id="puzzle_001",
        piece_size=64,
        puzzle_size=256,
        unrotate_pieces=True,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of cells: {dataset.num_cells}")

    # Verify cell indices
    verification = verify_cell_indices(dataset)
    print("\nCell index verification:")
    for key, value in verification.items():
        print(f"  {key}: {value}")

    # Test a few samples
    print("\nSample outputs:")
    for i in [0, 100, 500]:
        if i < len(dataset):
            piece, cell_index = dataset[i]
            info = dataset.get_sample_info(i)
            col, row = dataset.cell_index_to_coords(int(cell_index.item()))
            print(f"  [{i}] piece shape: {piece.shape}")
            print(f"       cell_index: {cell_index.item()}, col: {col}, row: {row}")
            print(f"       center: ({info['cx']:.4f}, {info['cy']:.4f})")
            print(f"       piece_id: {info['piece_id']}")
