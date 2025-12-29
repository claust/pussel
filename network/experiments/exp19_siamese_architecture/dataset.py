"""Dataset for 3x3 grid position + rotation prediction.

Exp18: Scales up training data from 10,000 to 20,000 puzzles to further improve generalization.
Based on exp17 architecture with ShuffleNetV2_x0.5 backbone.

Key features:
- 3x3 grid = 9 cells instead of 2x2 quadrants = 4 cells
- 20,000 training puzzles (2x exp17)
- Training: Random cell + random rotation sampling
- Testing: All 4 rotations = 36 samples per puzzle
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets"

# Grid configuration
GRID_SIZE = 3  # 3x3 grid
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 9 cells

# Cell centers (normalized coordinates) for 3x3 grid
# Cells are numbered row-major: 0,1,2 (top row), 3,4,5 (middle row), 6,7,8 (bottom row)
CELL_CENTERS = []
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        cx = (col + 0.5) / GRID_SIZE  # e.g., 0.167, 0.5, 0.833
        cy = (row + 0.5) / GRID_SIZE
        CELL_CENTERS.append((cx, cy))

# Rotation angles in degrees
ROTATION_ANGLES = [0, 90, 180, 270]


class GridRandomRotationDataset(Dataset):
    """Dataset that generates RANDOMLY rotated 3x3 grid pieces.

    Each sample consists of:
    - piece: One cell of a puzzle image (1/9th), randomly rotated
    - puzzle: The full puzzle image (not rotated)
    - target_position: Center coordinates (cx, cy) normalized to [0, 1]
    - cell_idx: Cell index (0-8)
    - rotation_idx: Rotation index (0=0deg, 1=90deg, 2=180deg, 3=270deg)
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
        augment: bool = False,
        random_rotation: bool = True,
        cells_per_puzzle: int = NUM_CELLS,
        seed: int = 42,
        preload_cache: bool = False,
    ):
        """Initialize the dataset with random rotation sampling.

        Args:
            puzzle_ids: List of puzzle IDs to include (e.g., ["puzzle_001"]).
            dataset_root: Root directory containing puzzles/ folder.
            piece_size: Size to resize piece images to (square).
            puzzle_size: Size to resize puzzle images to (square).
            augment: If True, apply data augmentation to pieces.
            random_rotation: If True, randomly sample rotation per access.
            cells_per_puzzle: Number of cells to sample per puzzle (1-9).
            seed: Random seed for cell selection.
            preload_cache: If True, pre-load all puzzle images into memory.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.augment = augment
        self.random_rotation = random_rotation
        self.cells_per_puzzle = min(cells_per_puzzle, NUM_CELLS)

        # Build sample list: (puzzle_id, cell_idx)
        # If cells_per_puzzle < NUM_CELLS, randomly select cells for each puzzle
        rng = random.Random(seed)
        self.samples: list[tuple[str, int]] = []
        for puzzle_id in puzzle_ids:
            if self.cells_per_puzzle >= NUM_CELLS:
                # Use all cells
                for cell_idx in range(NUM_CELLS):
                    self.samples.append((puzzle_id, cell_idx))
            else:
                # Randomly select cells for this puzzle
                selected_cells = rng.sample(range(NUM_CELLS), self.cells_per_puzzle)
                for cell_idx in selected_cells:
                    self.samples.append((puzzle_id, cell_idx))

        # Cache for loaded puzzle images (PIL Images)
        self._puzzle_cache: dict[str, Image.Image] = {}

        # Pre-load all puzzles into memory if requested
        if preload_cache:
            print(f"Pre-loading {len(puzzle_ids)} puzzles into memory...")
            for puzzle_id in puzzle_ids:
                self._load_puzzle(puzzle_id)
            print(f"Pre-loaded {len(self._puzzle_cache)} puzzles")

        # Transforms
        self.piece_transform = transforms.Compose(
            [
                transforms.Resize((piece_size, piece_size)),
                transforms.ToTensor(),
            ]
        )
        self.puzzle_transform = transforms.Compose(
            [
                transforms.Resize((puzzle_size, puzzle_size)),
                transforms.ToTensor(),
            ]
        )

        # Augmentation transforms for pieces
        if augment:
            self.piece_augment = transforms.Compose(
                [
                    # Color augmentations
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomGrayscale(p=0.1),
                    # Small geometric augmentations
                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    # Blur for robustness
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ]
            )
        else:
            self.piece_augment = None

        print(
            f"GridRandomRotationDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples ({self.cells_per_puzzle} cells/puzzle, random rotation)"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image, using cache if available."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.dataset_root / "puzzles" / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _extract_cell(self, puzzle_img: Image.Image, cell_idx: int) -> Image.Image:
        """Extract a cell from the puzzle image (3x3 grid).

        Args:
            puzzle_img: Full puzzle PIL Image.
            cell_idx: Cell index 0-8 (row-major order).

        Returns:
            Cropped cell as PIL Image.
        """
        w, h = puzzle_img.size
        cell_w, cell_h = w // GRID_SIZE, h // GRID_SIZE

        row = cell_idx // GRID_SIZE
        col = cell_idx % GRID_SIZE

        left = col * cell_w
        upper = row * cell_h
        right = left + cell_w
        lower = upper + cell_h

        return puzzle_img.crop((left, upper, right, lower))

    def _rotate_piece(self, piece_img: Image.Image, rotation_idx: int) -> Image.Image:
        """Rotate piece by the specified angle.

        Args:
            piece_img: Piece PIL Image.
            rotation_idx: 0=0deg, 1=90deg, 2=180deg, 3=270deg

        Returns:
            Rotated piece image.
        """
        angle = ROTATION_ANGLES[rotation_idx]
        if angle == 0:
            return piece_img
        # PIL rotates counter-clockwise, so negate for clockwise rotation
        return piece_img.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample with RANDOM rotation.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (piece_tensor, puzzle_tensor, target_coords, cell_idx,
            rotation_idx).
        """
        puzzle_id, cell_idx = self.samples[idx]

        # Randomly sample rotation
        if self.random_rotation:
            rotation_idx = random.randint(0, 3)
        else:
            rotation_idx = 0

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Extract cell as piece
        piece_img = self._extract_cell(puzzle_img, cell_idx)

        # Apply rotation to piece
        piece_img = self._rotate_piece(piece_img, rotation_idx)

        # Apply augmentation to piece if enabled
        if self.piece_augment is not None:
            piece_img = self.piece_augment(piece_img)

        # Apply transforms
        piece_result = self.piece_transform(piece_img)
        puzzle_result = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_result, torch.Tensor)
        assert isinstance(puzzle_result, torch.Tensor)

        # Target coordinates (center of cell)
        cx, cy = CELL_CENTERS[cell_idx]
        target = torch.tensor([cx, cy], dtype=torch.float32)

        return (
            piece_result,
            puzzle_result,
            target,
            torch.tensor(cell_idx),
            torch.tensor(rotation_idx),
        )

    def get_puzzle_tensor(self, puzzle_id: str) -> torch.Tensor:
        """Get transformed puzzle tensor for a specific puzzle."""
        puzzle_img = self._load_puzzle(puzzle_id)
        result = self.puzzle_transform(puzzle_img)
        assert isinstance(result, torch.Tensor)
        return result

    def clear_cache(self) -> None:
        """Clear the puzzle image cache to free memory."""
        self._puzzle_cache.clear()


class GridAllRotationsDataset(Dataset):
    """Dataset that generates ALL rotations for testing.

    For evaluation, we want to test all 4 rotations for each cell
    (deterministic). This ensures consistent metrics across evaluations.

    Each puzzle produces 36 samples: 9 cells x 4 rotations.
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
    ):
        """Initialize the dataset with all rotations.

        Args:
            puzzle_ids: List of puzzle IDs to include.
            dataset_root: Root directory containing puzzles/ folder.
            piece_size: Size to resize piece images to.
            puzzle_size: Size to resize puzzle images to.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size

        # Build sample list: (puzzle_id, cell_idx, rotation_idx)
        self.samples: list[tuple[str, int, int]] = []
        for puzzle_id in puzzle_ids:
            for cell_idx in range(NUM_CELLS):
                for rotation_idx in range(4):
                    self.samples.append((puzzle_id, cell_idx, rotation_idx))

        # Cache for loaded puzzle images
        self._puzzle_cache: dict[str, Image.Image] = {}

        # Transforms
        self.piece_transform = transforms.Compose(
            [
                transforms.Resize((piece_size, piece_size)),
                transforms.ToTensor(),
            ]
        )
        self.puzzle_transform = transforms.Compose(
            [
                transforms.Resize((puzzle_size, puzzle_size)),
                transforms.ToTensor(),
            ]
        )

        print(
            f"GridAllRotationsDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples ({NUM_CELLS * 4} per puzzle)"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image, using cache if available."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.dataset_root / "puzzles" / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _extract_cell(self, puzzle_img: Image.Image, cell_idx: int) -> Image.Image:
        """Extract a cell from the puzzle image (3x3 grid)."""
        w, h = puzzle_img.size
        cell_w, cell_h = w // GRID_SIZE, h // GRID_SIZE

        row = cell_idx // GRID_SIZE
        col = cell_idx % GRID_SIZE

        left = col * cell_w
        upper = row * cell_h
        right = left + cell_w
        lower = upper + cell_h

        return puzzle_img.crop((left, upper, right, lower))

    def _rotate_piece(self, piece_img: Image.Image, rotation_idx: int) -> Image.Image:
        """Rotate piece by the specified angle."""
        angle = ROTATION_ANGLES[rotation_idx]
        if angle == 0:
            return piece_img
        return piece_img.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (piece_tensor, puzzle_tensor, target_coords, cell_idx,
            rotation_idx).
        """
        puzzle_id, cell_idx, rotation_idx = self.samples[idx]

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Extract cell as piece
        piece_img = self._extract_cell(puzzle_img, cell_idx)

        # Apply rotation to piece
        piece_img = self._rotate_piece(piece_img, rotation_idx)

        # Apply transforms
        piece_result = self.piece_transform(piece_img)
        puzzle_result = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_result, torch.Tensor)
        assert isinstance(puzzle_result, torch.Tensor)

        # Target coordinates
        cx, cy = CELL_CENTERS[cell_idx]
        target = torch.tensor([cx, cy], dtype=torch.float32)

        return (
            piece_result,
            puzzle_result,
            target,
            torch.tensor(cell_idx),
            torch.tensor(rotation_idx),
        )

    def clear_cache(self) -> None:
        """Clear the puzzle image cache."""
        self._puzzle_cache.clear()


def get_puzzle_ids(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> list[str]:
    """Get list of all available puzzle IDs.

    Args:
        dataset_root: Root directory containing puzzles/ folder.

    Returns:
        Sorted list of puzzle IDs.
    """
    puzzles_dir = Path(dataset_root) / "puzzles"
    puzzle_files = sorted(puzzles_dir.glob("puzzle_*.jpg"))
    return [f.stem for f in puzzle_files]


def create_datasets(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    n_train_puzzles: int = 20000,
    n_test_puzzles: int = 200,
    piece_size: int = 128,
    puzzle_size: int = 256,
    cells_per_puzzle: int = NUM_CELLS,
    seed: int = 42,
    preload_cache: bool = False,
) -> tuple[GridRandomRotationDataset, GridAllRotationsDataset]:
    """Create train and test datasets with random/all rotations.

    Training: Random rotation per sample, configurable cells per puzzle
    Testing: All rotations (36 samples per puzzle) for complete evaluation

    Args:
        dataset_root: Root directory.
        n_train_puzzles: Number of puzzles for training (default: 20000).
        n_test_puzzles: Number of puzzles for testing.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        cells_per_puzzle: Number of cells to sample per puzzle for training (1-9).
        seed: Random seed.
        preload_cache: If True, pre-load all puzzle images into memory.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    all_puzzles = get_puzzle_ids(dataset_root)

    # Shuffle and split
    rng = random.Random(seed)
    shuffled = all_puzzles.copy()
    rng.shuffle(shuffled)

    # Limit to requested sizes
    n_train = min(n_train_puzzles, len(shuffled) - n_test_puzzles)
    n_test = min(n_test_puzzles, len(shuffled) - n_train)

    train_ids = sorted(shuffled[:n_train])
    test_ids = sorted(shuffled[n_train : n_train + n_test])

    cells = min(cells_per_puzzle, NUM_CELLS)
    print(f"\nDataset split (seed={seed}):")
    print(f"  Training puzzles: {len(train_ids)} ({len(train_ids) * cells} samples/epoch, {cells} cells/puzzle)")
    print(f"  Test puzzles: {len(test_ids)} ({len(test_ids) * NUM_CELLS * 4} samples)")

    train_dataset = GridRandomRotationDataset(
        puzzle_ids=train_ids,
        dataset_root=dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        augment=True,
        random_rotation=True,
        cells_per_puzzle=cells_per_puzzle,
        seed=seed,
        preload_cache=preload_cache,
    )

    test_dataset = GridAllRotationsDataset(
        puzzle_ids=test_ids,
        dataset_root=dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    print("Testing GridRandomRotationDataset for exp18 (3x3 grid, 20K puzzles)...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells")
    print(f"Cell centers: {CELL_CENTERS}")

    # Check available puzzles
    all_puzzles = get_puzzle_ids()
    print(f"\nTotal available puzzles: {len(all_puzzles)}")
    print(f"Sample puzzle IDs: {all_puzzles[:5]}")

    # Create datasets
    train_dataset, test_dataset = create_datasets(
        n_train_puzzles=100,
        n_test_puzzles=20,
        piece_size=128,
        puzzle_size=256,
    )

    # Test train dataset
    print("\n--- Training Dataset (Random Rotation) ---")
    piece, puzzle, target, cell, rotation = train_dataset[0]
    print("Sample 0:")
    print(f"  Piece shape: {piece.shape}")
    print(f"  Puzzle shape: {puzzle.shape}")
    print(f"  Target coords: ({target[0]:.3f}, {target[1]:.3f})")
    print(f"  Cell index: {cell.item()}")
    print(f"  Rotation index: {rotation.item()} " f"({ROTATION_ANGLES[int(rotation.item())]} deg)")
