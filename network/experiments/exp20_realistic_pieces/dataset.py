"""Dataset for 4x4 grid position + rotation prediction with realistic pieces.

Exp20: Uses pre-generated realistic puzzle pieces with Bezier curve edges.
Pieces are stored on disk with center coordinates in the filename.

Key features:
- 4x4 grid = 16 cells instead of 3x3 = 9 cells
- Pre-cut pieces loaded from disk (not extracted at runtime)
- Filename format: puzzle_id_x{cx}_y{cy}_rot{rotation}.png
- Black background (transparency filled)
"""

import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4"
DEFAULT_PUZZLE_ROOT = Path(__file__).parent.parent.parent / "datasets" / "puzzles"

# Grid configuration
GRID_SIZE = 4  # 4x4 grid
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 16 cells

# Cell centers (normalized coordinates) for 4x4 grid
# Cells are numbered row-major: 0-3 (top row), 4-7 (second row), etc.
CELL_CENTERS = []
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        cx = (col + 0.5) / GRID_SIZE  # 0.125, 0.375, 0.625, 0.875
        cy = (row + 0.5) / GRID_SIZE
        CELL_CENTERS.append((cx, cy))

# Rotation angles in degrees
ROTATION_ANGLES = [0, 90, 180, 270]

# Filename pattern: puzzle_id_x{cx}_y{cy}_rot{rotation}.png
PIECE_FILENAME_PATTERN = re.compile(r"(.+)_x([\d.]+)_y([\d.]+)_rot(\d+)\.png$")


def parse_piece_filename(filename: str) -> tuple[str, float, float, int] | None:
    """Parse piece filename to extract metadata.

    Args:
        filename: Piece filename like "puzzle_00001_x0.125_y0.125_rot90.png"

    Returns:
        Tuple of (puzzle_id, cx, cy, rotation) or None if pattern doesn't match.
    """
    match = PIECE_FILENAME_PATTERN.match(filename)
    if match:
        puzzle_id = match.group(1)
        cx = float(match.group(2))
        cy = float(match.group(3))
        rotation = int(match.group(4))
        return puzzle_id, cx, cy, rotation
    return None


def get_cell_index(cx: float, cy: float, grid_size: int = GRID_SIZE) -> int:
    """Convert center coordinates to cell index.

    Args:
        cx: X coordinate (0-1).
        cy: Y coordinate (0-1).
        grid_size: Grid size.

    Returns:
        Cell index (0 to grid_size^2 - 1).
    """
    col = int(cx * grid_size)
    row = int(cy * grid_size)
    # Clamp to valid range
    col = min(max(col, 0), grid_size - 1)
    row = min(max(row, 0), grid_size - 1)
    return row * grid_size + col


class RealisticPieceDataset(Dataset):
    """Dataset for training with pre-generated realistic puzzle pieces.

    Each sample consists of:
    - piece: Pre-cut realistic piece image (with black background)
    - puzzle: The full puzzle image (not cut)
    - target_position: Center coordinates (cx, cy) normalized to [0, 1]
    - cell_idx: Cell index (0-15)
    - rotation_idx: Rotation index (0=0deg, 1=90deg, 2=180deg, 3=270deg)

    Training mode: Random rotation per access for augmentation.
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
        augment: bool = False,
        random_rotation: bool = True,
        seed: int = 42,
    ):
        """Initialize the dataset.

        Args:
            puzzle_ids: List of puzzle IDs to include (e.g., ["puzzle_00001"]).
            dataset_root: Root directory containing realistic piece folders.
            puzzle_root: Root directory containing original puzzle images.
            piece_size: Size to resize piece images to (square).
            puzzle_size: Size to resize puzzle images to (square).
            augment: If True, apply data augmentation to pieces.
            random_rotation: If True, randomly sample rotation per access.
            seed: Random seed for reproducibility.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.puzzle_root = Path(puzzle_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.augment = augment
        self.random_rotation = random_rotation

        # Build sample list by scanning piece files
        self.samples: list[tuple[str, Path, float, float, int]] = []  # (puzzle_id, piece_path, cx, cy, base_rotation)

        for puzzle_id in puzzle_ids:
            puzzle_dir = self.dataset_root / puzzle_id
            if not puzzle_dir.exists():
                continue

            # Find all piece files for this puzzle
            piece_files = list(puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"))
            for piece_path in piece_files:
                parsed = parse_piece_filename(piece_path.name)
                if parsed:
                    _, cx, cy, rotation = parsed
                    self.samples.append((puzzle_id, piece_path, cx, cy, rotation))

        # Cache for loaded puzzle images (PIL Images)
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

        # Augmentation transforms for pieces
        if augment:
            self.piece_augment = transforms.Compose(
                [
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                ]
            )
        else:
            self.piece_augment = None

        print(
            f"RealisticPieceDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples (random_rotation={random_rotation})"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image, using cache if available."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.puzzle_root / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _load_piece(self, piece_path: Path) -> Image.Image:
        """Load piece image from disk."""
        return Image.open(piece_path).convert("RGB")

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
        return piece_img.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (piece_tensor, puzzle_tensor, target_coords, cell_idx, rotation_idx).
        """
        puzzle_id, piece_path, cx, cy, base_rotation = self.samples[idx]

        # Load piece (already has one rotation applied from generation)
        piece_img = self._load_piece(piece_path)

        # For training, we can apply additional random rotation
        if self.random_rotation:
            # Additional rotation on top of base rotation
            additional_rotation_idx = random.randint(0, 3)
            total_rotation = (base_rotation + ROTATION_ANGLES[additional_rotation_idx]) % 360
            rotation_idx = total_rotation // 90

            # Apply the additional rotation
            piece_img = self._rotate_piece(piece_img, additional_rotation_idx)
        else:
            rotation_idx = base_rotation // 90

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Apply augmentation to piece if enabled
        if self.piece_augment is not None:
            piece_img = self.piece_augment(piece_img)

        # Apply transforms
        piece_result = self.piece_transform(piece_img)
        puzzle_result = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_result, torch.Tensor)
        assert isinstance(puzzle_result, torch.Tensor)

        # Target coordinates (center of cell)
        target = torch.tensor([cx, cy], dtype=torch.float32)

        # Cell index
        cell_idx = get_cell_index(cx, cy)

        return (
            piece_result,
            puzzle_result,
            target,
            torch.tensor(cell_idx),
            torch.tensor(rotation_idx),
        )

    def clear_cache(self) -> None:
        """Clear the puzzle image cache to free memory."""
        self._puzzle_cache.clear()


class RealisticPieceTestDataset(Dataset):
    """Dataset for testing with ALL rotations per piece.

    For evaluation, we test all 4 rotations for each piece (deterministic).
    Each puzzle produces 64 samples: 16 cells x 4 rotations.
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
    ):
        """Initialize the test dataset.

        Args:
            puzzle_ids: List of puzzle IDs to include.
            dataset_root: Root directory containing realistic piece folders.
            puzzle_root: Root directory containing original puzzle images.
            piece_size: Size to resize piece images to.
            puzzle_size: Size to resize puzzle images to.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.puzzle_root = Path(puzzle_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size

        # Build sample list: for each piece, create 4 samples (one per rotation)
        self.samples: list[tuple[str, Path, float, float, int, int]] = []

        for puzzle_id in puzzle_ids:
            puzzle_dir = self.dataset_root / puzzle_id
            if not puzzle_dir.exists():
                continue

            # Get unique pieces (ignore the rotation in filename for grouping)
            piece_positions: dict[tuple[float, float], Path] = {}
            piece_files = list(puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"))

            for piece_path in piece_files:
                parsed = parse_piece_filename(piece_path.name)
                if parsed:
                    _, cx, cy, _ = parsed
                    # Use first file found for each position
                    if (cx, cy) not in piece_positions:
                        piece_positions[(cx, cy)] = piece_path

            # For each unique position, create samples for all 4 rotations
            for (cx, cy), piece_path in piece_positions.items():
                for rotation_idx in range(4):
                    self.samples.append((puzzle_id, piece_path, cx, cy, 0, rotation_idx))

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
            f"RealisticPieceTestDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples ({NUM_CELLS * 4} per puzzle)"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image, using cache if available."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.puzzle_root / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _load_piece(self, piece_path: Path) -> Image.Image:
        """Load piece image from disk."""
        return Image.open(piece_path).convert("RGB")

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
            Tuple of (piece_tensor, puzzle_tensor, target_coords, cell_idx, rotation_idx).
        """
        puzzle_id, piece_path, cx, cy, base_rotation, rotation_idx = self.samples[idx]

        # Load piece
        piece_img = self._load_piece(piece_path)

        # Apply test rotation
        piece_img = self._rotate_piece(piece_img, rotation_idx)

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Apply transforms
        piece_result = self.piece_transform(piece_img)
        puzzle_result = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_result, torch.Tensor)
        assert isinstance(puzzle_result, torch.Tensor)

        # Target coordinates
        target = torch.tensor([cx, cy], dtype=torch.float32)
        cell_idx = get_cell_index(cx, cy)

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
    """Get list of all available puzzle IDs from the realistic dataset.

    Args:
        dataset_root: Root directory containing puzzle subdirectories.

    Returns:
        Sorted list of puzzle IDs.
    """
    dataset_path = Path(dataset_root)
    puzzle_dirs = sorted([d.name for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("puzzle_")])
    return puzzle_dirs


def create_datasets(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
    n_train_puzzles: int = 500,
    n_test_puzzles: int = 50,
    piece_size: int = 128,
    puzzle_size: int = 256,
    seed: int = 42,
) -> tuple[RealisticPieceDataset, RealisticPieceTestDataset]:
    """Create train and test datasets.

    Training: Random rotation per sample for augmentation.
    Testing: All rotations (64 samples per puzzle) for complete evaluation.

    Args:
        dataset_root: Root directory containing realistic pieces.
        puzzle_root: Root directory containing original puzzles.
        n_train_puzzles: Number of puzzles for training.
        n_test_puzzles: Number of puzzles for testing.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    all_puzzles = get_puzzle_ids(dataset_root)

    if not all_puzzles:
        raise ValueError(f"No puzzles found in {dataset_root}. Run generate_dataset.py first.")

    # Shuffle and split
    rng = random.Random(seed)
    shuffled = all_puzzles.copy()
    rng.shuffle(shuffled)

    # Limit to requested sizes
    n_train = min(n_train_puzzles, len(shuffled) - n_test_puzzles)
    n_test = min(n_test_puzzles, len(shuffled) - n_train)

    train_ids = sorted(shuffled[:n_train])
    test_ids = sorted(shuffled[n_train : n_train + n_test])

    print(f"\nDataset split (seed={seed}):")
    print(f"  Training puzzles: {len(train_ids)} ({len(train_ids) * NUM_CELLS} pieces)")
    print(f"  Test puzzles: {len(test_ids)} ({len(test_ids) * NUM_CELLS * 4} samples)")

    train_dataset = RealisticPieceDataset(
        puzzle_ids=train_ids,
        dataset_root=dataset_root,
        puzzle_root=puzzle_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        augment=True,
        random_rotation=True,
        seed=seed,
    )

    test_dataset = RealisticPieceTestDataset(
        puzzle_ids=test_ids,
        dataset_root=dataset_root,
        puzzle_root=puzzle_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    print("Testing RealisticPieceDataset for exp20 (4x4 grid)...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")
    print(f"Puzzle root: {DEFAULT_PUZZLE_ROOT}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells")
    print(f"Cell centers: {CELL_CENTERS}")

    # Check available puzzles
    try:
        all_puzzles = get_puzzle_ids()
        print(f"\nTotal available puzzles: {len(all_puzzles)}")
        if all_puzzles:
            print(f"Sample puzzle IDs: {all_puzzles[:5]}")

            # Create small test datasets
            train_dataset, test_dataset = create_datasets(
                n_train_puzzles=10,
                n_test_puzzles=5,
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
            print(f"  Rotation index: {rotation.item()} ({ROTATION_ANGLES[int(rotation.item())]} deg)")
        else:
            print("\nNo puzzles found. Run generate_dataset.py first.")
    except Exception as e:
        print(f"Error: {e}")
        print("Run generate_dataset.py first to create the dataset.")
