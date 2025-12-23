"""Dataset for position + rotation prediction with RANDOM rotation sampling.

This module is identical to exp12 - the data loading and augmentation
strategy remains unchanged. The key difference in exp13 is the DATASET SIZE
(~4000 training puzzles instead of 800).

Key features:
- Exp13 training: Random 1 rotation per quadrant = 4 samples per puzzle
- Exp13 testing: All 4 rotations for deterministic evaluation
- Uses the full ~5000 puzzle dataset for improved generalization
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets"

# Quadrant centers (normalized coordinates)
QUADRANT_CENTERS = [
    (0.25, 0.25),  # top-left
    (0.75, 0.25),  # top-right
    (0.25, 0.75),  # bottom-left
    (0.75, 0.75),  # bottom-right
]

# Rotation angles in degrees
ROTATION_ANGLES = [0, 90, 180, 270]


class QuadrantRandomRotationDataset(Dataset):
    """Dataset that generates RANDOMLY rotated 2x2 quadrant pieces.

    Unlike exp10's dataset which generated all 4 rotations for each piece,
    this dataset randomly samples ONE rotation per piece. This:
    1. Reduces puzzle exposure from 16x to 4x per epoch (prevents memorization)
    2. Provides rotation diversity across epochs
    3. Matches exp9's data structure (4 samples per puzzle)

    Each sample consists of:
    - piece: One quadrant of a puzzle image, randomly rotated
    - puzzle: The full puzzle image (not rotated)
    - target_position: Center coordinates (cx, cy) normalized to [0, 1]
    - quadrant_idx: Quadrant index (0-3)
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
    ):
        """Initialize the dataset with random rotation sampling.

        Args:
            puzzle_ids: List of puzzle IDs to include (e.g., ["puzzle_001"]).
            dataset_root: Root directory containing puzzles/ folder.
            piece_size: Size to resize piece images to (square).
            puzzle_size: Size to resize puzzle images to (square).
            augment: If True, apply data augmentation to pieces.
            random_rotation: If True, randomly sample rotation per access.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.augment = augment
        self.random_rotation = random_rotation

        # Build sample list: (puzzle_id, quadrant_idx)
        self.samples: list[tuple[str, int]] = []
        for puzzle_id in puzzle_ids:
            for quadrant_idx in range(4):
                self.samples.append((puzzle_id, quadrant_idx))

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
            f"QuadrantRandomRotationDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples (4 per puzzle, random rotation)"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image, using cache if available."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.dataset_root / "puzzles" / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _extract_quadrant(self, puzzle_img: Image.Image, quadrant_idx: int) -> Image.Image:
        """Extract a quadrant from the puzzle image.

        Args:
            puzzle_img: Full puzzle PIL Image.
            quadrant_idx: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right

        Returns:
            Cropped quadrant as PIL Image.
        """
        w, h = puzzle_img.size
        half_w, half_h = w // 2, h // 2

        if quadrant_idx == 0:  # top-left
            box = (0, 0, half_w, half_h)
        elif quadrant_idx == 1:  # top-right
            box = (half_w, 0, w, half_h)
        elif quadrant_idx == 2:  # bottom-left
            box = (0, half_h, half_w, h)
        else:  # bottom-right (3)
            box = (half_w, half_h, w, h)

        return puzzle_img.crop(box)

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
            Tuple of (piece_tensor, puzzle_tensor, target_coords, quadrant_idx,
            rotation_idx).
        """
        puzzle_id, quadrant_idx = self.samples[idx]

        # Randomly sample rotation (key difference from exp10!)
        if self.random_rotation:
            rotation_idx = random.randint(0, 3)
        else:
            rotation_idx = 0

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Extract quadrant as piece
        piece_img = self._extract_quadrant(puzzle_img, quadrant_idx)

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

        # Target coordinates (center of quadrant)
        cx, cy = QUADRANT_CENTERS[quadrant_idx]
        target = torch.tensor([cx, cy], dtype=torch.float32)

        return (
            piece_result,
            puzzle_result,
            target,
            torch.tensor(quadrant_idx),
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


class QuadrantAllRotationsDataset(Dataset):
    """Dataset that generates ALL rotations for testing.

    For evaluation, we want to test all 4 rotations for each piece
    (deterministic). This ensures consistent metrics across evaluations.

    Each puzzle produces 16 samples: 4 quadrants x 4 rotations.
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

        # Build sample list: (puzzle_id, quadrant_idx, rotation_idx)
        self.samples: list[tuple[str, int, int]] = []
        for puzzle_id in puzzle_ids:
            for quadrant_idx in range(4):
                for rotation_idx in range(4):
                    self.samples.append((puzzle_id, quadrant_idx, rotation_idx))

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
            f"QuadrantAllRotationsDataset: {len(puzzle_ids)} puzzles, " f"{len(self.samples)} samples (16 per puzzle)"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image, using cache if available."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.dataset_root / "puzzles" / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _extract_quadrant(self, puzzle_img: Image.Image, quadrant_idx: int) -> Image.Image:
        """Extract a quadrant from the puzzle image."""
        w, h = puzzle_img.size
        half_w, half_h = w // 2, h // 2

        if quadrant_idx == 0:
            box = (0, 0, half_w, half_h)
        elif quadrant_idx == 1:
            box = (half_w, 0, w, half_h)
        elif quadrant_idx == 2:
            box = (0, half_h, half_w, h)
        else:
            box = (half_w, half_h, w, h)

        return puzzle_img.crop(box)

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
            Tuple of (piece_tensor, puzzle_tensor, target_coords, quadrant_idx,
            rotation_idx).
        """
        puzzle_id, quadrant_idx, rotation_idx = self.samples[idx]

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Extract quadrant as piece
        piece_img = self._extract_quadrant(puzzle_img, quadrant_idx)

        # Apply rotation to piece
        piece_img = self._rotate_piece(piece_img, rotation_idx)

        # Apply transforms
        piece_result = self.piece_transform(piece_img)
        puzzle_result = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_result, torch.Tensor)
        assert isinstance(puzzle_result, torch.Tensor)

        # Target coordinates
        cx, cy = QUADRANT_CENTERS[quadrant_idx]
        target = torch.tensor([cx, cy], dtype=torch.float32)

        return (
            piece_result,
            puzzle_result,
            target,
            torch.tensor(quadrant_idx),
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
    n_train_puzzles: int = 800,
    n_test_puzzles: int = 200,
    piece_size: int = 128,
    puzzle_size: int = 256,
    seed: int = 42,
) -> tuple[QuadrantRandomRotationDataset, QuadrantAllRotationsDataset]:
    """Create train and test datasets with random/all rotations.

    Training: Random rotation per sample (4 samples per puzzle)
    Testing: All rotations (16 samples per puzzle) for complete evaluation

    Args:
        dataset_root: Root directory.
        n_train_puzzles: Number of puzzles for training.
        n_test_puzzles: Number of puzzles for testing.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        seed: Random seed.

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

    print(f"\nDataset split (seed={seed}):")
    print(f"  Training puzzles: {len(train_ids)} ({len(train_ids) * 4} samples/epoch)")
    print(f"  Test puzzles: {len(test_ids)} ({len(test_ids) * 16} samples)")

    train_dataset = QuadrantRandomRotationDataset(
        puzzle_ids=train_ids,
        dataset_root=dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        augment=True,
        random_rotation=True,
    )

    test_dataset = QuadrantAllRotationsDataset(
        puzzle_ids=test_ids,
        dataset_root=dataset_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    print("Testing QuadrantRandomRotationDataset for exp12...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")

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
    piece, puzzle, target, quadrant, rotation = train_dataset[0]
    print("Sample 0:")
    print(f"  Piece shape: {piece.shape}")
    print(f"  Puzzle shape: {puzzle.shape}")
    print(f"  Target coords: ({target[0]:.2f}, {target[1]:.2f})")
    print(f"  Quadrant index: {quadrant.item()}")
    print(f"  Rotation index: {rotation.item()} ({ROTATION_ANGLES[int(rotation.item())]} deg)")
