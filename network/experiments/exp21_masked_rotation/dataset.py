"""Dataset for exp21: Masked rotation correlation.

Extends exp20 dataset to generate masks from black background regions.
Masks are derived at runtime by detecting near-black pixels.

Key changes from exp20:
- generate_mask() function to create binary masks
- Dataset __getitem__ returns (piece, puzzle, mask, target, cell, rotation)
- Mask is [1, H, W] tensor where 1 = puzzle content, 0 = background
"""

import random
import re
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Default paths - reuse exp20's dataset
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4"
DEFAULT_PUZZLE_ROOT = Path(__file__).parent.parent.parent / "datasets" / "puzzles"

# Grid configuration (same as exp20)
GRID_SIZE = 4
NUM_CELLS = GRID_SIZE * GRID_SIZE  # 16 cells

# Cell centers for 4x4 grid
CELL_CENTERS = []
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        cx = (col + 0.5) / GRID_SIZE
        cy = (row + 0.5) / GRID_SIZE
        CELL_CENTERS.append((cx, cy))

ROTATION_ANGLES = [0, 90, 180, 270]

# Filename pattern from exp20
PIECE_FILENAME_PATTERN = re.compile(r"(.+)_x([\d.]+)_y([\d.]+)_rot(\d+)\.png$")


def parse_piece_filename(filename: str) -> tuple[str, float, float, int] | None:
    """Parse piece filename to extract metadata."""
    match = PIECE_FILENAME_PATTERN.match(filename)
    if match:
        puzzle_id = match.group(1)
        cx = float(match.group(2))
        cy = float(match.group(3))
        rotation = int(match.group(4))
        return puzzle_id, cx, cy, rotation
    return None


def get_cell_index(cx: float, cy: float, grid_size: int = GRID_SIZE) -> int:
    """Convert center coordinates to cell index."""
    col = int(cx * grid_size)
    row = int(cy * grid_size)
    col = min(max(col, 0), grid_size - 1)
    row = min(max(row, 0), grid_size - 1)
    return row * grid_size + col


def generate_mask(piece_tensor: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Generate mask from piece by detecting non-black pixels.

    Args:
        piece_tensor: Piece image [3, H, W] with values in [0, 1].
        threshold: Pixels with mean RGB < threshold are considered background.
            Default 0.02 allows for slight noise in black regions.

    Returns:
        Binary mask [1, H, W] where 1 = puzzle content, 0 = background.
    """
    # Mean across RGB channels
    mean_rgb = piece_tensor.mean(dim=0, keepdim=True)  # [1, H, W]
    # Black pixels have mean ~0, puzzle content has mean > threshold
    mask = (mean_rgb > threshold).float()
    return mask


class MaskedPieceDataset(Dataset):
    """Dataset that returns piece images with masks.

    Each sample consists of:
    - piece: Pre-cut realistic piece image [3, H, W]
    - puzzle: Full puzzle image [3, H, W]
    - mask: Binary mask [1, H, W] where 1 = puzzle content
    - target_position: Center coordinates (cx, cy) in [0, 1]
    - cell_idx: Cell index (0-15)
    - rotation_idx: Rotation index (0=0deg, 1=90deg, 2=180deg, 3=270deg)
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
        mask_threshold: float = 0.02,
        seed: int = 42,
    ):
        """Initialize the dataset.

        Args:
            puzzle_ids: List of puzzle IDs to include.
            dataset_root: Root directory containing realistic piece folders.
            puzzle_root: Root directory containing original puzzle images.
            piece_size: Size to resize piece images to.
            puzzle_size: Size to resize puzzle images to.
            augment: If True, apply data augmentation.
            random_rotation: If True, randomly sample rotation per access.
            mask_threshold: Threshold for mask generation.
            seed: Random seed.
        """
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.puzzle_root = Path(puzzle_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.augment = augment
        self.random_rotation = random_rotation
        self.mask_threshold = mask_threshold

        # Build sample list
        self.samples: list[tuple[str, Path, float, float, int]] = []

        for puzzle_id in puzzle_ids:
            puzzle_dir = self.dataset_root / puzzle_id
            if not puzzle_dir.exists():
                continue

            piece_files = list(puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"))
            for piece_path in piece_files:
                parsed = parse_piece_filename(piece_path.name)
                if parsed:
                    _, cx, cy, rotation = parsed
                    self.samples.append((puzzle_id, piece_path, cx, cy, rotation))

        # Puzzle cache
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

        # Augmentation
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
            f"MaskedPieceDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples (random_rotation={random_rotation})"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image with caching."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.puzzle_root / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _load_piece(self, piece_path: Path) -> Image.Image:
        """Load piece image."""
        return Image.open(piece_path).convert("RGB")

    def _rotate_piece(self, piece_img: Image.Image, rotation_idx: int) -> Image.Image:
        """Rotate piece by specified angle."""
        angle = ROTATION_ANGLES[rotation_idx]
        if angle == 0:
            return piece_img
        return piece_img.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a sample with mask.

        Returns:
            Tuple of (piece, puzzle, mask, target_coords, cell_idx, rotation_idx).
        """
        puzzle_id, piece_path, cx, cy, base_rotation = self.samples[idx]

        # Load piece
        piece_img = self._load_piece(piece_path)

        # Apply rotation
        if self.random_rotation:
            additional_rotation_idx = random.randint(0, 3)
            total_rotation = (base_rotation + ROTATION_ANGLES[additional_rotation_idx]) % 360
            rotation_idx = total_rotation // 90
            piece_img = self._rotate_piece(piece_img, additional_rotation_idx)
        else:
            rotation_idx = base_rotation // 90

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Apply augmentation (before mask generation to avoid augmenting mask)
        if self.piece_augment is not None:
            piece_img = self.piece_augment(piece_img)

        # Transform to tensors
        piece_tensor = self.piece_transform(piece_img)
        puzzle_tensor = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_tensor, torch.Tensor)
        assert isinstance(puzzle_tensor, torch.Tensor)

        # Generate mask from piece (before any color augmentation ideally, but after rotation)
        # We generate from the transformed tensor since that's what the model sees
        mask = generate_mask(piece_tensor, threshold=self.mask_threshold)

        # Target coordinates
        target = torch.tensor([cx, cy], dtype=torch.float32)
        cell_idx = get_cell_index(cx, cy)

        return (
            piece_tensor,
            puzzle_tensor,
            mask,
            target,
            torch.tensor(cell_idx),
            torch.tensor(rotation_idx),
        )

    def clear_cache(self) -> None:
        """Clear puzzle cache."""
        self._puzzle_cache.clear()


class MaskedPieceTestDataset(Dataset):
    """Test dataset with masks - tests all 4 rotations per piece."""

    def __init__(
        self,
        puzzle_ids: list[str],
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
        mask_threshold: float = 0.02,
    ):
        """Initialize the test dataset."""
        self.puzzle_ids = puzzle_ids
        self.dataset_root = Path(dataset_root)
        self.puzzle_root = Path(puzzle_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size
        self.mask_threshold = mask_threshold

        # Build samples: each position tested with all 4 rotations
        self.samples: list[tuple[str, Path, float, float, int, int]] = []

        for puzzle_id in puzzle_ids:
            puzzle_dir = self.dataset_root / puzzle_id
            if not puzzle_dir.exists():
                continue

            # Get unique positions
            piece_positions: dict[tuple[float, float], Path] = {}
            piece_files = list(puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"))

            for piece_path in piece_files:
                parsed = parse_piece_filename(piece_path.name)
                if parsed:
                    _, cx, cy, _ = parsed
                    if (cx, cy) not in piece_positions:
                        piece_positions[(cx, cy)] = piece_path

            # Create samples for all rotations
            for (cx, cy), piece_path in piece_positions.items():
                for rotation_idx in range(4):
                    self.samples.append((puzzle_id, piece_path, cx, cy, 0, rotation_idx))

        # Cache
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
            f"MaskedPieceTestDataset: {len(puzzle_ids)} puzzles, "
            f"{len(self.samples)} samples ({NUM_CELLS * 4} per puzzle)"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load puzzle image with caching."""
        if puzzle_id not in self._puzzle_cache:
            puzzle_path = self.puzzle_root / f"{puzzle_id}.jpg"
            self._puzzle_cache[puzzle_id] = Image.open(puzzle_path).convert("RGB")
        return self._puzzle_cache[puzzle_id]

    def _load_piece(self, piece_path: Path) -> Image.Image:
        """Load piece image."""
        return Image.open(piece_path).convert("RGB")

    def _rotate_piece(self, piece_img: Image.Image, rotation_idx: int) -> Image.Image:
        """Rotate piece by specified angle."""
        angle = ROTATION_ANGLES[rotation_idx]
        if angle == 0:
            return piece_img
        return piece_img.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a test sample with mask."""
        puzzle_id, piece_path, cx, cy, _, rotation_idx = self.samples[idx]

        # Load and rotate piece
        piece_img = self._load_piece(piece_path)
        piece_img = self._rotate_piece(piece_img, rotation_idx)

        # Load puzzle
        puzzle_img = self._load_puzzle(puzzle_id)

        # Transform
        piece_tensor = self.piece_transform(piece_img)
        puzzle_tensor = self.puzzle_transform(puzzle_img)
        assert isinstance(piece_tensor, torch.Tensor)
        assert isinstance(puzzle_tensor, torch.Tensor)

        # Generate mask
        mask = generate_mask(piece_tensor, threshold=self.mask_threshold)

        # Targets
        target = torch.tensor([cx, cy], dtype=torch.float32)
        cell_idx = get_cell_index(cx, cy)

        return (
            piece_tensor,
            puzzle_tensor,
            mask,
            target,
            torch.tensor(cell_idx),
            torch.tensor(rotation_idx),
        )

    def clear_cache(self) -> None:
        """Clear puzzle cache."""
        self._puzzle_cache.clear()


def get_puzzle_ids(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> list[str]:
    """Get list of available puzzle IDs."""
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
    mask_threshold: float = 0.02,
    seed: int = 42,
) -> tuple[MaskedPieceDataset, MaskedPieceTestDataset]:
    """Create train and test datasets with mask support.

    Args:
        dataset_root: Root directory containing realistic pieces.
        puzzle_root: Root directory containing original puzzles.
        n_train_puzzles: Number of puzzles for training.
        n_test_puzzles: Number of puzzles for testing.
        piece_size: Size of piece images.
        puzzle_size: Size of puzzle images.
        mask_threshold: Threshold for mask generation.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    all_puzzles = get_puzzle_ids(dataset_root)

    if not all_puzzles:
        raise ValueError(f"No puzzles found in {dataset_root}. Run exp20's generate_dataset.py first.")

    # Shuffle and split
    rng = random.Random(seed)
    shuffled = all_puzzles.copy()
    rng.shuffle(shuffled)

    n_train = min(n_train_puzzles, len(shuffled) - n_test_puzzles)
    n_test = min(n_test_puzzles, len(shuffled) - n_train)

    train_ids = sorted(shuffled[:n_train])
    test_ids = sorted(shuffled[n_train : n_train + n_test])

    print(f"\nDataset split (seed={seed}):")
    print(f"  Training puzzles: {len(train_ids)} ({len(train_ids) * NUM_CELLS} pieces)")
    print(f"  Test puzzles: {len(test_ids)} ({len(test_ids) * NUM_CELLS * 4} samples)")

    train_dataset = MaskedPieceDataset(
        puzzle_ids=train_ids,
        dataset_root=dataset_root,
        puzzle_root=puzzle_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        augment=True,
        random_rotation=True,
        mask_threshold=mask_threshold,
        seed=seed,
    )

    test_dataset = MaskedPieceTestDataset(
        puzzle_ids=test_ids,
        dataset_root=dataset_root,
        puzzle_root=puzzle_root,
        piece_size=piece_size,
        puzzle_size=puzzle_size,
        mask_threshold=mask_threshold,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing MaskedPieceDataset for exp21...")
    print(f"Dataset root: {DEFAULT_DATASET_ROOT}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells")

    try:
        all_puzzles = get_puzzle_ids()
        print(f"\nTotal available puzzles: {len(all_puzzles)}")

        if all_puzzles:
            train_dataset, test_dataset = create_datasets(
                n_train_puzzles=10,
                n_test_puzzles=5,
            )

            # Test sample
            piece, puzzle, mask, target, cell, rotation = train_dataset[0]
            print("\n--- Sample 0 ---")
            print(f"  Piece shape: {piece.shape}")
            print(f"  Puzzle shape: {puzzle.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask coverage: {mask.mean().item():.1%}")
            print(f"  Target: ({target[0]:.3f}, {target[1]:.3f})")
            print(f"  Cell: {cell.item()}, Rotation: {rotation.item() * 90} deg")

            # Visualize mask
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(piece.permute(1, 2, 0).numpy())
            axes[0].set_title("Piece")
            axes[0].axis("off")

            axes[1].imshow(mask[0].numpy(), cmap="gray")
            axes[1].set_title(f"Mask ({mask.mean().item():.1%} coverage)")
            axes[1].axis("off")

            # Masked piece
            masked = piece * mask
            axes[2].imshow(masked.permute(1, 2, 0).numpy())
            axes[2].set_title("Masked Piece")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(Path(__file__).parent / "outputs" / "mask_test.png", dpi=150)
            print("\nSaved mask visualization to outputs/mask_test.png")
            plt.close()
        else:
            print("\nNo puzzles found. Run exp20's generate_dataset.py first.")
    except Exception as e:
        print(f"Error: {e}")
