"""Datasets for exp26: RGBA pieces + domain-randomization augmentation.

Reuses the exp20 harness pieces almost verbatim:

- The **frozen train/val/test split** is the exp20 one
  (``exp20_realistic_pieces/splits/realistic_4x4_v1.json``) — same puzzle
  IDs, so val/test remain comparable and north-star discipline (test
  touched once) is inherited.
- **train** loads RGBA pieces and applies ``augment.augment_piece`` /
  ``augment_puzzle`` so the piece and puzzle are photometrically
  independent (the exp25 anti-shortcut).
- **val / train_eval / test** black-composite the same RGBA pieces
  deterministically, reproducing the exp20 appearance so those metrics
  stay on the same footing as exp20/exp23. No augmentation is ever
  applied to val/test.
"""

import random
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ..exp20_realistic_pieces.dataset import (
    GRID_SIZE,
    NUM_CELLS,
    ROTATION_ANGLES,
    RealisticPieceTestDataset,
    get_cell_index,
    parse_piece_filename,
)
from ..exp20_realistic_pieces.splits import DEFAULT_SPLIT_PATH, load_split
from .augment import AugmentConfig, BackgroundSampler, augment_piece, augment_puzzle, black_composite

# exp26 stores RGBA pieces in a separate root so it never clobbers the
# black-composited exp20 dataset.
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4_rgba"
DEFAULT_PUZZLE_ROOT = Path(__file__).parent.parent.parent / "datasets" / "puzzles"


class AugmentedPieceDataset(torch.utils.data.Dataset):  # type: ignore[type-arg]
    """Training dataset: RGBA pieces with domain-randomization augmentation.

    Each ``__getitem__`` loads an RGBA piece, applies an additional random
    90-degree rotation (composing the 4-class label exactly as exp20 does),
    then runs the full ``augment_piece`` pipeline against an independently
    jittered puzzle. Turning ``AugmentConfig`` flags off (or passing
    ``augment_config.enabled=False``) recovers the un-augmented behaviour
    for the ablation.
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        augment_config: AugmentConfig,
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
        background_texture_ids: list[str] | None = None,
    ) -> None:
        """Initialize the augmented training dataset.

        Args:
            puzzle_ids: Puzzle IDs to include (train portion of the split).
            augment_config: Active domain-randomization config.
            dataset_root: Root directory of RGBA piece folders.
            puzzle_root: Root directory of source puzzle JPEGs.
            piece_size: Square size the piece is resized to.
            puzzle_size: Square size the puzzle is resized to.
            background_texture_ids: Puzzle IDs whose JPEGs may be used as
                texture backgrounds (defaults to ``puzzle_ids`` — training
                puzzles only, so val/test box art never leaks in).
        """
        self.puzzle_ids = puzzle_ids
        self.config = augment_config
        self.dataset_root = Path(dataset_root)
        self.puzzle_root = Path(puzzle_root)
        self.piece_size = piece_size
        self.puzzle_size = puzzle_size

        self.samples: list[tuple[str, Path, float, float, int]] = []
        for puzzle_id in puzzle_ids:
            puzzle_dir = self.dataset_root / puzzle_id
            if not puzzle_dir.exists():
                continue
            for piece_path in puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"):
                parsed = parse_piece_filename(piece_path.name)
                if parsed:
                    _, cx, cy, rotation = parsed
                    self.samples.append((puzzle_id, piece_path, cx, cy, rotation))

        texture_ids = background_texture_ids if background_texture_ids is not None else puzzle_ids
        texture_paths = [self.puzzle_root / f"{pid}.jpg" for pid in texture_ids]
        texture_paths = [p for p in texture_paths if p.exists()]
        self.background_sampler = BackgroundSampler(texture_paths=texture_paths)

        # LRU-bounded decoded-puzzle cache. exp20's unbounded cache grows to
        # every train puzzle (~8 GB of PIL images) PER DataLoader worker and
        # OOM-kills multi-worker training on the full 10k-puzzle split. With
        # random sampling the hit rate is near zero anyway, and re-decoding a
        # ~14 KB source JPEG costs single-digit milliseconds.
        self._puzzle_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self._puzzle_cache_max = 64

        self.piece_to_tensor = transforms.Compose([transforms.Resize((piece_size, piece_size)), transforms.ToTensor()])
        self.puzzle_to_tensor = transforms.Compose(
            [transforms.Resize((puzzle_size, puzzle_size)), transforms.ToTensor()]
        )

        print(
            f"AugmentedPieceDataset: {len(puzzle_ids)} puzzles, {len(self.samples)} samples, "
            f"{len(texture_paths)} texture bgs, aug={self.config.ablation_flags()}"
        )

    def _load_puzzle(self, puzzle_id: str) -> Image.Image:
        """Load a source puzzle image (RGB) through the LRU-bounded cache."""
        cached = self._puzzle_cache.get(puzzle_id)
        if cached is not None:
            self._puzzle_cache.move_to_end(puzzle_id)
            return cached
        with Image.open(self.puzzle_root / f"{puzzle_id}.jpg") as img:
            loaded = img.convert("RGB")
        self._puzzle_cache[puzzle_id] = loaded
        if len(self._puzzle_cache) > self._puzzle_cache_max:
            self._puzzle_cache.popitem(last=False)
        return loaded

    def __len__(self) -> int:
        """Return the number of pieces."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one augmented (piece, puzzle, target, cell, rotation) sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of piece tensor, puzzle tensor, (cx, cy) target,
            cell index and 4-class rotation index.
        """
        puzzle_id, piece_path, cx, cy, base_rotation = self.samples[idx]

        with Image.open(piece_path) as raw:
            piece_rgba = raw.convert("RGBA")

        # Compose the 4-class label exactly as exp20: add a random extra
        # 90-degree rotation on top of the baked-in base rotation.
        additional_idx = random.randint(0, 3)
        if additional_idx != 0:
            piece_rgba = piece_rgba.rotate(
                -ROTATION_ANGLES[additional_idx],
                expand=False,
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0, 0),
            )
        total_rotation = (base_rotation + ROTATION_ANGLES[additional_idx]) % 360
        rotation_idx = total_rotation // 90

        piece_rgb = augment_piece(piece_rgba, self.config, self.background_sampler)
        puzzle_rgb = augment_puzzle(self._load_puzzle(puzzle_id), self.config)

        piece_tensor = self.piece_to_tensor(piece_rgb)
        puzzle_tensor = self.puzzle_to_tensor(puzzle_rgb)
        assert isinstance(piece_tensor, torch.Tensor)
        assert isinstance(puzzle_tensor, torch.Tensor)

        return (
            piece_tensor,
            puzzle_tensor,
            torch.tensor([cx, cy], dtype=torch.float32),
            torch.tensor(get_cell_index(cx, cy)),
            torch.tensor(rotation_idx),
        )


class BlackCompositeTestDataset(RealisticPieceTestDataset):
    """Deterministic eval dataset over RGBA pieces, black-composited.

    Extends the exp20 test dataset (all-4-rotations, no augmentation) but
    black-composites the RGBA source pieces so the RGB fed to the model
    matches the exp20 / deployed appearance. Used for train_eval, val and
    test.
    """

    def _load_piece(self, piece_path: Path) -> Image.Image:
        """Load an RGBA piece and black-composite it to RGB."""
        with Image.open(piece_path) as raw:
            return black_composite(raw.convert("RGBA"))


def create_datasets_from_split(
    augment_config: AugmentConfig,
    split_path: Path | str | None = None,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
    piece_size: int = 128,
    puzzle_size: int = 256,
    allow_missing: bool = False,
) -> dict[str, torch.utils.data.Dataset]:  # type: ignore[type-arg]
    """Build train/train_eval/val/test datasets from the frozen exp20 split.

    Mirrors ``exp20_realistic_pieces.dataset.create_datasets_from_split``
    but returns the augmented train dataset and black-composite eval
    datasets over the RGBA piece root.

    Args:
        augment_config: Domain-randomization config for the train dataset.
        split_path: Frozen split JSON (default: exp20 v1 split).
        dataset_root: Root of RGBA piece folders.
        puzzle_root: Root of source puzzle JPEGs.
        piece_size: Square piece size.
        puzzle_size: Square puzzle size.
        allow_missing: Tolerate split puzzle dirs missing on disk (smoke
            tests only; results are NOT comparable to the benchmark).

    Returns:
        Mapping of split name to dataset.

    Raises:
        ValueError: If puzzle dirs are missing (unless ``allow_missing``)
            or a split resolves to zero samples.
    """
    split = load_split(split_path if split_path is not None else DEFAULT_SPLIT_PATH)
    root = Path(dataset_root)
    print("\nFrozen split: " + ", ".join(f"{name}={len(ids)} puzzles" for name, ids in split.items()))

    train_ids_present = [pid for pid in split["train"] if (root / pid).exists()]

    datasets: dict[str, torch.utils.data.Dataset] = {}  # type: ignore[type-arg]
    for name, ids in split.items():
        missing = [pid for pid in ids if not (root / pid).exists()]
        present_ids = ids
        if missing:
            message = f"{name}: {len(missing)}/{len(ids)} puzzle dirs missing under {root}"
            if not allow_missing:
                raise ValueError(
                    f"{message}. Generate RGBA pieces first (generate_dataset.py), or pass "
                    "allow_missing=True (train.py: --allow-missing-puzzles) for a smoke test "
                    "(results will NOT be comparable)."
                )
            present_ids = [pid for pid in ids if (root / pid).exists()]
            print(f"WARNING: {message} (allow_missing=True; results NOT comparable)")

        if name == "train":
            datasets[name] = AugmentedPieceDataset(
                puzzle_ids=present_ids,
                augment_config=augment_config,
                dataset_root=dataset_root,
                puzzle_root=puzzle_root,
                piece_size=piece_size,
                puzzle_size=puzzle_size,
                background_texture_ids=train_ids_present,
            )
        else:
            datasets[name] = BlackCompositeTestDataset(
                puzzle_ids=present_ids,
                dataset_root=dataset_root,
                puzzle_root=puzzle_root,
                piece_size=piece_size,
                puzzle_size=puzzle_size,
            )
        if len(datasets[name]) == 0:  # type: ignore[arg-type]
            raise ValueError(f"Split '{name}' has no samples under {root}; generate RGBA pieces first")

    return datasets


__all__ = [
    "GRID_SIZE",
    "NUM_CELLS",
    "AugmentedPieceDataset",
    "BlackCompositeTestDataset",
    "create_datasets_from_split",
]
