"""Datasets for exp30: exp26 augmentation on losslessly-rotated, square-framed pieces.

Everything about exp26 is kept — the frozen exp20 split, the RGBA train
path through ``augment.augment_piece``, the independently jittered puzzle,
the deterministic black-composited val/test — with exactly two changes,
both of them fixes for the label leaks documented in
``docs/synthetic-dataset-realism.html`` section 4:

1. **Lossless load-time rotation.** The extra random 90deg rotation (train)
   and the all-4-rotations sweep (eval) go through
   :func:`framing.rotate_lossless` (``Image.transpose``) instead of
   ``rotate(expand=False, BILINEAR)``. No clipping, no half-pixel blur.
2. **Real-path framing.** Immediately after that rotation the RGBA piece is
   run through :func:`framing.frame_rgba`, reproducing exp24's deployment
   geometry (largest opaque component -> 8% margin -> pad to square).

Why framing *before* ``augment_piece`` rather than after: the framed canvas
is **square and rotation-symmetric**, so every subsequent step — exp26's
rotation jitter, perspective, scale, halo, the background sample sized to
``piece.size``, the composite, and the final 128x128 resize — treats the
four rotation classes identically by construction. Aspect is preserved
(square in, square out, no squash) and the 8% transparent margin means the
silhouette cannot reach the input border for any rotation. ``augment_piece``
itself is imported from exp26 **unchanged**, so augmentation behaviour is
otherwise exp26-identical.

exp26's ``augment.py`` needs no neutralisation: its only geometric rotation
(``_rotate_rgba``) already uses ``expand=True``, and its perspective warp
maps the canvas corners inward, so neither reintroduces the clip/blur cues.
"""

import random
from pathlib import Path

import torch
from PIL import Image

from ..exp20_realistic_pieces.dataset import (
    GRID_SIZE,
    NUM_CELLS,
    ROTATION_ANGLES,
    get_cell_index,
)
from ..exp20_realistic_pieces.splits import DEFAULT_SPLIT_PATH, load_split
from ..exp26_domain_randomization.aug_dataset import AugmentedPieceDataset
from ..exp26_domain_randomization.aug_dataset import BlackCompositeTestDataset as Exp26BlackCompositeTestDataset
from ..exp26_domain_randomization.augment import AugmentConfig, augment_piece, augment_puzzle
from .framing import frame_and_black_composite, frame_rgba, rotate_lossless

# exp30 pieces live in their own root: same geometry as the exp26 RGBA set
# but with the base rotation applied losslessly, so the two must not mix.
DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4_rgba_v2"
DEFAULT_PUZZLE_ROOT = Path(__file__).parent.parent.parent / "datasets" / "puzzles"


class FramedAugmentedPieceDataset(AugmentedPieceDataset):
    """exp26's augmented train dataset with lossless rotation + square framing.

    Only ``__getitem__`` changes; the sample scan, background sampler,
    puzzle LRU cache and tensor transforms are inherited from exp26
    untouched.
    """

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

        # Compose the 4-class label exactly as exp20/exp26 (a random extra
        # 90deg rotation on top of the baked-in base rotation), but apply it
        # with an exact transpose so no content is clipped and no half-pixel
        # blur cue is introduced.
        additional_idx = random.randint(0, 3)
        piece_rgba = rotate_lossless(piece_rgba, ROTATION_ANGLES[additional_idx])
        total_rotation = (base_rotation + ROTATION_ANGLES[additional_idx]) % 360
        rotation_idx = total_rotation // 90

        # Frame to the deployment geometry BEFORE augmenting: the canvas is
        # then square and rotation-symmetric, so every exp26 augmentation
        # downstream is rotation-independent by construction.
        piece_rgba = frame_rgba(piece_rgba)

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


def piece_to_model_input(piece_path: Path | str, applied_rotation_idx: int, piece_size: int = 128) -> Image.Image:
    """Turn one stored piece PNG into exactly what the model sees at eval time.

    The single-sample form of :class:`FramedBlackCompositeTestDataset`'s
    piece branch: load RGBA -> lossless clockwise rotation -> real-path
    square framing -> black composite -> resize. Exposed as a module-level
    hook so ``probes.py`` (and any future analysis) can read the exp30 eval
    view without instantiating a dataset over a full split.

    Args:
        piece_path: Path to a stored RGBA piece PNG.
        applied_rotation_idx: Additional test-time rotation index
            (0=0deg, 1=90deg, 2=180deg, 3=270deg, clockwise).
        piece_size: Square model-input size.

    Returns:
        The RGB model input, ``piece_size`` x ``piece_size``.
    """
    with Image.open(piece_path) as raw:
        piece_rgba = raw.convert("RGBA")
    rotated = rotate_lossless(piece_rgba, ROTATION_ANGLES[applied_rotation_idx])
    framed = frame_and_black_composite(rotated)
    return framed.resize((piece_size, piece_size), Image.Resampling.BILINEAR)


class FramedBlackCompositeTestDataset(Exp26BlackCompositeTestDataset):
    """Deterministic eval dataset: lossless rotation, real-path framing, black.

    Extends exp26's black-composite eval dataset (all-4-rotations, no
    augmentation). ``_load_piece`` now keeps the alpha channel and
    ``_rotate_piece`` does rotation, framing and black-compositing in one
    step, so the RGB handed to the resize is a **square** crop with an 8%
    black margin — byte-compatible with what
    ``exp24_piece_classifier.data_prep.rgba_to_classifier_input`` produces
    for a real rembg cut-out, and identical in framing across all four
    rotations.
    """

    def _load_piece(self, piece_path: Path) -> Image.Image:
        """Load an RGBA piece, keeping alpha for the framing step.

        Args:
            piece_path: Path to the piece PNG.

        Returns:
            The RGBA piece image.
        """
        with Image.open(piece_path) as raw:
            return raw.convert("RGBA")

    def _rotate_piece(self, piece_img: Image.Image, rotation_idx: int) -> Image.Image:
        """Rotate losslessly, frame to the real geometry and composite on black.

        Args:
            piece_img: RGBA piece image.
            rotation_idx: 0=0deg, 1=90deg, 2=180deg, 3=270deg (clockwise).

        Returns:
            A square RGB image with the rotated piece centred on black.
        """
        rotated = rotate_lossless(piece_img, ROTATION_ANGLES[rotation_idx])
        return frame_and_black_composite(rotated)


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

    Mirrors ``exp26_domain_randomization.aug_dataset.create_datasets_from_split``
    exactly, but instantiates the exp30 framed dataset classes over the
    lossless RGBA piece root.

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
    print("exp30 framing: lossless transpose rotation + 8%-margin square pad (exp24 real-path geometry)")

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
            datasets[name] = FramedAugmentedPieceDataset(
                puzzle_ids=present_ids,
                augment_config=augment_config,
                dataset_root=dataset_root,
                puzzle_root=puzzle_root,
                piece_size=piece_size,
                puzzle_size=puzzle_size,
                background_texture_ids=train_ids_present,
            )
        else:
            datasets[name] = FramedBlackCompositeTestDataset(
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
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_PUZZLE_ROOT",
    "GRID_SIZE",
    "NUM_CELLS",
    "FramedAugmentedPieceDataset",
    "FramedBlackCompositeTestDataset",
    "create_datasets_from_split",
    "piece_to_model_input",
]
