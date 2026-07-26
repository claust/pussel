"""Datasets for exp31: two independent captures of two different prints.

Everything structural is exp30's, which is everything comparable about exp26's:
the frozen exp20 split, the exp20 harness, the RGBA piece root
(``datasets/realistic_4x4_rgba_v2`` — exp31 reuses exp30's stored pieces
unchanged, since exp31 changes only what happens at *load* time), the
lossless ``Image.transpose`` label rotation, and exp30's real-path square
framing. The deterministic val/test datasets are exp30's, untouched.

The one change is the **train** path: instead of exp26's
``augment_piece`` + ``augment_puzzle`` — which share a source file and
therefore a demosaic, white balance, JPEG block grid, noise fingerprint and
sharpening — the piece and the overview are rendered by
:func:`capture.augment_view_pair` as two independent captures, and
Asymmetric Random Patching adds content to exactly one of them.

Why val/test stay clean and unaugmented: the pixel-identity shortcut is a
property of the *training* pair, and keeping the evaluation protocol
byte-compatible with exp20/exp26/exp30 is the only way the synthetic
accuracy numbers remain comparable across the four experiments. The
inherited limitation (also exp26's and exp30's) is that checkpoint selection
therefore happens on a distribution where the shortcut still works; there is
no real validation set.

For probes and acceptance gates, :func:`view_pair` renders both views for one
piece at native *and* model-input scale, with the piece's mask, from a
reproducible seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..exp20_realistic_pieces.dataset import (
    GRID_SIZE,
    NUM_CELLS,
    ROTATION_ANGLES,
    get_cell_index,
    parse_piece_filename,
)
from ..exp20_realistic_pieces.splits import DEFAULT_SPLIT_PATH, load_split
from ..exp26_domain_randomization.augment import BackgroundSampler
from ..exp30_generator_fixes.framed_dataset import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_PUZZLE_ROOT,
    FramedAugmentedPieceDataset,
    FramedBlackCompositeTestDataset,
)
from ..exp30_generator_fixes.framed_dataset import piece_to_model_input as exp30_piece_to_model_input
from ..exp30_generator_fixes.framing import frame_rgba, rotate_lossless
from .capture import CaptureConfig, PatchSource, augment_view_pair

# Resize kernel used for the two model-input views, matching the
# ``transforms.Resize`` default the exp20/exp26/exp30 datasets use.
INPUT_RESAMPLE = Image.Resampling.BILINEAR


class CapturePieceDataset(FramedAugmentedPieceDataset):
    """Training dataset: piece and overview rendered as independent captures.

    Inherits exp30's sample scan, background sampler, puzzle LRU cache and
    tensor transforms; only ``__getitem__`` changes. The ARP patch source is
    built over the same **training-only** puzzle JPEGs the background sampler
    uses, so no val/test box art and no north_star imagery can leak in
    through a patch.
    """

    def __init__(
        self,
        puzzle_ids: list[str],
        augment_config: CaptureConfig,
        dataset_root: Path | str = DEFAULT_DATASET_ROOT,
        puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
        piece_size: int = 128,
        puzzle_size: int = 256,
        background_texture_ids: list[str] | None = None,
    ) -> None:
        """Initialize the dataset and its ARP patch source.

        Signature mirrors exp26/exp30's train dataset exactly; the only
        addition is the patch source, built over the same training-only
        puzzle JPEGs the background sampler already resolved.

        Args:
            puzzle_ids: Puzzle IDs to include (train portion of the split).
            augment_config: Active exp31 capture config.
            dataset_root: Root directory of RGBA piece folders.
            puzzle_root: Root directory of source puzzle JPEGs.
            piece_size: Square size the piece is resized to.
            puzzle_size: Square size the overview is resized to.
            background_texture_ids: Puzzle IDs usable as background/patch
                content (defaults to ``puzzle_ids``).
        """
        super().__init__(
            puzzle_ids=puzzle_ids,
            augment_config=augment_config,
            dataset_root=dataset_root,
            puzzle_root=puzzle_root,
            piece_size=piece_size,
            puzzle_size=puzzle_size,
            background_texture_ids=background_texture_ids,
        )
        self.patch_source = PatchSource(texture_paths=list(self.background_sampler.texture_paths))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one (piece, overview, target, cell, rotation) training sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of piece tensor, puzzle tensor, (cx, cy) target, cell index
            and 4-class rotation index.
        """
        puzzle_id, piece_path, cx, cy, base_rotation = self.samples[idx]

        with Image.open(piece_path) as raw:
            piece_rgba = raw.convert("RGBA")

        # Label composition is exp20's, applied losslessly as in exp30.
        additional_idx = random.randint(0, 3)
        piece_rgba = rotate_lossless(piece_rgba, ROTATION_ANGLES[additional_idx])
        rotation_idx = ((base_rotation + ROTATION_ANGLES[additional_idx]) % 360) // 90

        assert isinstance(self.config, CaptureConfig), "CapturePieceDataset requires a CaptureConfig"
        piece_rgb, _mask, puzzle_rgb = augment_view_pair(
            piece_rgba,
            self._load_puzzle(puzzle_id),
            self.config,
            background_sampler=self.background_sampler,
            patch_source=self.patch_source,
            exclude_stem=puzzle_id,
        )

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


@dataclass
class ViewPair:
    """Both views of one piece, at native and model-input scale.

    The probe contract for exp31 (``ncc_probe.py`` and any later analysis).
    Every field is a PIL image except the labels; ``piece_mask_*`` is what
    makes a *masked* NCC possible, and the ``*_raw_*`` fields are the
    un-degraded baseline so a single call yields both the "synthetic raw"
    (0.990 today) and the "synthetic + exp31" numbers section 8 compares.
    """

    piece_native: Image.Image
    piece_input: Image.Image
    piece_mask_native: Image.Image
    piece_mask_input: Image.Image
    overview_native: Image.Image
    overview_input: Image.Image
    piece_raw_native: Image.Image
    piece_raw_mask_native: Image.Image
    overview_raw_native: Image.Image
    puzzle_id: str
    piece_path: Path
    cx: float
    cy: float
    rotation_idx: int


def deterministic_seed(piece_path: Path | str, applied_rotation_idx: int, base: int = 31) -> int:
    """Return a stable per-(piece, rotation) seed.

    Derived from the piece filename rather than from an enumeration index, so
    a probe's sample set is reproducible regardless of directory order or
    sampling strategy.

    Args:
        piece_path: Path to the stored piece PNG.
        applied_rotation_idx: Applied rotation index (0-3).
        base: Base offset, to draw independent realizations of the same set.

    Returns:
        A non-negative seed below 2**31.
    """
    stem = Path(piece_path).name
    digest = 0
    for char in stem:
        digest = (digest * 131 + ord(char)) % (2**31 - 1)
    return (digest * 4 + applied_rotation_idx + base) % (2**31 - 1)


def _seed_globals(seed: int) -> None:
    """Seed the process-global RNGs the augmentations draw from.

    The exp26 stages :mod:`capture` composes (``ColorJitter``,
    ``RandomPerspective``, ``BackgroundSampler``) draw from ``random`` and
    ``np.random``, so reproducing a view pair means seeding those globals.
    Intended for probes and tests, never for the training path.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _resize(image: Image.Image, size: int) -> Image.Image:
    """Resize an image to a square of ``size`` px with the dataset's kernel.

    Args:
        image: Image to resize.
        size: Target side length.

    Returns:
        The resized image.
    """
    return image.resize((size, size), INPUT_RESAMPLE)


def view_pair(
    piece_path: Path | str,
    puzzle_path: Path | str,
    config: CaptureConfig | None = None,
    applied_rotation_idx: int = 0,
    seed: int | None = None,
    piece_size: int = 128,
    puzzle_size: int = 256,
    background_sampler: BackgroundSampler | None = None,
    patch_source: PatchSource | None = None,
) -> ViewPair:
    """Render both exp31 views for one stored piece — the probe entry point.

    This is the interface the acceptance-gate probes are meant to call. One
    call returns, for a single piece:

    - the exp31 piece and overview views at **native** scale (the scale the
      capture chains actually operate at, which is where a masked NCC search
      should be run), and at **model-input** scale (128 / 256 px);
    - the piece's alpha **mask** at both scales, for masked correlation;
    - the corresponding **un-degraded** views (exp30's clean framing and the
      raw source JPEG), so the "synthetic raw" baseline and the exp31 number
      come out of one call and one framing.

    Args:
        piece_path: Path to a stored RGBA piece PNG.
        puzzle_path: Path to the source puzzle JPEG (the overview source).
        config: Capture config (defaults to :class:`CaptureConfig`).
        applied_rotation_idx: Extra rotation index (0=0deg .. 3=270deg,
            clockwise), composed on top of the piece's baked-in rotation
            exactly as the train and eval paths do.
        seed: Seed for a reproducible realization. Seeds the process-global
            RNGs (see :func:`_seed_globals`); None leaves them alone.
        piece_size: Square model-input size for the piece view.
        puzzle_size: Square model-input size for the overview view.
        background_sampler: Background sampler for the piece composite (None
            uses procedural/black backgrounds only).
        patch_source: ARP patch content supplier (None builds an empty one,
            which falls back to procedural patches).

    Returns:
        The :class:`ViewPair` for this piece.
    """
    active = config if config is not None else CaptureConfig()
    if seed is not None:
        _seed_globals(seed)

    piece_path = Path(piece_path)
    puzzle_path = Path(puzzle_path)
    with Image.open(piece_path) as raw:
        stored = raw.convert("RGBA")
    with Image.open(puzzle_path) as raw_puzzle:
        overview_source = raw_puzzle.convert("RGB")

    rotated = rotate_lossless(stored, ROTATION_ANGLES[applied_rotation_idx])
    # exp30's un-jittered framing: the "synthetic raw" baseline the section 8
    # NCC probe compares against, in the same geometry as the exp31 view.
    raw_framed = frame_rgba(rotated)

    piece_rgb, mask, overview_rgb = augment_view_pair(
        rotated,
        overview_source,
        active,
        background_sampler=background_sampler,
        patch_source=patch_source,
        exclude_stem=puzzle_id_of(piece_path),
    )

    cx, cy, rotation_idx = _labels(piece_path, applied_rotation_idx)
    return ViewPair(
        piece_native=piece_rgb,
        piece_input=_resize(piece_rgb, piece_size),
        piece_mask_native=mask,
        piece_mask_input=_resize(mask, piece_size),
        overview_native=overview_rgb,
        overview_input=_resize(overview_rgb, puzzle_size),
        piece_raw_native=raw_framed.convert("RGB"),
        piece_raw_mask_native=raw_framed.getchannel("A"),
        overview_raw_native=overview_source,
        puzzle_id=puzzle_id_of(piece_path),
        piece_path=piece_path,
        cx=cx,
        cy=cy,
        rotation_idx=rotation_idx,
    )


def puzzle_id_of(piece_path: Path | str) -> str:
    """Return the puzzle ID a stored piece belongs to.

    Args:
        piece_path: Path to a stored piece PNG (``<puzzle_id>_x..._y..._rot...``).

    Returns:
        The puzzle ID (the parent directory's name).
    """
    return Path(piece_path).parent.name


def _labels(piece_path: Path, applied_rotation_idx: int) -> tuple[float, float, int]:
    """Parse a stored piece's labels, composing the applied rotation.

    Args:
        piece_path: Path to a stored piece PNG.
        applied_rotation_idx: Applied rotation index (0-3).

    Returns:
        Tuple of (cx, cy, rotation_idx).

    Raises:
        ValueError: If the filename does not carry labels. Returning a
            placeholder here would be worse than failing: (0.0, 0.0) is a
            *plausible* cell centre, so a misnamed file would silently
            anchor the NCC probe at the wrong ground-truth location.
    """
    parsed = parse_piece_filename(piece_path.name)
    if parsed is None:
        raise ValueError(
            f"Cannot parse labels from piece filename {piece_path.name!r}; expected "
            "'<puzzle_id>_x<cx>_y<cy>_rot<deg>.png' as written by generate_dataset.py."
        )
    _, cx, cy, base_rotation = parsed
    return cx, cy, ((base_rotation + ROTATION_ANGLES[applied_rotation_idx]) % 360) // 90


def piece_to_model_input(piece_path: Path | str, applied_rotation_idx: int, piece_size: int = 128) -> Image.Image:
    """Return exactly what the model sees for one piece at **eval** time.

    exp31's eval path is exp30's, unchanged, so this delegates. Kept under
    the same name so the acceptance-probe machinery that discovers a
    pipeline's eval view by hook name finds exp31 too.

    Args:
        piece_path: Path to a stored RGBA piece PNG.
        applied_rotation_idx: Applied rotation index (0-3, clockwise).
        piece_size: Square model-input size.

    Returns:
        The RGB model input, ``piece_size`` x ``piece_size``.
    """
    return exp30_piece_to_model_input(piece_path, applied_rotation_idx, piece_size)


def create_datasets_from_split(
    capture_config: CaptureConfig,
    split_path: Path | str | None = None,
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    puzzle_root: Path | str = DEFAULT_PUZZLE_ROOT,
    piece_size: int = 128,
    puzzle_size: int = 256,
    allow_missing: bool = False,
) -> dict[str, torch.utils.data.Dataset]:  # type: ignore[type-arg]
    """Build train/train_eval/val/test datasets from the frozen exp20 split.

    Mirrors ``exp30_generator_fixes.framed_dataset.create_datasets_from_split``
    exactly — same split, same roots, same eval classes — with the train
    dataset swapped for :class:`CapturePieceDataset`.

    Args:
        capture_config: exp31 capture config for the train dataset.
        split_path: Frozen split JSON (default: exp20 v1 split).
        dataset_root: Root of RGBA piece folders (exp30's v2 root).
        puzzle_root: Root of source puzzle JPEGs.
        piece_size: Square piece size.
        puzzle_size: Square puzzle size.
        allow_missing: Tolerate split puzzle dirs missing on disk (smoke
            tests only; results are NOT comparable to the benchmark).

    Returns:
        Mapping of split name to dataset.

    Raises:
        ValueError: If puzzle dirs are missing (unless ``allow_missing``) or a
            split resolves to zero samples.
    """
    split = load_split(split_path if split_path is not None else DEFAULT_SPLIT_PATH)
    root = Path(dataset_root)
    print("\nFrozen split: " + ", ".join(f"{name}={len(ids)} puzzles" for name, ids in split.items()))
    print("exp31: independent per-view capture chains + ARP + rembg-slop + bright rim + box-photo overview")

    train_ids_present = [pid for pid in split["train"] if (root / pid).exists()]

    datasets: dict[str, torch.utils.data.Dataset] = {}  # type: ignore[type-arg]
    for name, ids in split.items():
        present_ids = _present_ids(name, ids, root, allow_missing)
        if name == "train":
            datasets[name] = CapturePieceDataset(
                puzzle_ids=present_ids,
                augment_config=capture_config,
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


def _present_ids(name: str, ids: list[str], root: Path, allow_missing: bool) -> list[str]:
    """Filter a split's puzzle IDs to those present on disk.

    Args:
        name: Split name (for messages).
        ids: The split's puzzle IDs.
        root: Root of RGBA piece folders.
        allow_missing: Whether missing dirs are tolerated.

    Returns:
        The IDs that exist on disk (all of them when nothing is missing).

    Raises:
        ValueError: If dirs are missing and ``allow_missing`` is False.
    """
    missing = [pid for pid in ids if not (root / pid).exists()]
    if not missing:
        return ids
    message = f"{name}: {len(missing)}/{len(ids)} puzzle dirs missing under {root}"
    if not allow_missing:
        raise ValueError(
            f"{message}. exp31 reuses exp30's RGBA pieces — generate them with "
            "experiments.exp30_generator_fixes.generate_dataset, or pass allow_missing=True "
            "(train.py: --allow-missing-puzzles) for a smoke test (results will NOT be comparable)."
        )
    print(f"WARNING: {message} (allow_missing=True; results NOT comparable)")
    return [pid for pid in ids if (root / pid).exists()]


__all__ = [
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_PUZZLE_ROOT",
    "GRID_SIZE",
    "NUM_CELLS",
    "CapturePieceDataset",
    "FramedBlackCompositeTestDataset",
    "ViewPair",
    "create_datasets_from_split",
    "deterministic_seed",
    "piece_to_model_input",
    "puzzle_id_of",
    "view_pair",
]
