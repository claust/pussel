#!/usr/bin/env python3
r"""Acceptance probes for the exp30 synthetic-generator fixes.

Implements the cheap, training-free acceptance gate from
``docs/synthetic-dataset-realism.html`` §8, targeting the two label-leaking
generator bugs documented in §4.1 and §4.2. Every probe reports per rotation
label class (0/90/180/270) and returns a PASS/FAIL verdict; the process exit
code is 0 only when all probes pass, so this can gate CI or a retraining run.

Stages
------
``raw_png``
    The stored piece PNG, content = ``alpha > --alpha-threshold``, keyed by
    the rotation baked into the filename. One record per sampled piece
    (**unpaired**: each class sees different content).
``raw_rotated``
    The piece after the pipeline's own load-time 90-degree rotation, at
    native canvas scale, content = non-black. Every piece contributes one
    record per rotation label, so classes are **content-paired** — this is
    the §4.2 protocol ("measured on the same piece at all four rotations").
``model_input``
    The same view after the pipeline's deterministic eval transform
    (black composite -> rotate -> resize to ``--piece-size``), i.e. exactly
    what the network sees. Also content-paired.

Probes and thresholds
---------------------
**Border-touch** (§4.1) — tight-cropped pieces rotated with ``expand=False``
make "alpha touches all four canvas borders" a 100%-precision rule for
``rotation in {0, 180}``. The §8 condition is "≈ 0 *and* independent of the
rotation label", split here by stage: rotation-independence is gated on both
``raw_png`` and ``model_input`` (cross-class spread of ``P(all 4)`` ``< 0.02``
and of every single-edge fraction ``< 0.10``), while the absolute level is
gated on ``model_input`` only (``P(all 4) < 0.01`` and every single-edge
fraction ``< 0.10``). The model input is what the network sees and what the
real crops are compared against; a pipeline that stores tight crops and frames
them at load time is free of the shortcut even though its stored PNGs touch
every border in every class. The single-edge checks matter because partial
clipping leaks the label without touching all four.

**Sharpness** (§4.2) — ``expand=False`` bilinear rotation on a non-square
canvas puts 90/270 on half-pixel offsets, leaving them measurably blurrier
while 0/180 stay pixel-exact. Measured as masked mean ``|gradient|`` (central
differences, so the metric is itself rotation-invariant) and mean
``|Laplacian|`` over the eroded piece interior. Pass requires a per-class
max/min ratio ``< 1.05`` for both metrics on the two content-paired stages.
The documented gaps are 1.10 (``|gradient|``) and 1.48 (``|Laplacian|``).
The ``raw_png`` numbers are reported as INFO only: with one baked rotation
per piece the classes hold different content, and at a few hundred pieces
that noise (measured at ±5% and occasionally sign-flipped) is larger than
the effect. The square-canvas fraction is reported alongside them, since a
square canvas is the structural precondition for a lossless in-place 90°
rotation.

**Framing** (§5) — mask area fraction of the model input and content
bounding-box aspect ratio, per class. Only the mask area is gated (per-class
max/min ``< 1.10``); a rotation-dependent mask area is a zoom cue that
square-padding removes. The aspect ratio is reported as a geometric mean and
left ungated: a piece's aspect at 90° is the reciprocal of its aspect at 0°,
so whenever the source cells are non-square the class means differ by
construction — in real data as much as synthetic. The real reference (64
north_star eval crops) is a mask area of ~0.44, printed for comparison but
not gated: it is a generator target, not a leakage condition.

Usage (run from the ``network/`` directory)::

    # Baseline: the exp26 dataset. Expected to FAIL border-touch + sharpness.
    uv run python -m experiments.exp30_generator_fixes.probes --pipeline exp26

    # The fixed dataset, with a machine-readable dump for CI.
    uv run python -m experiments.exp30_generator_fixes.probes \\
        --pipeline exp30 --sample 400 --seed 0 --json probes_exp30.json

``--pipeline`` selects both the default dataset root and the eval load path
that defines the ``raw_rotated`` / ``model_input`` views. Pipeline modules are
imported lazily inside their branch, so ``--pipeline exp26`` works even when
exp30 is not implemented yet.
"""

import argparse
import importlib
import inspect
import io
import json
import random
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..exp20_realistic_pieces.dataset import ROTATION_ANGLES, parse_piece_filename

NETWORK_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOTS: dict[str, Path] = {
    "exp26": NETWORK_ROOT / "datasets" / "realistic_4x4_rgba",
    "exp30": NETWORK_ROOT / "datasets" / "realistic_4x4_rgba_v2",
}
DEFAULT_PUZZLE_ROOT = NETWORK_ROOT / "datasets" / "puzzles"

# Default thresholds (rationale in the module docstring).
MAX_ALL_BORDER_TOUCH = 0.01
MAX_BORDER_SPREAD = 0.02
MAX_EDGE_TOUCH = 0.10
MAX_SHARPNESS_RATIO = 1.05
MAX_FRAMING_RATIO = 1.10

# Median mask-area fraction of the 64 real north_star eval crops (doc §5).
REAL_MASK_AREA_REFERENCE = 0.44

EDGES = ("top", "bottom", "left", "right")
RAW_STAGE = "raw_png"
ROTATED_STAGE = "raw_rotated"
INPUT_STAGE = "model_input"
PAIRED_STAGES = (ROTATED_STAGE, INPUT_STAGE)

# (piece path, applied rotation index) -> (native-scale rotated RGB or None,
# model-input RGB); both HxWx3 float arrays in [0, 255].
PipelineAdapter = Callable[[Path, int], tuple[np.ndarray | None, np.ndarray]]


@dataclass
class ProbeVerdict:
    """Outcome of one probe.

    Attributes:
        name: Human-readable probe name.
        passed: Whether every gating condition held.
        reasons: One line per condition, prefixed PASS/FAIL/INFO.
    """

    name: str
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    def check(self, ok: bool, text: str) -> None:
        """Record a gating condition and fold it into ``passed``.

        Args:
            ok: Whether the condition held.
            text: Description of the condition and its measured value.
        """
        self.passed = self.passed and ok
        self.reasons.append(("PASS " if ok else "FAIL ") + text)

    def info(self, text: str) -> None:
        """Record a non-gating observation.

        Args:
            text: Description of the observation.
        """
        self.reasons.append("INFO " + text)


def _mean(values: list[float]) -> float:
    """Return the mean of ``values``, or NaN when empty."""
    return float(np.mean(values)) if values else float("nan")


def _ratio(values: list[float]) -> float:
    """Return max/min over the finite, positive entries of ``values``."""
    finite = [v for v in values if np.isfinite(v) and v > 0]
    if len(finite) < 2:
        return float("nan")
    return max(finite) / min(finite)


def _spread(values: list[float]) -> float:
    """Return max-min over the finite entries of ``values``."""
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    return max(finite) - min(finite)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render an aligned plain-text table.

    Args:
        headers: Column headers.
        rows: Row cells, already stringified.

    Returns:
        The rendered table, without a trailing newline.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = ["  ".join(h.rjust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    lines.extend("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(lines)


def _list_pieces(puzzle_dir: Path) -> list[tuple[str, Path, int]]:
    """List one puzzle's parseable piece PNGs.

    Args:
        puzzle_dir: Directory holding a single puzzle's pieces.

    Returns:
        List of ``(puzzle_id, piece_path, base_rotation)`` tuples.
    """
    entries: list[tuple[str, Path, int]] = []
    for piece_path in sorted(puzzle_dir.glob(f"{puzzle_dir.name}_x*_y*_rot*.png")):
        parsed = parse_piece_filename(piece_path.name)
        if parsed is not None:
            entries.append((puzzle_dir.name, piece_path, parsed[3]))
    return entries


def sample_pieces(dataset_root: Path, sample: int, seed: int) -> list[tuple[str, Path, int]]:
    """Draw a seeded, puzzle-stratified sample of stored piece PNGs.

    Puzzles are shuffled, then pieces are taken round-robin so the sample
    spreads over as many puzzles as possible.

    Args:
        dataset_root: Root holding one subdirectory per puzzle.
        sample: Number of pieces to draw.
        seed: RNG seed.

    Returns:
        List of ``(puzzle_id, piece_path, base_rotation)`` tuples.

    Raises:
        FileNotFoundError: If the root has no puzzle subdirectories.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    puzzle_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if not puzzle_dirs:
        raise FileNotFoundError(f"No puzzle subdirectories under {dataset_root}")

    rng = random.Random(seed)
    rng.shuffle(puzzle_dirs)
    if sample <= len(puzzle_dirs):
        puzzle_dirs = puzzle_dirs[:sample]

    per_puzzle: list[list[tuple[str, Path, int]]] = []
    for puzzle_dir in puzzle_dirs:
        entries = _list_pieces(puzzle_dir)
        if entries:
            rng.shuffle(entries)
            per_puzzle.append(entries)

    picked: list[tuple[str, Path, int]] = []
    depth = 0
    while len(picked) < sample:
        available = [entries for entries in per_puzzle if depth < len(entries)]
        if not available:
            break
        for entries in available:
            picked.append(entries[depth])
            if len(picked) >= sample:
                break
        depth += 1
    return picked


def erode_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a boolean mask with a 4-connected structuring element.

    Args:
        mask: Boolean HxW mask.
        iterations: Number of erosion passes.

    Returns:
        The eroded boolean mask (the canvas border is always cleared).
    """
    eroded = mask
    for _ in range(iterations):
        out = eroded.copy()
        out[1:, :] &= eroded[:-1, :]
        out[:-1, :] &= eroded[1:, :]
        out[:, 1:] &= eroded[:, :-1]
        out[:, :-1] &= eroded[:, 1:]
        out[0, :] = False
        out[-1, :] = False
        out[:, 0] = False
        out[:, -1] = False
        eroded = out
    return eroded


def edge_touch_flags(mask: np.ndarray) -> dict[str, bool]:
    """Return per-edge content-touch flags for a boolean mask."""
    return {
        "top": bool(mask[0].any()),
        "bottom": bool(mask[-1].any()),
        "left": bool(mask[:, 0].any()),
        "right": bool(mask[:, -1].any()),
    }


def mask_geometry(mask: np.ndarray) -> tuple[float, float]:
    """Return (mask area fraction, content bounding-box aspect ratio).

    Args:
        mask: Boolean HxW content mask.

    Returns:
        The fraction of the canvas covered by content and the width/height
        aspect ratio of the content bounding box (NaN when empty).
    """
    area = float(mask.mean())
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return area, float("nan")
    height = float(rows[-1] - rows[0] + 1)
    width = float(cols[-1] - cols[0] + 1)
    return area, width / height


def sharpness_metrics(rgb: np.ndarray, mask: np.ndarray, erode_iterations: int) -> tuple[float, float]:
    """Return masked mean ``|gradient|`` and mean ``|Laplacian|``.

    Both are computed on the luminance of ``rgb`` over the eroded interior of
    ``mask``, so the piece silhouette itself never contributes an edge. Both
    operators are central-difference (symmetric), which makes the metrics
    invariant to a lossless 90/180-degree rotation of the input — any
    per-class difference therefore comes from resampling, not from the
    operator.

    Args:
        rgb: HxWx3 float array in [0, 255].
        mask: Boolean HxW content mask.
        erode_iterations: Erosion passes applied to ``mask`` first.

    Returns:
        Tuple of (mean gradient magnitude, mean absolute Laplacian); both NaN
        when the eroded interior is too small to measure.
    """
    interior = erode_mask(mask, erode_iterations)
    if int(interior.sum()) < 64:
        return float("nan"), float("nan")

    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    dy, dx = np.gradient(gray)
    gradient = np.sqrt(dx * dx + dy * dy)

    laplacian = np.zeros_like(gray)
    laplacian[1:-1, 1:-1] = 4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return float(gradient[interior].mean()), float(np.abs(laplacian)[interior].mean())


def _to_array(image: Any) -> np.ndarray:
    """Convert a PIL image or CHW tensor to an HxWx3 float array in [0, 255]."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.float32)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] < array.shape[2]:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if float(array.max(initial=0.0)) <= 1.0:
        array = array * 255.0
    return array


def build_pipeline_adapter(pipeline: str, dataset_root: Path, puzzle_root: Path, piece_size: int) -> PipelineAdapter:
    """Build the callable that reproduces a pipeline's deterministic eval view.

    Args:
        pipeline: ``"exp26"`` or ``"exp30"``.
        dataset_root: Root of stored piece PNGs.
        puzzle_root: Root of source puzzle JPEGs (unused by the probes, but
            required by the dataset constructors).
        piece_size: Square model-input size.

    Returns:
        A callable mapping ``(piece_path, applied_rotation_idx)`` to the
        native-scale rotated view (or None when the pipeline only exposes a
        model-input hook) and the model-input view.

    Raises:
        ValueError: If ``pipeline`` is unknown.
    """
    if pipeline == "exp26":
        return _build_exp26_adapter(dataset_root, puzzle_root, piece_size)
    if pipeline == "exp30":
        return _build_exp30_adapter(dataset_root, puzzle_root, piece_size)
    raise ValueError(f"Unknown pipeline: {pipeline}")


def _adapter_from_dataset(dataset: Any) -> PipelineAdapter:
    """Wrap an eval dataset following the exp20/exp26 piece contract.

    Args:
        dataset: Object exposing ``_load_piece(path)``,
            ``_rotate_piece(image, rotation_idx)`` and ``piece_transform``.

    Returns:
        The pipeline adapter.
    """

    def adapter(piece_path: Path, applied_rotation_idx: int) -> tuple[np.ndarray | None, np.ndarray]:
        piece = dataset._load_piece(piece_path)
        piece = dataset._rotate_piece(piece, applied_rotation_idx)
        return _to_array(piece), _to_array(dataset.piece_transform(piece))

    return adapter


def _build_exp26_adapter(dataset_root: Path, puzzle_root: Path, piece_size: int) -> PipelineAdapter:
    """Build the exp26 eval view: black composite, rotate, resize to square."""
    from ..exp26_domain_randomization.aug_dataset import BlackCompositeTestDataset

    with redirect_stdout(io.StringIO()):
        dataset = BlackCompositeTestDataset(
            puzzle_ids=[],
            dataset_root=dataset_root,
            puzzle_root=puzzle_root,
            piece_size=piece_size,
        )
    return _adapter_from_dataset(dataset)


def _build_exp30_adapter(dataset_root: Path, puzzle_root: Path, piece_size: int) -> PipelineAdapter:
    """Build the exp30 eval view, adapting defensively to its dataset API.

    Looks for an eval dataset class following the exp20/exp26 contract
    (``_load_piece`` + ``_rotate_piece`` + ``piece_transform``) first, then
    falls back to a module-level ``piece_to_model_input(path, rotation_idx)``
    hook (which forfeits the native-scale ``raw_rotated`` stage).

    Args:
        dataset_root: Root of stored piece PNGs.
        puzzle_root: Root of source puzzle JPEGs.
        piece_size: Square model-input size.

    Returns:
        The pipeline adapter.

    Raises:
        RuntimeError: If no compatible entry point is found, naming exactly
            what was searched for.
    """
    module_names = ("framed_dataset", "dataset", "aug_dataset", "piece_dataset", "datasets")
    class_names = (
        "FramedBlackCompositeTestDataset",
        "BlackCompositeTestDataset",
        "PaddedPieceTestDataset",
        "SquarePaddedTestDataset",
        "FixedPieceTestDataset",
        "PieceTestDataset",
    )
    hook_names = ("piece_to_model_input", "load_model_input")
    searched: list[str] = []
    for module_name in module_names:
        qualified = f"{__package__}.{module_name}"
        try:
            module = importlib.import_module(qualified)
        except ImportError:
            searched.append(f"{qualified} (not importable)")
            continue
        for class_name in class_names:
            dataset_cls = getattr(module, class_name, None)
            if dataset_cls is not None:
                return _adapter_from_exp30_class(
                    dataset_cls, qualified, class_name, dataset_root, puzzle_root, piece_size
                )
        for hook_name in hook_names:
            hook = getattr(module, hook_name, None)
            if callable(hook):
                print(f"exp30 adapter: using {qualified}.{hook_name}; the '{ROTATED_STAGE}' stage is unavailable")
                return _adapter_from_hook(hook, piece_size)
        searched.append(f"{qualified} (no {class_names} class, no {hook_names} hook)")
    raise RuntimeError(
        "exp30 probe adapter: could not find the eval load path. Searched: "
        + "; ".join(searched)
        + ". Expose either an eval dataset class with _load_piece(path), _rotate_piece(image, rotation_idx) "
        "and a piece_transform, or a module-level piece_to_model_input(piece_path, applied_rotation_idx) "
        "returning an image/array, then re-run this probe."
    )


def _adapter_from_hook(hook: Any, piece_size: int) -> PipelineAdapter:
    """Wrap a module-level ``piece_to_model_input`` style hook.

    The optional third parameter (``piece_size``) is passed only when the
    hook declares it, so both signatures are supported.

    Args:
        hook: Callable taking ``(piece_path, applied_rotation_idx)`` and
            optionally ``piece_size``, returning an image or array.
        piece_size: Square model-input size.

    Returns:
        The pipeline adapter; its native-scale view is always None.
    """
    try:
        takes_size = len(inspect.signature(hook).parameters) >= 3
    except (TypeError, ValueError):
        takes_size = False

    def adapter(piece_path: Path, applied_rotation_idx: int) -> tuple[np.ndarray | None, np.ndarray]:
        view = (
            hook(piece_path, applied_rotation_idx, piece_size) if takes_size else hook(piece_path, applied_rotation_idx)
        )
        return None, _to_array(view)

    return adapter


def _adapter_from_exp30_class(
    dataset_cls: Any,
    qualified: str,
    class_name: str,
    dataset_root: Path,
    puzzle_root: Path,
    piece_size: int,
) -> PipelineAdapter:
    """Instantiate an exp30 eval dataset and wrap its piece branch.

    Args:
        dataset_cls: The dataset class object.
        qualified: Fully qualified module name (for error messages).
        class_name: The class name (for error messages).
        dataset_root: Root of stored piece PNGs.
        puzzle_root: Root of source puzzle JPEGs.
        piece_size: Square model-input size.

    Returns:
        The pipeline adapter.

    Raises:
        RuntimeError: If the class cannot be constructed or does not follow
            the expected eval contract.
    """
    try:
        with redirect_stdout(io.StringIO()):
            dataset = dataset_cls(
                puzzle_ids=[],
                dataset_root=dataset_root,
                puzzle_root=puzzle_root,
                piece_size=piece_size,
            )
    except Exception as exc:
        raise RuntimeError(
            f"exp30 probe adapter: {qualified}.{class_name} could not be constructed with "
            f"(puzzle_ids=[], dataset_root=..., puzzle_root=..., piece_size={piece_size}): {exc!r}. "
            "Update _build_exp30_adapter in probes.py to match the new signature."
        ) from exc
    missing = [name for name in ("_load_piece", "_rotate_piece", "piece_transform") if not hasattr(dataset, name)]
    if missing:
        raise RuntimeError(
            f"exp30 probe adapter: {qualified}.{class_name} is missing {missing}; the probe needs the "
            "exp20/exp26 eval contract (_load_piece, _rotate_piece, piece_transform) or a module-level "
            "piece_to_model_input hook."
        )
    return _adapter_from_dataset(dataset)


def _measure(rgb: np.ndarray, mask: np.ndarray, erode_iterations: int) -> dict[str, Any]:
    """Build the common per-view measurement record."""
    gradient, laplacian = sharpness_metrics(rgb, mask, erode_iterations)
    area, aspect = mask_geometry(mask)
    flags = edge_touch_flags(mask)
    record: dict[str, Any] = {
        "touch_all": all(flags.values()),
        "gradient": gradient,
        "laplacian": laplacian,
        "mask_area": area,
        "aspect": aspect,
    }
    record.update({f"touch_{edge}": flag for edge, flag in flags.items()})
    return record


def collect_raw_records(
    samples: list[tuple[str, Path, int]], alpha_threshold: int, erode_iterations: int
) -> list[dict[str, Any]]:
    """Measure the raw stored PNGs, keyed by the baked-in rotation.

    Args:
        samples: ``(puzzle_id, piece_path, base_rotation)`` tuples.
        alpha_threshold: Alpha strictly above this counts as content. The
            default of 0 matches the tight ``getbbox()`` crop that creates
            the §4.1 shortcut.
        erode_iterations: Erosion passes before the sharpness measurement.

    Returns:
        One record per piece.
    """
    records: list[dict[str, Any]] = []
    for _, piece_path, base_rotation in samples:
        with Image.open(piece_path) as opened:
            rgba = np.asarray(opened.convert("RGBA"), dtype=np.float32)
        mask = rgba[..., 3] > alpha_threshold
        if not mask.any():
            continue
        record = _measure(rgba[..., :3], mask, erode_iterations)
        record.update(
            {
                "rotation": base_rotation,
                "path": str(piece_path),
                "width": int(rgba.shape[1]),
                "height": int(rgba.shape[0]),
                "square_canvas": int(rgba.shape[1]) == int(rgba.shape[0]),
            }
        )
        records.append(record)
    return records


def collect_paired_records(
    samples: list[tuple[str, Path, int]],
    adapter: PipelineAdapter,
    content_threshold: float,
    erode_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure the pipeline views at all four applied rotations.

    Every piece contributes one record per rotation label, so the per-class
    comparisons are content-paired.

    Args:
        samples: ``(puzzle_id, piece_path, base_rotation)`` tuples.
        adapter: Pipeline eval-view adapter.
        content_threshold: Max-channel value strictly above this counts as
            content (both views are composited on black).
        erode_iterations: Erosion passes before the sharpness measurement.

    Returns:
        Tuple of (native-scale rotated records, model-input records), keyed
        by the total rotation label the model is trained against.
    """
    rotated_records: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for _, piece_path, base_rotation in samples:
        for applied_idx in range(4):
            rotated, model_input = adapter(piece_path, applied_idx)
            common = {
                "rotation": (base_rotation + ROTATION_ANGLES[applied_idx]) % 360,
                "path": str(piece_path),
                "applied_rotation": ROTATION_ANGLES[applied_idx],
            }
            for view, sink in ((rotated, rotated_records), (model_input, input_records)):
                if view is None:
                    continue
                mask = view.max(axis=2) > content_threshold
                if not mask.any():
                    continue
                record = _measure(view, mask, erode_iterations)
                record.update(common)
                sink.append(record)
    return rotated_records, input_records


def _group_by_rotation(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Group records by their rotation label."""
    return {rotation: [r for r in records if r["rotation"] == rotation] for rotation in ROTATION_ANGLES}


def border_probe(
    stages: dict[str, list[dict[str, Any]]], thresholds: dict[str, float]
) -> tuple[ProbeVerdict, dict[str, Any]]:
    """Run the border-touch probe on the raw and model-input stages.

    Args:
        stages: Mapping of stage name to its records.
        thresholds: Threshold overrides (``all_touch``, ``spread``, ``edge``).

    Returns:
        The verdict and a JSON-serializable stats dict.
    """
    verdict = ProbeVerdict(name="border-touch")
    stats: dict[str, Any] = {}
    for stage in (RAW_STAGE, INPUT_STAGE):
        records = stages[stage]
        if not records:
            verdict.info(f"{stage}: no records, skipped")
            continue
        grouped = _group_by_rotation(records)
        rows: list[list[str]] = []
        per_class: dict[str, Any] = {}
        for rotation, group in grouped.items():
            fractions = {edge: _mean([float(r[f"touch_{edge}"]) for r in group]) for edge in EDGES}
            all_four = _mean([float(r["touch_all"]) for r in group])
            per_class[str(rotation)] = {"n": len(group), "all4": all_four, **fractions}
            rows.append(
                [f"{rotation}", f"{len(group)}", f"{all_four:.3f}"] + [f"{fractions[edge]:.3f}" for edge in EDGES]
            )
        print(f"\n[border-touch] stage: {stage}")
        print(format_table(["rot", "n", "P(all 4)", *EDGES], rows))

        all_values = [per_class[str(rotation)]["all4"] for rotation in ROTATION_ANGLES]
        edge_values = {edge: [per_class[str(rotation)][edge] for rotation in ROTATION_ANGLES] for edge in EDGES}
        max_all = max(all_values)
        all_spread = _spread(all_values)
        max_edge = max(max(values) for values in edge_values.values())
        edge_spread = max(_spread(values) for values in edge_values.values())
        # Rotation-independence is gated at both stages: it is the leak. The
        # absolute level is gated only on the model input, because that is
        # what the network sees and what the real crops are compared against
        # -- a pipeline that stores tight crops and frames them at load time
        # is free of the shortcut even though its stored PNGs touch every
        # border in every class.
        verdict.check(
            all_spread < thresholds["spread"], f"{stage}: P(all 4) spread={all_spread:.3f} < {thresholds['spread']}"
        )
        verdict.check(
            edge_spread < thresholds["edge"], f"{stage}: single-edge spread={edge_spread:.3f} < {thresholds['edge']}"
        )
        if stage == INPUT_STAGE:
            verdict.check(
                max_all < thresholds["all_touch"], f"{stage}: max P(all 4)={max_all:.3f} < {thresholds['all_touch']}"
            )
            verdict.check(
                max_edge < thresholds["edge"], f"{stage}: max single-edge touch={max_edge:.3f} < {thresholds['edge']}"
            )
        else:
            verdict.info(
                f"{stage}: max P(all 4)={max_all:.3f}, max single-edge touch={max_edge:.3f} "
                "- absolute level not gating on stored PNGs, only rotation-independence is"
            )
        stats[stage] = {
            "per_class": per_class,
            "max_all4": max_all,
            "all4_spread": all_spread,
            "max_edge": max_edge,
            "edge_spread": edge_spread,
        }
    return verdict, stats


def _sharpness_stage_stats(grouped: dict[int, list[dict[str, Any]]], metric: str) -> dict[str, Any]:
    """Summarize one sharpness metric across the four rotation classes."""
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for rotation, group in grouped.items():
        values = [r[metric] for r in group if np.isfinite(r[metric])]
        means[str(rotation)] = _mean(values)
        counts[str(rotation)] = len(values)
    even = _mean([means["0"], means["180"]])
    odd = _mean([means["90"], means["270"]])
    return {
        "per_class": means,
        "counts": counts,
        "even_0_180": even,
        "odd_90_270": odd,
        "even_odd_ratio": _ratio([even, odd]),
        "class_ratio": _ratio(list(means.values())),
    }


def _print_sharpness_stage(stage: str, metric: str, summary: dict[str, Any]) -> None:
    """Print the per-class table and ratios for one stage/metric."""
    rows = [
        [f"{rotation}", f"{summary['counts'][str(rotation)]}", f"{summary['per_class'][str(rotation)]:.4f}"]
        for rotation in ROTATION_ANGLES
    ]
    print(f"\n[sharpness] stage: {stage} · metric: mean |{metric}|")
    print(format_table(["rot", "n", f"mean |{metric}|"], rows))
    print(
        f"  0/180 = {summary['even_0_180']:.4f}   90/270 = {summary['odd_90_270']:.4f}   "
        f"even/odd ratio = {summary['even_odd_ratio']:.4f}   per-class max/min = {summary['class_ratio']:.4f}"
    )


def sharpness_probe(stages: dict[str, list[dict[str, Any]]], max_ratio: float) -> tuple[ProbeVerdict, dict[str, Any]]:
    """Run the sharpness probe on every available stage.

    The two content-paired stages gate the verdict; the unpaired ``raw_png``
    stage is reported as INFO because its classes hold different content.

    Args:
        stages: Mapping of stage name to its records.
        max_ratio: Maximum tolerated per-class max/min ratio.

    Returns:
        The verdict and a JSON-serializable stats dict.
    """
    verdict = ProbeVerdict(name="sharpness")
    stats: dict[str, Any] = {}
    for stage, records in stages.items():
        if not records:
            verdict.info(f"{stage}: no records, skipped")
            continue
        grouped = _group_by_rotation(records)
        stage_stats: dict[str, Any] = {}
        for metric in ("gradient", "laplacian"):
            summary = _sharpness_stage_stats(grouped, metric)
            _print_sharpness_stage(stage, metric, summary)
            stage_stats[metric] = summary
            ratio = summary["class_ratio"]
            text = f"{stage}/{metric}: per-class max/min={ratio:.4f} (limit {max_ratio})"
            if stage in PAIRED_STAGES:
                verdict.check(bool(np.isfinite(ratio)) and ratio < max_ratio, text)
            else:
                verdict.info(text + " - unpaired, content-confounded, not gating")
        stats[stage] = stage_stats

    raw_records = stages.get(RAW_STAGE, [])
    if raw_records:
        square = _mean([float(r["square_canvas"]) for r in raw_records])
        print(f"\n[sharpness] stored canvases square: {square:.3f} of sampled pieces")
        verdict.info(
            f"{RAW_STAGE}: square-canvas fraction={square:.3f} "
            "(square canvas = precondition for a lossless in-place 90 deg rotation)"
        )
        stats.setdefault(RAW_STAGE, {})["square_canvas_fraction"] = square
    return verdict, stats


def framing_probe(records: list[dict[str, Any]], max_ratio: float) -> tuple[ProbeVerdict, dict[str, Any]]:
    """Run the framing probe on the model-input stage.

    Args:
        records: Model-input records (content-paired across rotations).
        max_ratio: Maximum tolerated per-class max/min ratio.

    Returns:
        The verdict and a JSON-serializable stats dict.
    """
    verdict = ProbeVerdict(name="framing")
    if not records:
        verdict.info(f"{INPUT_STAGE}: no records, skipped")
        return verdict, {}

    grouped = _group_by_rotation(records)
    areas: dict[str, float] = {}
    aspects: dict[str, float] = {}
    rows: list[list[str]] = []
    for rotation, group in grouped.items():
        areas[str(rotation)] = _mean([r["mask_area"] for r in group])
        # Geometric mean: a piece's aspect at 90 degrees is the reciprocal of
        # its aspect at 0 degrees, so only a log-space average makes the two
        # classes comparable.
        logs = [float(np.log(r["aspect"])) for r in group if np.isfinite(r["aspect"]) and r["aspect"] > 0]
        aspects[str(rotation)] = float(np.exp(_mean(logs)))
        rows.append([f"{rotation}", f"{len(group)}", f"{areas[str(rotation)]:.4f}", f"{aspects[str(rotation)]:.4f}"])
    print(f"\n[framing] stage: {INPUT_STAGE}")
    print(format_table(["rot", "n", "mask area frac", "aspect (geom)"], rows))
    overall_area = _mean([r["mask_area"] for r in records])
    aspect_ratio = _ratio(list(aspects.values()))
    print(
        f"  overall mask area = {overall_area:.4f}; real north_star reference "
        f"~= {REAL_MASK_AREA_REFERENCE:.2f} (doc section 5, not gated)"
    )
    print(f"  aspect max/min = {aspect_ratio:.4f} (not gated: reciprocal by construction on non-square cells)")

    ratio = _ratio(list(areas.values()))
    verdict.check(
        bool(np.isfinite(ratio)) and ratio < max_ratio, f"{INPUT_STAGE}/mask area: max/min={ratio:.4f} < {max_ratio}"
    )
    verdict.info(
        f"{INPUT_STAGE}/aspect: max/min={aspect_ratio:.4f} - not gating, the 0 and 90 degree class means are "
        "reciprocals whenever the source cells are non-square, in real data as much as synthetic"
    )
    verdict.info(f"{INPUT_STAGE}: overall mask area={overall_area:.4f} vs real {REAL_MASK_AREA_REFERENCE:.2f}")
    return verdict, {
        "per_class_mask_area": areas,
        "per_class_aspect_geomean": aspects,
        "mask_area_ratio": ratio,
        "aspect_ratio": aspect_ratio,
        "overall_mask_area": overall_area,
        "real_mask_area_reference": REAL_MASK_AREA_REFERENCE,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Acceptance probes for the exp30 generator fixes (doc sections 4.1, 4.2, 5, 8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pipeline", choices=sorted(DATASET_ROOTS), default="exp30", help="Which load path to probe")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Piece root (default: per-pipeline)")
    parser.add_argument("--puzzle-root", type=Path, default=DEFAULT_PUZZLE_ROOT, help="Source puzzle JPEG root")
    parser.add_argument("--sample", type=int, default=400, help="Pieces to sample (stratified over puzzles)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--piece-size", type=int, default=128, help="Model input size")
    parser.add_argument("--alpha-threshold", type=int, default=0, help="Raw stage: alpha > this is content")
    parser.add_argument("--content-threshold", type=float, default=8.0, help="Composited stages: max channel > this")
    parser.add_argument("--erode", type=int, default=2, help="Mask erosion passes before sharpness")
    parser.add_argument("--max-border-touch", type=float, default=MAX_ALL_BORDER_TOUCH, help="Max P(all 4 borders)")
    parser.add_argument("--max-border-spread", type=float, default=MAX_BORDER_SPREAD, help="Max cross-class spread")
    parser.add_argument("--max-edge-touch", type=float, default=MAX_EDGE_TOUCH, help="Max single-edge touch fraction")
    parser.add_argument("--max-sharpness-ratio", type=float, default=MAX_SHARPNESS_RATIO, help="Max sharpness ratio")
    parser.add_argument("--max-framing-ratio", type=float, default=MAX_FRAMING_RATIO, help="Max framing ratio")
    parser.add_argument("--json", type=Path, default=None, help="Also write machine-readable results here")
    return parser


def _dump_json(path: Path, args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Write the machine-readable result file.

    Args:
        path: Destination path.
        args: Parsed arguments (recorded as the run config).
        payload: Probe results and verdicts.
    """
    payload["config"] = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items() if key != "json"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    """Run every probe and return the process exit code.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 when every probe passes, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    args.dataset_root = args.dataset_root if args.dataset_root is not None else DATASET_ROOTS[args.pipeline]

    samples = sample_pieces(args.dataset_root, args.sample, args.seed)
    adapter = build_pipeline_adapter(args.pipeline, args.dataset_root, args.puzzle_root, args.piece_size)

    print(f"pipeline={args.pipeline}  root={args.dataset_root}")
    print(f"sampled {len(samples)} pieces from {len({s[0] for s in samples})} puzzles (seed={args.seed})")

    raw_records = collect_raw_records(samples, args.alpha_threshold, args.erode)
    rotated_records, input_records = collect_paired_records(samples, adapter, args.content_threshold, args.erode)
    stages = {RAW_STAGE: raw_records, ROTATED_STAGE: rotated_records, INPUT_STAGE: input_records}

    border_verdict, border_stats = border_probe(
        stages,
        {"all_touch": args.max_border_touch, "spread": args.max_border_spread, "edge": args.max_edge_touch},
    )
    sharp_verdict, sharp_stats = sharpness_probe(stages, args.max_sharpness_ratio)
    framing_verdict, framing_stats = framing_probe(input_records, args.max_framing_ratio)

    verdicts = [border_verdict, sharp_verdict, framing_verdict]
    print("\n=== verdicts ===")
    for verdict in verdicts:
        print(f"{verdict.name}: {'PASS' if verdict.passed else 'FAIL'}")
        for reason in verdict.reasons:
            print(f"    {reason}")
    overall = all(v.passed for v in verdicts)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")

    if args.json is not None:
        _dump_json(
            args.json,
            args,
            {
                "sampled_pieces": len(samples),
                "probes": {
                    "border_touch": {"passed": border_verdict.passed, "stats": border_stats},
                    "sharpness": {"passed": sharp_verdict.passed, "stats": sharp_stats},
                    "framing": {"passed": framing_verdict.passed, "stats": framing_stats},
                },
                "reasons": [reason for verdict in verdicts for reason in verdict.reasons],
                "overall_passed": overall,
            },
        )

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
