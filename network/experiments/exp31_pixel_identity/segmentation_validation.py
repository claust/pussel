#!/usr/bin/env python3
r"""Validate exp31's cheap segmentation-slop model against real rembg.

``capture.apply_segmentation_slop`` is an explicit, randomised model of what
rembg does to a piece's mask (boundary softness, tab-neck rounding, a couple
of pixels of slop). It exists because the honest alternative — running true
rembg on ~192k pieces every epoch — is far too slow. This script measures
whether the cheap model actually lands where rembg lands, on the three
statistics that matter, and reports the rembg throughput so the
"GPU pre-pass at generation time instead?" question is answered with a number
rather than a guess.

Three mask sources are compared on the *same* synthetic pieces:

- ``exact`` — the generator's Bezier lattice cut, untouched (the artifact
  being removed).
- ``cheap`` — ``apply_segmentation_slop`` (what exp31 trains on).
- ``rembg`` — real ``rembg``/u2net, the deployed path
  (``exp24_piece_classifier/build_positives.py``,
  ``exp25_north_star_eval/evaluate.py``), run on the piece composited over a
  surface so the segmenter has something to segment.

Metrics (medians reported, all at the piece's native scale):

- ``soft_px`` — mean boundary transition width in pixels: the count of
  partially transparent pixels divided by the silhouette perimeter. The exact
  cut is a step, so this is ~0 for ``exact``.
- ``alpha_grad`` — mean \\|grad alpha\\| over the transition band, the
  boundary-gradient softness in alpha units per pixel.
- ``area_ratio`` — mask area divided by the exact mask's area: >1 means the
  segmenter keeps background at the rim, <1 means it eats the piece.
- ``rim_ratio`` — boundary luminance divided by interior luminance on the
  black-composited crop, the section 5 metric (real 1.08, synthetic 0.98,
  exp26-augmented 0.69).

Run from the network/ directory:
    uv run python -m experiments.exp31_pixel_identity.segmentation_validation \\
        --sample 240 --dataset-root datasets/realistic_4x4_rgba_v2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from .capture import CaptureConfig, apply_scene_surface, apply_segmentation_slop

DEFAULT_DATASET_ROOT = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4_rgba_v2"

# Section 5's measured real values, for the comparison table's last column.
REAL_RIM_RATIO = 1.08
SYNTHETIC_RIM_RATIO = 0.98
EXP26_RIM_RATIO = 0.69


@dataclass
class MaskStats:
    """Boundary statistics for one mask realization."""

    soft_px: float
    alpha_grad: float
    area_ratio: float
    rim_ratio: float


def _perimeter(binary: np.ndarray) -> int:
    """Count mask pixels that touch a non-mask pixel.

    Args:
        binary: uint8 mask, nonzero = subject.

    Returns:
        The perimeter in pixels (at least 1).
    """
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8))
    return max(1, int((binary.astype(bool) & ~eroded.astype(bool)).sum()))


def mask_stats(alpha: np.ndarray, rgb: np.ndarray, exact_area: int) -> MaskStats | None:
    """Measure the four boundary statistics for one mask.

    Args:
        alpha: Float alpha in [0, 1], HxW.
        rgb: The matching RGB in [0, 255], HxWx3 (already the piece's colour,
            not yet black-composited — this function composites).
        exact_area: Pixel area of the exact generator mask, for ``area_ratio``.

    Returns:
        The statistics, or None when the mask is too small to measure.
    """
    binary = (alpha > 0.5).astype(np.uint8)
    if binary.sum() < 100 or exact_area <= 0:
        return None

    partial = (alpha > 0.05) & (alpha < 0.95)
    grad_y, grad_x = np.gradient(alpha)
    grad = np.sqrt(grad_x**2 + grad_y**2)

    luma = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) * alpha
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    edge = (distance > 0) & (distance <= 3)
    core = distance > 6
    if edge.sum() < 20 or core.sum() < 20:
        return None

    return MaskStats(
        soft_px=float(partial.sum()) / _perimeter(binary),
        alpha_grad=float(grad[partial].mean()) if partial.any() else 0.0,
        area_ratio=float(binary.sum()) / exact_area,
        rim_ratio=float(luma[edge].mean() / max(1e-6, luma[core].mean())),
    )


def _stats_from_rgba(rgba: Image.Image, exact_area: int) -> MaskStats | None:
    """Measure one RGBA piece.

    Args:
        rgba: RGBA piece.
        exact_area: Exact mask area for the ratio.

    Returns:
        The statistics, or None when unmeasurable.
    """
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
    return mask_stats(alpha, np.asarray(rgba.convert("RGB"), dtype=np.float32), exact_area)


def _rembg_runner() -> Callable[[Image.Image], Image.Image]:
    """Build the deployed rembg call.

    Returns:
        A callable turning an RGB image into rembg's RGBA output.
    """
    from rembg import new_session, remove  # local import: heavy, pulls onnxruntime

    session = new_session("u2net")

    def run(image: Image.Image) -> Image.Image:
        out = remove(image, session=session)
        assert isinstance(out, Image.Image)
        return out.convert("RGBA")

    return run


def sample_pieces(dataset_root: Path, sample: int, seed: int) -> list[Path]:
    """Pick pieces spread as evenly as possible over the available puzzles.

    Round-robin over shuffled puzzle directories, taking one more piece from
    each per pass, so a small dataset root (a smoke sample of 60 puzzles) still
    yields the requested number of pieces without over-weighting one puzzle.

    Args:
        dataset_root: Root of RGBA piece folders.
        sample: Number of pieces wanted.
        seed: RNG seed for the choice.

    Returns:
        The chosen piece paths (fewer than ``sample`` only when the root
        genuinely holds fewer pieces).
    """
    rng = random.Random(seed)
    puzzle_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir())
    rng.shuffle(puzzle_dirs)

    per_puzzle: list[list[Path]] = []
    for puzzle_dir in puzzle_dirs:
        pieces = sorted(puzzle_dir.glob("*.png"))
        if pieces:
            rng.shuffle(pieces)
            per_puzzle.append(pieces)

    chosen: list[Path] = []
    for depth in range(max((len(p) for p in per_puzzle), default=0)):
        for pieces in per_puzzle:
            if depth < len(pieces):
                chosen.append(pieces[depth])
                if len(chosen) >= sample:
                    return chosen
    return chosen


def _summarize(rows: list[MaskStats]) -> dict[str, float]:
    """Reduce per-piece statistics to medians and a spread.

    Args:
        rows: Per-piece statistics.

    Returns:
        Mapping of metric name to its median, plus n and the rim IQR bounds.
    """
    if not rows:
        return {"n": 0.0}
    return {
        "n": float(len(rows)),
        "soft_px": float(np.median([r.soft_px for r in rows])),
        "alpha_grad": float(np.median([r.alpha_grad for r in rows])),
        "area_ratio": float(np.median([r.area_ratio for r in rows])),
        "rim_ratio": float(np.median([r.rim_ratio for r in rows])),
        "rim_p25": float(np.percentile([r.rim_ratio for r in rows], 25)),
        "rim_p75": float(np.percentile([r.rim_ratio for r in rows], 75)),
    }


def collect(
    piece_paths: list[Path],
    config: CaptureConfig,
    use_rembg: bool,
    seed: int,
) -> dict[str, Any]:
    """Measure all three mask sources over a sample of pieces.

    Args:
        piece_paths: Pieces to measure.
        config: Capture config driving the cheap model.
        use_rembg: Whether to run real rembg (the slow branch).
        seed: Base RNG seed.

    Returns:
        Mapping with one summary per source plus timings.
    """
    runner = _rembg_runner() if use_rembg else None
    collected: dict[str, list[MaskStats]] = {"exact": [], "cheap": [], "rembg": []}
    rembg_seconds = 0.0

    for index, piece_path in enumerate(piece_paths):
        random.seed(seed + index)
        np.random.seed((seed + index) % (2**32))
        with Image.open(piece_path) as raw:
            stored = raw.convert("RGBA")

        # The photo-like input both models see: the piece on a surface, with
        # room around it. Same stage the training path runs.
        on_surface = apply_scene_surface(stored, config)
        exact_area = int((np.asarray(on_surface.getchannel("A")) > 128).sum())

        for name, rgba in (
            ("exact", on_surface),
            ("cheap", apply_segmentation_slop(on_surface, config)),
        ):
            stats = _stats_from_rgba(rgba, exact_area)
            if stats is not None:
                collected[name].append(stats)

        if runner is not None:
            started = time.perf_counter()
            segmented = runner(on_surface.convert("RGB"))
            rembg_seconds += time.perf_counter() - started
            # rembg decides its own RGB, but the piece content is the same;
            # keep its alpha over the surface image so the rim luminance is
            # measured on identical pixels.
            merged = on_surface.convert("RGB").convert("RGBA")
            merged.putalpha(segmented.getchannel("A"))
            stats = _stats_from_rgba(merged, exact_area)
            if stats is not None:
                collected["rembg"].append(stats)

    result: dict[str, Any] = {name: _summarize(rows) for name, rows in collected.items()}
    if runner is not None and piece_paths:
        result["rembg_seconds_per_piece"] = rembg_seconds / len(piece_paths)
    return result


def print_table(result: dict[str, Any]) -> None:
    """Print the comparison table.

    Args:
        result: Output of :func:`collect`.
    """
    header = f"{'source':10s} {'n':>4s} {'soft_px':>8s} {'alpha_grad':>11s} {'area_ratio':>11s} {'rim_ratio':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for name in ("exact", "cheap", "rembg"):
        summary = result.get(name, {})
        if not summary.get("n"):
            print(f"{name:10s} {'-':>4s} {'(not run)':>8s}")
            continue
        print(
            f"{name:10s} {int(summary['n']):4d} {summary['soft_px']:8.2f} {summary['alpha_grad']:11.4f} "
            f"{summary['area_ratio']:11.3f} {summary['rim_ratio']:10.3f}"
        )
    print(
        f"\nSection 5 reference rim_ratio: real {REAL_RIM_RATIO}, synthetic {SYNTHETIC_RIM_RATIO}, "
        f"exp26-augmented {EXP26_RIM_RATIO}"
    )
    if "rembg_seconds_per_piece" in result:
        per = result["rembg_seconds_per_piece"]
        print(f"rembg: {per * 1000:.0f} ms/piece on this machine -> {per * 191968 / 3600:.1f} h for a 192k pre-pass")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(description="Validate exp31's segmentation-slop model against real rembg")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="RGBA piece root")
    parser.add_argument("--sample", type=int, default=240, help="Number of pieces to measure")
    parser.add_argument("--seed", type=int, default=31, help="Base RNG seed")
    parser.add_argument("--no-rembg", action="store_true", help="Skip the real-rembg branch (fast, cheap model only)")
    parser.add_argument("--json-out", type=Path, default=None, help="Write the summary JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the validation.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code (0 on success, 1 when no pieces were found).
    """
    args = build_parser().parse_args(argv)
    if not args.dataset_root.is_dir():
        print(f"ERROR: no dataset root at {args.dataset_root}")
        return 1

    piece_paths = sample_pieces(args.dataset_root, args.sample, args.seed)
    if not piece_paths:
        print(f"ERROR: no pieces under {args.dataset_root}")
        return 1
    print(f"Measuring {len(piece_paths)} pieces from {args.dataset_root} (rembg={'off' if args.no_rembg else 'on'})")

    result = collect(piece_paths, CaptureConfig(), use_rembg=not args.no_rembg, seed=args.seed)
    print_table(result)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
