#!/usr/bin/env python3
r"""NCC-headroom acceptance gate for the exp31 piece<->overview pixel-identity fix.

Reproduces the §4.3 measurement of ``docs/synthetic-dataset-realism.html`` and
turns it into a PASS/FAIL gate that must be cleared **before** any exp31
retraining. The shortcut it targets is structural: the synthetic "overview" is
the byte-identical source JPEG the pieces were cut from, so a synthetic piece
correlates near-perfectly with its ground-truth cell while a real photographed
piece does not. §8 lists the pass condition as "median masked NCC at the
ground-truth location drops from 0.99 toward the real 0.73".

What is measured
----------------
For every sampled piece, in the view the pipeline under test actually produces:

``gt``
    The best masked zero-mean NCC over a ``--scales`` x ``--rotations`` grid,
    restricted to template placements whose centre falls inside the **ground
    truth** cell of the overview.
``decoy``
    The identical search restricted to one randomly chosen **wrong** cell.

The decoy is not decoration. The goal is to remove pixel-identity *headroom*
without destroying the true-vs-decoy *contrast*: a change that drags the GT
score down to the decoy level has made the data unlearnable, not realistic.
Both numbers come from the same response maps, so they cannot drift apart for
implementation reasons.

Protocol details (identical for every domain)
--------------------------------------------
* **Alignment.** The discrete 90-degree relationship between piece and overview
  is resolved losslessly first (``exp30.framing.rotate_lossless`` for synthetic
  base rotations, ``np.rot90`` for the north_star ``rotation`` column). The
  probe therefore measures appearance/structure headroom only — the rotation
  label leaks of §4.1/§4.2 are already gated by
  ``exp30_generator_fixes/probes.py`` and must not be re-measured here.
* **Mask.** The piece's own silhouette: ``alpha > --alpha-threshold`` for RGBA
  sources, ``max(R,G,B) > --content-threshold`` for black-composited ones, then
  eroded by ``--erode`` pixels. Zero-mean subtraction and contrast
  normalisation happen **inside the mask only**, per channel, on both sides of
  the correlation. The erosion matters: with the raw silhouette included, the
  antialiased piece boundary (synthetic) and the bright die-cut cardboard rim
  (real) dominate the score. At ``--erode 2`` a raw synthetic piece scores
  exactly 1.000 at its true location, reproducing the doc's "0 deg/180 deg
  exactly 1.000".
* **Template.** Cropped to the mask bounding box, so a corner piece can reach
  its true (corner-aligned) placement — a padded square canvas cannot.
* **Scale grid.** ``--scales`` log-spaced linear factors over
  ``[1/--scale-span, --scale-span]``, applied to a per-domain *reference area*:
  the stored piece's own native mask area for synthetic domains (exact, because
  synthetic pieces are literally cut from the overview) and the overview cell
  area for real photos (physical pieces tile the puzzle, so piece area ~= cell
  area). Normalising on **area** rather than bounding-box side makes the grid
  invariant to the pipeline's own rotation jitter, and re-deriving the factor
  from the *current* mask area means an intentional piece/overview resolution
  asymmetry (Test 2) is undone by the search instead of being scored as if the
  matcher could not solve scale. The default is **9 steps over +-25%**, not the
  doc's 7: exp26's own ``scale_min/scale_max`` of 0.85/1.15 means the search has
  to undo up to a 1.18x factor, which a 7-step +-15% grid cannot reach — it left
  42% of exp30's winning candidates pinned to the grid edge. 9 steps over +-25%
  keeps the same step resolution (1.057) and drops that to 11%. ``--scales 7
  --scale-span 1.15`` restores the doc's exact grid.
* **Rotation grid.** ``--rotations`` angles over
  ``[---rotation-span, +--rotation-span]``; the 9-degree default covers exp26's
  ``rotation_jitter_deg`` of 8.0.
* The fraction of winning candidates pinned to each end of each grid is
  reported. A high **high-side** fraction means the grid is too narrow and the
  numbers understate the true headroom — read it before trusting a PASS. A high
  low-side fraction is expected on real photos (~34%): a masked NCC maximum is
  biased towards smaller templates, and the rembg mask over-covers the piece's
  true footprint, so the search leans on the small end of the grid.

Thresholds and their justification
----------------------------------
The gate is defined against the **real** domain measured by *this* tool at its
default settings (``--reference measured``), not against the doc's table, so
that ``--pipeline real`` passes its own gate — a reference the reference domain
fails is not a gate. ``--reference doc`` swaps in the published numbers and
``--reference-json`` accepts a ``--json`` dump from a previous
``--pipeline real`` run, which is the right choice whenever the real data or
the protocol flags change.

Every number below was measured with this file at ``--sample 200 --seed 0`` and
otherwise-default flags (see BASELINES / REFERENCES in the code):

===========================  =========  =========  =========  ======
domain                       GT median  frac >0.8  decoy med  margin
===========================  =========  =========  =========  ======
exp20 (raw, no aug)              1.000      0.990      0.435   0.565
exp26 (full exp26 aug)           0.933      0.830      0.432   0.501
exp30 (aug + generator fixes)    0.937      0.840      0.453   0.484
real (north_star)                0.679      0.340      0.303   0.377
doc section 4.3, real            0.730      0.410      0.407   0.323
===========================  =========  =========  =========  ======

Re-running the real domain at seeds 1 and 2 gives GT medians 0.688 and 0.710
and >0.8 fractions 0.325 and 0.330, so the reference is stable to ~0.03.

1. **GT median in ``real +- 0.10``** -> ``[0.579, 0.779]``. The point of Test 2
   is to move this metric from "template lookup" to "photograph of a different
   print". The upper bound is set so that no pipeline still carrying the
   shortcut can pass: raw synthetic measures 1.000 and full exp26/exp30
   augmentation 0.933-0.937, both far above 0.779. The lower bound is the
   anti-vandalism guard — a pipeline that blurred the piece into noise would
   otherwise "pass" the headroom test. +-0.10 is ~3x the seed-to-seed spread of
   the reference (0.031) and ~5x the shift from a protocol variant
   (``--erode 1`` moves real by 0.017), so it is wide enough not to gate on
   measurement noise and narrow enough to exclude every known-bad pipeline.
2. **Fraction > ``--high-ncc`` (0.8) in ``real +- 0.12``** -> ``[0.220, 0.460]``.
   This is the tail metric §8 cares about: the median can look reasonable while
   a large minority of pieces still match their source pixel-for-pixel. +-0.12
   excludes the doc's 56%/84% as well as this tool's starker 0.830-0.990 for
   exp20/exp26/exp30, and the binomial standard error at ``--sample 200`` is
   ~0.033, so the band is ~3.5 sigma wide.
3. **GT - decoy margin >= ``real - 0.10``** (``>= 0.277``) **and
   ``<= real + 0.15``** (``<= 0.527``). The lower bound is the load-bearing
   one: it is the "did we destroy the signal" check, and 0.10 matches the GT
   median tolerance because the decoy median is at least as stable as the GT
   median across seeds (0.303/0.320/0.333). The upper bound is a softer "the
   data is still easier than reality" flag — raw synthetic sits at 0.565 and
   fails it — and is deliberately looser (0.15) because a margin above the real
   one is a realism gap rather than a correctness bug, and §8's classical-parity
   metric is the sharper instrument for it. Note that exp26 and exp30 *pass* the
   margin bound while failing the first two: their problem is headroom, not
   contrast, which is exactly the distinction this gate has to draw.

The decoy median itself is reported but not gated: a decoy that climbs towards
the GT score is caught by the margin bound, and a decoy that collapses is not a
defect.

Known deltas from the doc's table
---------------------------------
Reproduced: the ordering and the size of the gap, and — closely — two of the
three rows. exp26 measures **0.933** here against the doc's 0.937, and real
measures 0.679 against the doc's 0.730 (0.688/0.710 at other seeds).

Differs: (a) raw synthetic measures **1.000**, not 0.990, and its >0.8 fraction
is 0.990, not 0.56. The doc's mask included the antialiased silhouette boundary,
where the stored piece's RGB is blended toward the transparent fill; that
boundary — not the interior — is what pulled its 90 deg/270 deg pieces below
0.8. With ``--alpha-threshold 254`` and ``--erode 2`` the interior is
byte-identical to the source at *every* base rotation, which is the *stronger*
form of the same finding: the pixel identity is total, not merely high. Verified
directly — at ``--alpha-threshold 254`` a single piece scores exactly 1.000000,
and relaxing the threshold to 127 drops it to 0.99 and to 0 (no erosion,
``alpha > 0``) to 0.76-0.92.
(b) The decoy medians here are ~0.10 lower in every domain, consistently, so
the margins are correspondingly ~0.1 higher. This probe scores a decoy only at
placements whose template centre falls inside the one chosen wrong cell; the
doc's one-off script appears to have allowed a looser wrong-cell window, which
raises the max. The offset is uniform across domains, so it shifts the margin
reference rather than distorting any comparison — and the gate is calibrated on
this tool's own real measurement, so neither delta weakens it.

Data locations
--------------
Real data defaults to the **main checkout**, never the worktree copy: this
worktree's ``network/datasets/north_star/v1`` is a stale pre-orientation-fix
version whose overviews differ from the main checkout's in every file, and
silently using it collapsed a classical baseline from 78% to 4%. The main
checkout is derived by stripping ``.claude/worktrees/<name>`` from this file's
path; ``--real-dataset-root`` / ``--real-cache-dir`` override it. Synthetic
piece roots prefer the local worktree copy and fall back to the main checkout;
the source puzzle JPEGs live only in the main checkout.

Usage (run from the ``network/`` directory)
-------------------------------------------
::

    # Documented baselines. Both are expected to FAIL.
    uv run python -m experiments.exp31_pixel_identity.ncc_probe --pipeline exp20
    uv run python -m experiments.exp31_pixel_identity.ncc_probe --pipeline exp30

    # Re-measure the real reference (this is what the gate is calibrated on).
    uv run python -m experiments.exp31_pixel_identity.ncc_probe \
        --pipeline real --sample 200 --json outputs/ncc_real.json

    # The gate itself, calibrated on that measurement, for CI / pre-retraining.
    uv run python -m experiments.exp31_pixel_identity.ncc_probe \
        --pipeline exp31 --reference-json outputs/ncc_real.json \
        --json outputs/ncc_exp31.json

Exit code is 0 only when every gated condition holds.

exp31 interface expected from the sibling module
------------------------------------------------
``--pipeline exp31`` looks, in ``experiments.exp31_pixel_identity``, for a
module named one of ``augmentation``, ``augment``, ``aug``, ``degradation``,
``asymmetric`` exposing

* a zero-argument-callable config class *or factory* named one of
  ``AugmentConfig``, ``Exp31AugmentConfig``, ``Exp31Config``,
  ``PixelIdentityConfig``, ``DegradationConfig``, ``CaptureConfig`` (optional;
  ``None`` is passed when absent),
* a piece function named one of ``augment_piece_rgba``, ``degrade_piece_rgba``
  (preferred: returns RGBA, so the exact mask survives) or ``augment_piece``,
  ``degrade_piece`` (returns RGB; the mask then falls back to
  ``--content-threshold`` on a black composite, which is less exact),
* an overview function named one of ``augment_overview``, ``degrade_overview``,
  ``augment_puzzle``,

each called as ``fn(image, config)`` and falling back to ``fn(image)`` when the
signature takes one argument, plus an optional module-level
``DEFAULT_DATASET_ROOT``. Anything else raises a RuntimeError naming every
module and symbol that was searched.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
from PIL import Image

from ..exp20_realistic_pieces.dataset import GRID_SIZE, get_cell_index, parse_piece_filename
from ..exp30_generator_fixes.framing import frame_rgba, rotate_lossless
from ..exp30_generator_fixes.probes import format_table

NETWORK_ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_PIPELINES = ("exp20", "exp26", "exp30", "exp31")
PIPELINES = SYNTHETIC_PIPELINES + ("real",)

# Piece roots per synthetic pipeline, relative to network/datasets.
PIECE_ROOTS = {
    "exp20": "realistic_4x4_rgba",
    "exp26": "realistic_4x4_rgba",
    "exp30": "realistic_4x4_rgba_v2",
    "exp31": "realistic_4x4_rgba_v2",
}

# Defaults, all justified in the module docstring.
DEFAULT_SAMPLE = 200
DEFAULT_SCALES = 9
DEFAULT_ROTATIONS = 7
DEFAULT_SCALE_SPAN = 1.25
DEFAULT_ROTATION_SPAN = 9.0
DEFAULT_ERODE = 2
DEFAULT_ALPHA_THRESHOLD = 254
DEFAULT_CONTENT_THRESHOLD = 8.0
DEFAULT_HIGH_NCC = 0.8
DEFAULT_REAL_CELL_PX = 64

GT_MEDIAN_TOLERANCE = 0.10
HIGH_FRACTION_TOLERANCE = 0.12
MARGIN_TOLERANCE_LOW = 0.10
MARGIN_TOLERANCE_HIGH = 0.15

# Winning-candidate fraction on a grid end above which the grid is called out.
EDGE_SATURATION_WARN = 0.25

# Gate references. "measured" comes from --pipeline real --sample 200 --seed 0
# at the default flags; "doc" is the published section 4.3 table.
REFERENCES: dict[str, dict[str, float]] = {
    "measured": {"gt_median": 0.679, "high_fraction": 0.340, "decoy_median": 0.303, "margin": 0.377},
    "doc": {"gt_median": 0.730, "high_fraction": 0.410, "decoy_median": 0.407, "margin": 0.323},
}

# Baselines measured with this tool (--sample 200 --seed 0, default flags), for
# the comparison table. The doc's section 4.3 values are in REFERENCES["doc"].
BASELINES: dict[str, dict[str, float]] = {
    "exp20 (raw)": {"gt_median": 1.000, "high_fraction": 0.990, "decoy_median": 0.435, "margin": 0.565},
    "exp26 (aug)": {"gt_median": 0.933, "high_fraction": 0.830, "decoy_median": 0.432, "margin": 0.501},
    "exp30 (aug+fixes)": {"gt_median": 0.937, "high_fraction": 0.840, "decoy_median": 0.453, "margin": 0.484},
    "real (north_star)": {"gt_median": 0.679, "high_fraction": 0.340, "decoy_median": 0.303, "margin": 0.377},
}

# Correlation guards. A response position is scored only where the masked
# per-channel image variance exceeds MIN_VARIANCE_PER_PIXEL intensity levels
# squared per channel; flat overview patches carry no correlation signal and
# would otherwise divide by ~0 and report NCC > 1.
MIN_VARIANCE_PER_PIXEL = 0.75
MIN_TEMPLATE_NORM = 1e-3
MIN_MASK_PIXELS = 16


@dataclass
class SearchGrid:
    """The scale x rotation candidate grid shared by every domain.

    Attributes:
        scales: Linear scale factors applied to the reference area's square root.
        angles: Continuous rotation offsets in degrees.
    """

    scales: list[float]
    angles: list[float]

    @classmethod
    def build(cls, n_scales: int, scale_span: float, n_rotations: int, rotation_span: float) -> "SearchGrid":
        """Build a symmetric grid that always contains the identity candidate.

        Args:
            n_scales: Number of scale steps (odd values keep 1.0 on the grid).
            scale_span: Half-width of the scale range as a multiplicative factor.
            n_rotations: Number of rotation steps (odd values keep 0.0 on the grid).
            rotation_span: Half-width of the rotation range in degrees.

        Returns:
            The candidate grid.
        """
        scales = np.geomspace(1.0 / scale_span, scale_span, n_scales) if n_scales > 1 else np.array([1.0])
        angles = np.linspace(-rotation_span, rotation_span, n_rotations) if n_rotations > 1 else np.array([0.0])
        return cls(scales=[float(s) for s in scales], angles=[float(a) for a in angles])


@dataclass
class ProbeConfig:
    """Everything a worker needs to reproduce one domain's view and search.

    Attributes:
        pipeline: Domain name from PIPELINES.
        piece_root: Root of stored piece PNGs (synthetic pipelines).
        puzzle_root: Root of source puzzle JPEGs (synthetic pipelines).
        real_dataset_root: north_star v1 root (real pipeline).
        real_cache_dir: north_star prepared-crop cache (real pipeline).
        grid: Candidate search grid.
        erode: Mask erosion passes before correlating.
        alpha_threshold: Alpha strictly above this counts as piece content.
        content_threshold: Max-channel value above this counts as content.
        real_cell_px: Overview pixels per cell for the real domain.
        seed: Base seed for augmentation draws and decoy choice.
    """

    pipeline: str
    piece_root: Path
    puzzle_root: Path
    real_dataset_root: Path
    real_cache_dir: Path
    grid: SearchGrid
    erode: int
    alpha_threshold: int
    content_threshold: float
    real_cell_px: int
    seed: int


@dataclass
class Verdict:
    """Outcome of the gate.

    Attributes:
        passed: Whether every gated condition held.
        reasons: One line per condition, prefixed PASS/FAIL/INFO.
    """

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


# --------------------------------------------------------------------------
# Path resolution
# --------------------------------------------------------------------------


def main_checkout_network_root() -> Path:
    """Return the main checkout's ``network/`` directory.

    A git worktree of this repo lives at ``<main>/.claude/worktrees/<name>``,
    so stripping everything from ``.claude`` onwards yields the main checkout.
    Outside a worktree this returns the current network root.

    Returns:
        Path to the main checkout's network directory.
    """
    parts = NETWORK_ROOT.parts
    if ".claude" in parts:
        return Path(*parts[: parts.index(".claude")]) / "network"
    return NETWORK_ROOT


def resolve_dataset(relative: str) -> Path:
    """Resolve a ``datasets/`` subpath, preferring the local worktree copy.

    Args:
        relative: Path relative to ``network/datasets``.

    Returns:
        The local path when it exists, else the main checkout's path.
    """
    local = NETWORK_ROOT / "datasets" / relative
    if local.exists():
        return local
    return main_checkout_network_root() / "datasets" / relative


def real_dataset_defaults() -> tuple[Path, Path]:
    """Return the north_star dataset root and eval cache, always from main.

    The worktree's own ``north_star/v1`` is a stale pre-orientation-fix copy;
    using it silently produces garbage, so the main checkout wins even when a
    local copy exists.

    Returns:
        Tuple of (dataset root, eval cache directory).
    """
    root = main_checkout_network_root() / "datasets" / "north_star"
    return root / "v1", root / "v1_eval_cache"


# --------------------------------------------------------------------------
# Masked zero-mean NCC
# --------------------------------------------------------------------------


def masked_zncc_response(overview: np.ndarray, template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked zero-mean normalized cross-correlation over every placement.

    Both sides are zero-meaned per channel **inside the mask only** and
    normalized by their masked L2 norm, so the score is invariant to the
    near-affine intensity map exp26 calls photometric jitter. Implemented with
    four ``cv2.matchTemplate`` TM_CCORR passes rather than
    ``TM_CCOEFF_NORMED(mask=...)`` because OpenCV's masked CCOEFF does not
    subtract the mask-weighted image mean exactly, and this probe has to report
    an exact 1.000 on a byte-identical pair.

    Args:
        overview: HxWx3 float32 overview in [0, 255].
        template: hxwx3 float32 template in [0, 255].
        mask: hxw float32 mask with values in {0, 1}.

    Returns:
        A (H-h+1)x(W-w+1) float32 response in [-1, 1]; positions whose masked
        image variance is negligible score 0.
    """
    count = float(mask.sum())
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    template_mean = (template * mask3).sum(axis=(0, 1)) / count
    centered = (template - template_mean) * mask3
    template_norm = float(np.sqrt((centered * centered).sum()))
    if template_norm < MIN_TEMPLATE_NORM:
        height = overview.shape[0] - template.shape[0] + 1
        width = overview.shape[1] - template.shape[1] + 1
        return np.zeros((height, width), dtype=np.float32)

    numerator = cv2.matchTemplate(overview, centered, cv2.TM_CCORR)
    sum_squares = cv2.matchTemplate(overview * overview, mask3, cv2.TM_CCORR)
    sums = np.stack([cv2.matchTemplate(overview[:, :, c], mask, cv2.TM_CCORR) for c in range(3)], axis=0)
    variance = sum_squares - (sums * sums).sum(axis=0) / count

    floor = MIN_VARIANCE_PER_PIXEL * count
    denominator = np.sqrt(np.maximum(variance, 0.0)) * template_norm
    response = np.where(variance > floor, numerator / np.maximum(denominator, 1e-8), 0.0)
    return np.clip(response, -1.0, 1.0).astype(np.float32)


def cell_rect(cell: int, rows: int, cols: int, width: int, height: int) -> tuple[float, float, float, float]:
    """Return the (x0, y0, x1, y1) pixel rect of a row-major cell index.

    Args:
        cell: Row-major cell index.
        rows: Grid rows.
        cols: Grid columns.
        width: Overview width in pixels.
        height: Overview height in pixels.

    Returns:
        The cell's rectangle in overview pixel coordinates.
    """
    row, col = divmod(cell, cols)
    return (
        col * width / cols,
        row * height / rows,
        (col + 1) * width / cols,
        (row + 1) * height / rows,
    )


def best_in_cell(response: np.ndarray, template_shape: tuple[int, int], rect: tuple[float, ...]) -> float | None:
    """Best response over placements whose template centre falls inside a cell.

    Centre-based binning is how every classical baseline here assigns a match
    to a cell (exp23/exp25 ``get_cell_index`` / ``cell_index``), so the probe
    inherits the same notion of "at this cell".

    Args:
        response: Response map indexed by template top-left position.
        template_shape: (height, width) of the template.
        rect: Target cell rect (x0, y0, x1, y1) in overview pixels.

    Returns:
        The best score, or None when no placement centres inside the cell.
    """
    tpl_h, tpl_w = template_shape
    x0 = max(0, int(np.ceil(rect[0] - tpl_w / 2)))
    x1 = min(response.shape[1] - 1, int(np.floor(rect[2] - tpl_w / 2)))
    y0 = max(0, int(np.ceil(rect[1] - tpl_h / 2)))
    y1 = min(response.shape[0] - 1, int(np.floor(rect[3] - tpl_h / 2)))
    if x1 < x0 or y1 < y0:
        return None
    return float(response[y0 : y1 + 1, x0 : x1 + 1].max())


def bbox_crop(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop an image and its mask to the mask's bounding box.

    Args:
        rgb: HxWx3 float32 image.
        mask: HxW float32 mask.

    Returns:
        The cropped (image, mask), or None when the mask is empty.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return np.ascontiguousarray(rgb[y0:y1, x0:x1]), np.ascontiguousarray(mask[y0:y1, x0:x1])


def _rotate_view(rgb: np.ndarray, mask: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an image and mask about their centre on a padded canvas.

    Args:
        rgb: HxWx3 float32 image.
        mask: HxW float32 mask.
        angle: Counter-clockwise rotation in degrees.

    Returns:
        The rotated (image, mask); the mask is re-binarised at 0.5.
    """
    height, width = mask.shape
    diagonal = int(np.ceil(np.hypot(height, width)))
    pad_y = (diagonal - height) // 2 + 2
    pad_x = (diagonal - width) // 2 + 2
    rgb = np.pad(rgb, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
    mask = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)))
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated, (rotated_mask > 0.5).astype(np.float32)


def prepare_template(
    rgb: np.ndarray, mask: np.ndarray, angle: float, target_area: float, erode: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build one search candidate: rotate, area-normalize, erode, crop.

    The resize factor is derived from the *current* mask area, so any
    resolution asymmetry the pipeline introduced between piece and overview is
    undone by the search rather than scored as a matcher failure. Erosion runs
    last so it never influences the area normalisation, and it only removes
    pixels, which keeps a byte-identical pair at exactly 1.000.

    Args:
        rgb: HxWx3 float32 piece image.
        mask: HxW float32 piece mask.
        angle: Rotation offset in degrees.
        target_area: Target mask area in overview pixels squared.
        erode: Erosion passes (structuring element ``2*erode+1`` square).

    Returns:
        The (template, mask) pair cropped to the mask bbox, or None when the
        candidate degenerates to fewer than MIN_MASK_PIXELS pixels.
    """
    if angle != 0.0:
        rgb, mask = _rotate_view(rgb, mask, angle)
    area = float(mask.sum())
    if area < MIN_MASK_PIXELS:
        return None
    factor = float(np.sqrt(target_area / area))
    if abs(factor - 1.0) > 1e-6:
        height, width = mask.shape
        size = (max(4, int(round(width * factor))), max(4, int(round(height * factor))))
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, size, interpolation=interpolation)
        mask = (cv2.resize(mask, size, interpolation=interpolation) > 0.5).astype(np.float32)
    if erode > 0:
        element = np.ones((2 * erode + 1, 2 * erode + 1), dtype=np.uint8)
        mask = cv2.erode(mask, element)
    if mask.sum() < MIN_MASK_PIXELS:
        return None
    return bbox_crop(rgb, mask)


def _iter_candidates(
    rgb: np.ndarray, mask: np.ndarray, reference_area: float, grid: SearchGrid, erode: int, limit: tuple[int, int]
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield every (scale index, angle index, template, mask) candidate.

    Args:
        rgb: HxWx3 float32 piece image.
        mask: HxW float32 piece mask.
        reference_area: Domain reference mask area in overview pixels squared.
        grid: Candidate grid.
        erode: Mask erosion passes.
        limit: (height, width) of the overview; larger templates are skipped.

    Yields:
        Candidate tuples in grid order.
    """
    for scale_idx, scale in enumerate(grid.scales):
        for angle_idx, angle in enumerate(grid.angles):
            candidate = prepare_template(rgb, mask, angle, reference_area * scale * scale, erode)
            if candidate is None:
                continue
            template, template_mask = candidate
            if template.shape[0] >= limit[0] or template.shape[1] >= limit[1]:
                continue
            yield scale_idx, angle_idx, template, template_mask


def search_piece(
    overview: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    reference_area: float,
    cells: dict[str, tuple[float, ...]],
    grid: SearchGrid,
    erode: int,
) -> dict[str, Any]:
    """Run the full scale x rotation search for one piece against named cells.

    Args:
        overview: HxWx3 float32 overview in [0, 255].
        rgb: HxWx3 float32 piece image.
        mask: HxW float32 piece mask.
        reference_area: Domain reference mask area in overview pixels squared.
        cells: Mapping of cell name ("gt", "decoy") to its pixel rect.
        grid: Candidate grid.
        erode: Mask erosion passes.

    Returns:
        Mapping with one score per cell name plus the winning grid indices and
        the number of evaluated candidates.
    """
    height, width = overview.shape[:2]
    best = dict.fromkeys(cells, -2.0)
    scale_idx = angle_idx = -1
    evaluated = 0
    for candidate_scale, candidate_angle, template, template_mask in _iter_candidates(
        rgb, mask, reference_area, grid, erode, (height, width)
    ):
        evaluated += 1
        response = masked_zncc_response(overview, template, template_mask)
        for name, rect in cells.items():
            value = best_in_cell(response, template.shape[:2], rect)
            if value is None or value <= best[name]:
                continue
            best[name] = value
            if name == "gt":
                scale_idx, angle_idx = candidate_scale, candidate_angle
    result: dict[str, Any] = {name: (score if score > -2.0 else None) for name, score in best.items()}
    result.update({"scale_idx": scale_idx, "angle_idx": angle_idx, "candidates": evaluated})
    return result


# --------------------------------------------------------------------------
# Synthetic domain views
# --------------------------------------------------------------------------


def stable_seed(seed: int, kind: str, key: str) -> int:
    """Derive a process-independent per-sample seed.

    ``hash()`` on a string is salted per interpreter, so using it here would
    make the augmentation draws — and therefore the measured numbers — differ
    between runs and between workers. CRC32 is stable everywhere.

    Args:
        seed: Base seed.
        kind: Namespace, so a piece and its overview never share a draw.
        key: Stable per-sample key.

    Returns:
        A seed in [0, 2**31).
    """
    return zlib.crc32(f"{seed}:{kind}:{key}".encode()) % (2**31)


def _rgba_to_arrays(rgba: Image.Image, alpha_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """Split an RGBA piece into a float image and a thresholded mask.

    Args:
        rgba: RGBA piece image.
        alpha_threshold: Alpha strictly above this counts as content.

    Returns:
        Tuple of (HxWx3 float32 image, HxW float32 mask).
    """
    array = np.asarray(rgba.convert("RGBA"), dtype=np.float32)
    return np.ascontiguousarray(array[..., :3]), (array[..., 3] > alpha_threshold).astype(np.float32)


def _luminance_mask(rgb: np.ndarray, content_threshold: float) -> np.ndarray:
    """Build a content mask for a black-composited RGB image.

    Args:
        rgb: HxWx3 float32 image in [0, 255].
        content_threshold: Max-channel value above this counts as content.

    Returns:
        HxW float32 mask.
    """
    return (rgb.max(axis=2) > content_threshold).astype(np.float32)


def _aligned_piece(piece_path: Path, base_rotation: int) -> Image.Image:
    """Load a stored piece and undo its baked rotation losslessly.

    The stored rotation is clockwise, so the inverse is a clockwise rotation by
    ``360 - base_rotation``; ``rotate_lossless`` is an exact pixel permutation,
    so the alignment step introduces no blur of its own.

    Args:
        piece_path: Path to the stored RGBA piece PNG.
        base_rotation: Clockwise rotation baked in at generation time.

    Returns:
        The RGBA piece aligned with the source overview.
    """
    with Image.open(piece_path) as raw:
        rgba = raw.convert("RGBA")
    return rotate_lossless(rgba, (360 - base_rotation) % 360)


def _exp26_augmented_view(rgba: Image.Image, config: Any, probe: "ProbeConfig") -> tuple[np.ndarray, np.ndarray]:
    """Apply exp26's augmentation while keeping the exact piece mask.

    ``augment_piece`` returns RGB, which would force a luminance mask and
    discard dark piece pixels, so its two halves are called directly:
    ``_augment_geometry`` (RGBA in, RGBA out, alpha intact) then a black
    composite and ``_augment_appearance``. The augmentation itself is exp26's
    unmodified code. Compositing on black rather than sampling exp26's
    background modes is deliberate: black is the deployed rembg appearance and
    exp26's own majority mode, and it keeps the mask exact.

    Args:
        rgba: RGBA piece aligned with the overview.
        config: exp26 ``AugmentConfig``.
        probe: Probe configuration (mask thresholds).

    Returns:
        Tuple of (HxWx3 float32 augmented image, HxW float32 mask).
    """
    from ..exp26_domain_randomization.augment import _augment_appearance, _augment_geometry, _composite

    geometry = _augment_geometry(rgba, config)
    appearance = _augment_appearance(_composite(geometry, None), config)
    rgb = np.asarray(appearance.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(geometry.getchannel("A"), dtype=np.float32)
    if alpha.shape != rgb.shape[:2]:
        return rgb, _luminance_mask(rgb, probe.content_threshold)
    return rgb, (alpha > probe.alpha_threshold).astype(np.float32)


def _exp31_module() -> Any:
    """Import the sibling's exp31 augmentation module.

    Returns:
        The first importable candidate module.

    Raises:
        RuntimeError: When no candidate module can be imported, naming each.
    """
    searched: list[str] = []
    for name in ("augmentation", "augment", "aug", "degradation", "asymmetric"):
        # RunPod flattens the package into one directory and runs modules as
        # scripts, so __package__ is empty there; fall back to the bare name
        # rather than trying to import ".augmentation".
        candidates = [f"{__package__}.{name}", name] if __package__ else [name]
        for qualified in candidates:
            try:
                return importlib.import_module(qualified)
            except ImportError:
                searched.append(qualified)
    raise RuntimeError(
        "exp31 NCC probe: no augmentation module found. Searched: "
        + ", ".join(searched)
        + ". Expose one of those modules with a piece function (augment_piece_rgba / degrade_piece_rgba / "
        "augment_piece / degrade_piece) and an overview function (augment_overview / degrade_overview / "
        "augment_puzzle), then re-run this probe."
    )


def _first_attr(module: Any, names: tuple[str, ...]) -> tuple[str, Any] | None:
    """Return the first present attribute of ``module`` from ``names``.

    Args:
        module: Module to inspect.
        names: Candidate attribute names, in priority order.

    Returns:
        Tuple of (name, value), or None when none is present.
    """
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return name, value
    return None


def _call_with_config(function: Any, image: Any, config: Any) -> Any:
    """Call ``function(image, config)``, degrading to ``function(image)``.

    Args:
        function: The sibling's augmentation callable.
        image: Image argument.
        config: Config argument, possibly None.

    Returns:
        Whatever the callable returns.
    """
    try:
        takes_config = len(inspect.signature(function).parameters) >= 2
    except (TypeError, ValueError):
        takes_config = True
    if takes_config and config is not None:
        return function(image, config)
    return function(image)


@dataclass
class Exp31Hooks:
    """Resolved entry points into the sibling's exp31 augmentation module.

    Attributes:
        module_name: Fully qualified module that supplied the hooks.
        piece: Callable turning an RGBA piece into an augmented view.
        piece_name: Attribute name of ``piece``.
        piece_returns_rgba: Whether ``piece`` preserves the alpha channel.
        overview: Callable turning an RGB overview into an augmented view.
        overview_name: Attribute name of ``overview``.
        config: Config instance passed to both callables, or None.
    """

    module_name: str
    piece: Callable[..., Any]
    piece_name: str
    piece_returns_rgba: bool
    overview: Callable[..., Any]
    overview_name: str
    config: Any


PIECE_RGBA_NAMES = ("augment_piece_rgba", "degrade_piece_rgba")
PIECE_RGB_NAMES = ("augment_piece", "degrade_piece")
OVERVIEW_NAMES = ("augment_overview", "degrade_overview", "augment_puzzle")
CONFIG_NAMES = (
    "AugmentConfig",
    "Exp31AugmentConfig",
    "Exp31Config",
    "PixelIdentityConfig",
    "DegradationConfig",
    "CaptureConfig",
)


def resolve_exp31_hooks() -> Exp31Hooks:
    """Locate the sibling's exp31 augmentation entry points, defensively.

    Returns:
        The resolved hooks.

    Raises:
        RuntimeError: When a required entry point is missing, naming exactly
            which module was inspected and which symbols were searched for.
    """
    module = _exp31_module()
    piece = _first_attr(module, PIECE_RGBA_NAMES)
    returns_rgba = piece is not None
    if piece is None:
        piece = _first_attr(module, PIECE_RGB_NAMES)
    overview = _first_attr(module, OVERVIEW_NAMES)
    if piece is None or overview is None:
        raise RuntimeError(
            f"exp31 NCC probe: {module.__name__} is missing an entry point. Searched for a piece function in "
            f"{PIECE_RGBA_NAMES + PIECE_RGB_NAMES} (found: {piece[0] if piece else None}) and an overview "
            f"function in {OVERVIEW_NAMES} (found: {overview[0] if overview else None}). Both are required, "
            "each callable as fn(image, config) or fn(image)."
        )
    config_entry = _first_attr(module, CONFIG_NAMES)
    config = None
    if config_entry is not None:
        try:
            config = config_entry[1]()
        except Exception as exc:  # noqa: B902 - report, do not guess a signature
            raise RuntimeError(
                f"exp31 NCC probe: {module.__name__}.{config_entry[0]}() is not zero-argument constructible "
                f"({exc!r}). Give it defaults, or drop it and accept fn(image)."
            ) from exc
    if not returns_rgba:
        print(f"WARNING: {module.__name__}.{piece[0]} returns RGB; the piece mask falls back to a luminance threshold")
    return Exp31Hooks(
        module_name=module.__name__,
        piece=piece[1],
        piece_name=piece[0],
        piece_returns_rgba=returns_rgba,
        overview=overview[1],
        overview_name=overview[0],
        config=config,
    )


def _exp31_piece_view(rgba: Image.Image, hooks: Exp31Hooks, content_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the exp31 piece view through the sibling's augmentation.

    Args:
        rgba: RGBA piece aligned with the overview.
        hooks: Resolved exp31 entry points.
        content_threshold: Mask threshold used when alpha is not preserved.

    Returns:
        Tuple of (HxWx3 float32 image, HxW float32 mask).

    Raises:
        RuntimeError: When the hook does not return a PIL image.
    """
    result = _call_with_config(hooks.piece, rgba, hooks.config)
    if not isinstance(result, Image.Image):
        raise RuntimeError(
            f"exp31 NCC probe: {hooks.module_name}.{hooks.piece_name} returned {type(result)!r}; "
            "a PIL RGBA (preferred) or RGB image is required."
        )
    if hooks.piece_returns_rgba and result.mode == "RGBA":
        return _rgba_to_arrays(result, DEFAULT_ALPHA_THRESHOLD)
    rgb = np.asarray(result.convert("RGB"), dtype=np.float32)
    return rgb, _luminance_mask(rgb, content_threshold)


def synthetic_piece_view(config: ProbeConfig, piece: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    """Build one synthetic domain's piece view and its reference area.

    Args:
        config: Probe configuration.
        piece: Piece descriptor from ``plan_synthetic_jobs``.

    Returns:
        Tuple of (HxWx3 float32 image, HxW float32 mask, reference mask area).
    """
    from ..exp26_domain_randomization.augment import AugmentConfig, seed_everything

    path = Path(piece["path"])
    rgba = _aligned_piece(path, int(piece["base_rotation"]))
    reference_rgb, reference_mask = _rgba_to_arrays(rgba, config.alpha_threshold)
    reference_area = float(reference_mask.sum())
    if config.pipeline == "exp20":
        return reference_rgb, reference_mask, reference_area

    seed_everything(stable_seed(config.seed, "piece", str(piece["path"])))
    if config.pipeline == "exp31":
        rgb, mask = _exp31_piece_view(rgba, resolve_exp31_hooks(), config.content_threshold)
    else:
        framed = frame_rgba(rgba) if config.pipeline == "exp30" else rgba
        rgb, mask = _exp26_augmented_view(framed, AugmentConfig(), config)
    return rgb, mask, reference_area


def synthetic_overview(config: ProbeConfig, puzzle_id: str) -> np.ndarray:
    """Load one synthetic overview through the pipeline's own puzzle path.

    The overview is used at whatever resolution the pipeline produces (256x256
    for exp20/exp26/exp30), so a pipeline that deliberately renders the
    overview at a different pixel-per-cell budget is measured as it is.

    Args:
        config: Probe configuration.
        puzzle_id: Source puzzle identifier.

    Returns:
        HxWx3 float32 overview in [0, 255].
    """
    from ..exp26_domain_randomization.augment import AugmentConfig, augment_puzzle, seed_everything

    with Image.open(config.puzzle_root / f"{puzzle_id}.jpg") as raw:
        overview = raw.convert("RGB")
    if config.pipeline == "exp20":
        return np.asarray(overview, dtype=np.float32)
    seed_everything(stable_seed(config.seed, "overview", puzzle_id))
    if config.pipeline == "exp31":
        hooks = resolve_exp31_hooks()
        result = _call_with_config(hooks.overview, overview, hooks.config)
        if not isinstance(result, Image.Image):
            raise RuntimeError(
                f"exp31 NCC probe: {hooks.module_name}.{hooks.overview_name} returned {type(result)!r}; "
                "a PIL RGB image is required."
            )
        return np.asarray(result.convert("RGB"), dtype=np.float32)
    return np.asarray(augment_puzzle(overview, AugmentConfig()), dtype=np.float32)


# --------------------------------------------------------------------------
# Real domain view
# --------------------------------------------------------------------------


def real_overview(config: ProbeConfig, puzzle_id: str, rows: int, cols: int) -> np.ndarray:
    """Load a north_star overview, crop it to the puzzle and set its cell budget.

    Reuses exp25's ``crop_overview`` so the puzzle/poster region is found
    exactly as the real-photo benchmark finds it, then resizes to
    ``--real-cell-px`` pixels per cell. The default of 64 matches the synthetic
    domains' 256px / 4 cells, which is what makes the NCC values comparable.

    Args:
        config: Probe configuration.
        puzzle_id: north_star puzzle identifier.
        rows: Grid rows.
        cols: Grid columns.

    Returns:
        HxWx3 float32 overview in [0, 255].
    """
    from ..exp25_north_star_eval.evaluate import crop_overview

    with Image.open(config.real_dataset_root / puzzle_id / "overview.jpg") as raw:
        rgb = np.array(raw.convert("RGB"))
    x0, y0, x1, y1 = crop_overview(rgb)
    size = (cols * config.real_cell_px, rows * config.real_cell_px)
    resized = cv2.resize(rgb[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def real_piece_view(config: ProbeConfig, piece: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load one prepared north_star crop and align it with the overview.

    The crop is the exact eval input (bbox -> rembg -> largest component on
    black -> pad square) cached by exp25, so the probe measures the deployment
    domain including its segmentation artefacts.

    Args:
        config: Probe configuration.
        piece: Piece descriptor from ``plan_real_jobs``.

    Returns:
        Tuple of (HxWx3 float32 image, HxW float32 mask).
    """
    with Image.open(config.real_cache_dir / piece["cache_name"]) as raw:
        rgb = np.asarray(raw.convert("RGB"), dtype=np.float32)
    mask = _luminance_mask(rgb, config.content_threshold)
    steps = int(piece["base_rotation"]) // 90
    if steps:
        rgb = np.ascontiguousarray(np.rot90(rgb, k=steps))
        mask = np.ascontiguousarray(np.rot90(mask, k=steps))
    return rgb, mask


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def round_robin(groups: list[list[Any]], sample: int, rng: random.Random) -> list[Any]:
    """Draw up to ``sample`` items round-robin across shuffled groups.

    Spreading over as many puzzles as possible is what makes the median a
    domain statistic rather than a statement about a handful of motifs.

    Args:
        groups: One list of items per puzzle.
        sample: Number of items to draw.
        rng: Seeded RNG.

    Returns:
        The drawn items.
    """
    shuffled = [list(group) for group in groups if group]
    rng.shuffle(shuffled)
    for group in shuffled:
        rng.shuffle(group)
    picked: list[Any] = []
    depth = 0
    while len(picked) < sample:
        available = [group for group in shuffled if depth < len(group)]
        if not available:
            break
        for group in available:
            picked.append(group[depth])
            if len(picked) >= sample:
                break
        depth += 1
    return picked


def plan_synthetic_jobs(config: ProbeConfig, sample: int, seed: int) -> list[dict[str, Any]]:
    """Plan one job per sampled synthetic puzzle.

    Args:
        config: Probe configuration.
        sample: Number of pieces to draw.
        seed: Sampling seed.

    Returns:
        One job dict per puzzle, each holding its sampled piece descriptors.

    Raises:
        FileNotFoundError: When the piece root holds no parseable piece PNGs.
    """
    if not config.piece_root.is_dir():
        raise FileNotFoundError(f"Piece root does not exist: {config.piece_root}")
    groups: list[list[dict[str, Any]]] = []
    for puzzle_dir in sorted(p for p in config.piece_root.iterdir() if p.is_dir()):
        entries: list[dict[str, Any]] = []
        for path in sorted(puzzle_dir.glob(f"{puzzle_dir.name}_x*_y*_rot*.png")):
            parsed = parse_piece_filename(path.name)
            if parsed is None:
                continue
            _, cx, cy, base_rotation = parsed
            entries.append(
                {
                    "puzzle_id": puzzle_dir.name,
                    "key": path.name,
                    "path": str(path),
                    "base_rotation": base_rotation,
                    "gt_cell": get_cell_index(cx, cy),
                }
            )
        groups.append(entries)
    picked = round_robin(groups, sample, random.Random(seed))
    if not picked:
        raise FileNotFoundError(f"No parseable piece PNGs ('<puzzle>_x*_y*_rot*.png') under {config.piece_root}")
    return _group_jobs(picked, GRID_SIZE, GRID_SIZE)


def plan_real_jobs(config: ProbeConfig, sample: int, seed: int) -> list[dict[str, Any]]:
    """Plan one job per sampled north_star puzzle.

    Args:
        config: Probe configuration.
        sample: Number of piece photos to draw.
        seed: Sampling seed.

    Returns:
        One job dict per puzzle.

    Raises:
        FileNotFoundError: When the metadata or the prepared-crop cache is absent.
    """
    from ..exp25_north_star_eval.evaluate import cache_name

    metadata = config.real_dataset_root / "metadata.csv"
    if not metadata.exists():
        raise FileNotFoundError(f"north_star metadata not found: {metadata}")
    if not config.real_cache_dir.is_dir():
        raise FileNotFoundError(
            f"north_star eval cache not found: {config.real_cache_dir}. Run "
            "experiments/exp25_north_star_eval/evaluate.py once to build it (it needs rembg)."
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    with open(metadata) as handle:
        for row in csv.DictReader(handle):
            rows, cols = int(row["rows"]), int(row["cols"])
            grouped.setdefault(row["puzzle_id"], []).append(
                {
                    "puzzle_id": row["puzzle_id"],
                    "key": row["piece_file"],
                    "cache_name": cache_name({"piece_file": row["piece_file"]}),
                    "base_rotation": int(row["rotation"]),
                    "gt_cell": int(row["row"]) * cols + int(row["col"]),
                    "rows": rows,
                    "cols": cols,
                }
            )
    picked = round_robin([grouped[key] for key in sorted(grouped)], sample, random.Random(seed))
    return _group_jobs(picked, 0, 0)


def _group_jobs(picked: list[dict[str, Any]], rows: int, cols: int) -> list[dict[str, Any]]:
    """Group sampled piece descriptors into one job per puzzle.

    Args:
        picked: Sampled piece descriptors.
        rows: Grid rows, or 0 to read them from each descriptor.
        cols: Grid columns, or 0 to read them from each descriptor.

    Returns:
        One job dict per puzzle, in sorted puzzle order.
    """
    jobs: dict[str, dict[str, Any]] = {}
    for piece in picked:
        job = jobs.setdefault(
            piece["puzzle_id"],
            {
                "puzzle_id": piece["puzzle_id"],
                "rows": rows or int(piece["rows"]),
                "cols": cols or int(piece["cols"]),
                "pieces": [],
            },
        )
        job["pieces"].append(piece)
    return [jobs[key] for key in sorted(jobs)]


# --------------------------------------------------------------------------
# Job execution
# --------------------------------------------------------------------------


def run_job(payload: tuple[ProbeConfig, dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure every sampled piece of one puzzle (worker entry point).

    Args:
        payload: Tuple of (probe config, job dict).

    Returns:
        One record per piece.
    """
    config, job = payload
    rows, cols = int(job["rows"]), int(job["cols"])
    if config.pipeline == "real":
        overview = real_overview(config, job["puzzle_id"], rows, cols)
    else:
        overview = synthetic_overview(config, job["puzzle_id"])
    height, width = overview.shape[:2]
    cell_area = (width / cols) * (height / rows)

    records: list[dict[str, Any]] = []
    for piece in job["pieces"]:
        if config.pipeline == "real":
            rgb, mask = real_piece_view(config, piece)
            reference_area = cell_area
        else:
            rgb, mask, reference_area = synthetic_piece_view(config, piece)
        cropped = bbox_crop(rgb, mask)
        if cropped is None:
            continue
        gt_cell = int(piece["gt_cell"])
        decoy_cell = _decoy_cell(gt_cell, rows * cols, config.seed, piece["key"])
        cells = {
            "gt": cell_rect(gt_cell, rows, cols, width, height),
            "decoy": cell_rect(decoy_cell, rows, cols, width, height),
        }
        result = search_piece(overview, cropped[0], cropped[1], reference_area, cells, config.grid, config.erode)
        result.update(
            {
                "puzzle_id": job["puzzle_id"],
                "key": piece["key"],
                "gt_cell": gt_cell,
                "decoy_cell": decoy_cell,
                "px_per_cell": float(np.sqrt(cell_area)),
            }
        )
        records.append(result)
    return records


def _decoy_cell(gt_cell: int, n_cells: int, seed: int, key: str) -> int:
    """Pick a reproducible wrong cell for one piece.

    Args:
        gt_cell: Ground-truth cell index.
        n_cells: Number of cells in the puzzle.
        seed: Base seed.
        key: Stable per-piece key.

    Returns:
        A cell index different from ``gt_cell`` (or ``gt_cell`` for a 1-cell grid).
    """
    others = [cell for cell in range(n_cells) if cell != gt_cell]
    if not others:
        return gt_cell
    return random.Random(f"{seed}:{key}").choice(others)


def run_all(config: ProbeConfig, jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Execute every job, in parallel when ``workers > 1``.

    Args:
        config: Probe configuration.
        jobs: Job dicts from the planners.
        workers: Worker process count (1 runs in-process).

    Returns:
        All per-piece records.
    """
    payloads = [(config, job) for job in jobs]
    records: list[dict[str, Any]] = []
    if workers <= 1:
        for payload in payloads:
            records.extend(run_job(payload))
        return records
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for chunk in pool.map(run_job, payloads):
            records.extend(chunk)
    return records


# --------------------------------------------------------------------------
# Statistics and gate
# --------------------------------------------------------------------------


def summarize(records: list[dict[str, Any]], grid: SearchGrid, high_ncc: float) -> dict[str, Any]:
    """Aggregate per-piece records into the domain statistics.

    Args:
        records: Per-piece records.
        grid: Candidate grid (for the edge-saturation diagnostics).
        high_ncc: Threshold for the "fraction above" tail metric.

    Returns:
        JSON-serializable statistics dict.
    """
    gt = np.array([r["gt"] for r in records if r["gt"] is not None], dtype=np.float64)
    decoy = np.array([r["decoy"] for r in records if r["decoy"] is not None], dtype=np.float64)
    scale_idx = np.array([r["scale_idx"] for r in records if r["scale_idx"] >= 0])
    angle_idx = np.array([r["angle_idx"] for r in records if r["angle_idx"] >= 0])
    gt_median = float(np.median(gt)) if gt.size else float("nan")
    decoy_median = float(np.median(decoy)) if decoy.size else float("nan")
    return {
        "n_pieces": len(records),
        "n_gt": int(gt.size),
        "n_decoy": int(decoy.size),
        "gt_median": gt_median,
        "high_fraction": float((gt > high_ncc).mean()) if gt.size else float("nan"),
        "decoy_median": decoy_median,
        "margin": gt_median - decoy_median,
        "gt_p25": float(np.percentile(gt, 25)) if gt.size else float("nan"),
        "gt_p75": float(np.percentile(gt, 75)) if gt.size else float("nan"),
        "scale_edge_low": _edge_fraction(scale_idx, len(grid.scales), "low"),
        "scale_edge_high": _edge_fraction(scale_idx, len(grid.scales), "high"),
        "rotation_edge_low": _edge_fraction(angle_idx, len(grid.angles), "low"),
        "rotation_edge_high": _edge_fraction(angle_idx, len(grid.angles), "high"),
        "median_px_per_cell": float(np.median([r["px_per_cell"] for r in records])) if records else float("nan"),
        "candidates_per_piece": float(np.mean([r["candidates"] for r in records])) if records else float("nan"),
    }


def _edge_fraction(indices: np.ndarray, size: int, side: str) -> float:
    """Fraction of winning indices sitting on one end of the grid.

    Args:
        indices: Winning grid indices.
        size: Grid length.
        side: ``"low"`` for the first step, ``"high"`` for the last.

    Returns:
        The saturation fraction, or NaN when there is nothing to measure.
    """
    if indices.size == 0 or size < 2:
        return float("nan")
    target = 0 if side == "low" else size - 1
    return float((indices == target).mean())


def gate(stats: dict[str, Any], reference: dict[str, float], high_ncc: float) -> Verdict:
    """Apply the acceptance gate to one domain's statistics.

    Args:
        stats: Statistics from ``summarize``.
        reference: Real-domain reference values.
        high_ncc: Threshold used for the tail metric (for the message text).

    Returns:
        The verdict.
    """
    verdict = Verdict()
    if not stats["n_gt"]:
        verdict.check(False, "no ground-truth scores - zero evidence cannot pass the gate")
        return verdict

    _gate_band(verdict, "GT median NCC", stats["gt_median"], reference["gt_median"], GT_MEDIAN_TOLERANCE)
    _gate_band(
        verdict,
        f"fraction NCC > {high_ncc}",
        stats["high_fraction"],
        reference["high_fraction"],
        HIGH_FRACTION_TOLERANCE,
    )
    _gate_band(
        verdict,
        "GT-decoy margin",
        stats["margin"],
        reference["margin"],
        MARGIN_TOLERANCE_LOW,
        MARGIN_TOLERANCE_HIGH,
    )
    verdict.info(f"decoy median NCC = {stats['decoy_median']:.3f} (real reference {reference['decoy_median']:.3f})")
    for label in ("scale", "rotation"):
        low, high = stats[f"{label}_edge_low"], stats[f"{label}_edge_high"]
        note = "grid covers the pipeline's own jitter"
        if high >= EDGE_SATURATION_WARN:
            note = "grid TOO NARROW on the high side: widen the span, the headroom is understated"
        elif low >= EDGE_SATURATION_WARN:
            note = "grid saturates low (the search prefers a smaller template than the reference area)"
        verdict.info(f"{label} grid edge-saturation low={low:.3f} high={high:.3f} ({note})")
    return verdict


def _gate_band(
    verdict: Verdict,
    label: str,
    value: float,
    reference: float,
    tolerance_low: float,
    tolerance_high: float | None = None,
) -> None:
    """Check one metric against a band around the real reference.

    Args:
        verdict: Verdict to record into.
        label: Metric name for the message.
        value: Measured value.
        reference: Real-domain reference value.
        tolerance_low: Allowed shortfall below the reference.
        tolerance_high: Allowed excess above the reference (defaults to
            ``tolerance_low``).
    """
    high = tolerance_low if tolerance_high is None else tolerance_high
    low_bound, high_bound = reference - tolerance_low, reference + high
    ok = bool(np.isfinite(value)) and low_bound <= value <= high_bound
    verdict.check(ok, f"{label} = {value:.3f} in [{low_bound:.3f}, {high_bound:.3f}] (real {reference:.3f})")


def print_report(pipeline: str, stats: dict[str, Any], reference_name: str, reference: dict[str, float]) -> None:
    """Print the aligned per-domain comparison table.

    Args:
        pipeline: Domain that was measured.
        stats: Statistics from ``summarize``.
        reference_name: Name of the active reference.
        reference: Active reference values.
    """
    rows: list[list[str]] = []
    for name, values in (*BASELINES.items(), ("doc section 4.3 (real)", REFERENCES["doc"])):
        rows.append([name, "-", *(f"{values[key]:.3f}" for key in ("gt_median", "high_fraction", "decoy_median"))])
        rows[-1].append(f"{values['margin']:.3f}")
    rows.append(
        [
            f"{pipeline} (this run)",
            str(stats["n_gt"]),
            f"{stats['gt_median']:.3f}",
            f"{stats['high_fraction']:.3f}",
            f"{stats['decoy_median']:.3f}",
            f"{stats['margin']:.3f}",
        ]
    )
    print("\n=== masked zero-mean NCC at the ground-truth cell (doc section 4.3) ===")
    print(format_table(["domain", "n", "GT median", "frac > hi", "decoy med", "margin"], rows))
    print(
        f"\nthis run: GT IQR [{stats['gt_p25']:.3f}, {stats['gt_p75']:.3f}]  "
        f"px/cell {stats['median_px_per_cell']:.0f}  candidates/piece {stats['candidates_per_piece']:.0f}"
    )
    print(f"reference for the gate: {reference_name} -> {reference}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="NCC-headroom acceptance gate for the exp31 pixel-identity fix (doc sections 4.3, 8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pipeline", choices=PIPELINES, default="exp31", help="Domain to measure")
    parser.add_argument("--piece-root", type=Path, default=None, help="Stored piece root (default: per-pipeline)")
    parser.add_argument("--puzzle-root", type=Path, default=None, help="Source puzzle JPEG root")
    parser.add_argument("--real-dataset-root", type=Path, default=None, help="north_star v1 root (default: MAIN)")
    parser.add_argument("--real-cache-dir", type=Path, default=None, help="north_star eval cache (default: MAIN)")
    parser.add_argument("--real-cell-px", type=int, default=DEFAULT_REAL_CELL_PX, help="Real overview px per cell")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="Pieces to sample (spread over puzzles)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling / augmentation / decoy seed")
    parser.add_argument("--workers", type=int, default=8, help="Worker processes (1 = in-process)")
    parser.add_argument("--scales", type=int, default=DEFAULT_SCALES, help="Scale steps")
    parser.add_argument("--scale-span", type=float, default=DEFAULT_SCALE_SPAN, help="Scale half-range factor")
    parser.add_argument("--rotations", type=int, default=DEFAULT_ROTATIONS, help="Rotation steps")
    parser.add_argument("--rotation-span", type=float, default=DEFAULT_ROTATION_SPAN, help="Rotation half-range (deg)")
    parser.add_argument("--erode", type=int, default=DEFAULT_ERODE, help="Mask erosion passes before correlating")
    parser.add_argument("--alpha-threshold", type=int, default=DEFAULT_ALPHA_THRESHOLD, help="Alpha > this is content")
    parser.add_argument(
        "--content-threshold", type=float, default=DEFAULT_CONTENT_THRESHOLD, help="Max channel > this is content"
    )
    parser.add_argument("--high-ncc", type=float, default=DEFAULT_HIGH_NCC, help="Threshold for the tail metric")
    parser.add_argument("--reference", choices=sorted(REFERENCES), default="measured", help="Built-in gate reference")
    parser.add_argument(
        "--reference-json", type=Path, default=None, help="Gate against a previous --pipeline real dump"
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write machine-readable results here")
    return parser


def load_reference(args: argparse.Namespace) -> tuple[str, dict[str, float]]:
    """Resolve the gate reference from the built-ins or a previous JSON dump.

    Args:
        args: Parsed arguments.

    Returns:
        Tuple of (reference name, reference values).

    Raises:
        ValueError: When the JSON dump is not a ``--pipeline real`` measurement
            or lacks the required statistics.
    """
    if args.reference_json is None:
        return args.reference, REFERENCES[args.reference]
    payload = json.loads(args.reference_json.read_text())
    if payload.get("config", {}).get("pipeline") != "real":
        raise ValueError(
            f"{args.reference_json} is not a '--pipeline real' measurement; the gate needs the real domain"
        )
    stats = payload.get("stats", {})
    keys = ("gt_median", "high_fraction", "decoy_median", "margin")
    missing = [key for key in keys if not isinstance(stats.get(key), (int, float))]
    if missing:
        raise ValueError(f"{args.reference_json} is missing statistics {missing}")
    return f"json:{args.reference_json.name}", {key: float(stats[key]) for key in keys}


def build_config(args: argparse.Namespace) -> ProbeConfig:
    """Assemble the worker configuration from parsed arguments.

    Args:
        args: Parsed arguments.

    Returns:
        The probe configuration.
    """
    real_root, real_cache = real_dataset_defaults()
    piece_relative = PIECE_ROOTS.get(args.pipeline, PIECE_ROOTS["exp30"])
    return ProbeConfig(
        pipeline=args.pipeline,
        piece_root=args.piece_root or resolve_dataset(piece_relative),
        puzzle_root=args.puzzle_root or resolve_dataset("puzzles"),
        real_dataset_root=args.real_dataset_root or real_root,
        real_cache_dir=args.real_cache_dir or real_cache,
        grid=SearchGrid.build(args.scales, args.scale_span, args.rotations, args.rotation_span),
        erode=args.erode,
        alpha_threshold=args.alpha_threshold,
        content_threshold=args.content_threshold,
        real_cell_px=args.real_cell_px,
        seed=args.seed,
    )


def _dump_json(path: Path, args: argparse.Namespace, payload: dict[str, Any]) -> None:
    """Write the machine-readable result file.

    Args:
        path: Destination path.
        args: Parsed arguments (recorded as the run config).
        payload: Statistics, verdict and reference.
    """
    payload["config"] = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items() if key != "json"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    """Measure one domain, print the table and gate on it.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 when every gated condition holds, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    reference_name, reference = load_reference(args)
    config = build_config(args)

    if args.pipeline == "exp31":
        # Resolve the sibling's interface up front: a RuntimeError raised inside
        # a worker arrives wrapped in a process-pool traceback, which buries the
        # message that says exactly what is missing.
        hooks = resolve_exp31_hooks()
        print(
            f"exp31 hooks: {hooks.module_name}.{hooks.piece_name} "
            f"(rgba={hooks.piece_returns_rgba}) + .{hooks.overview_name}, config={type(hooks.config).__name__}"
        )

    jobs = plan_real_jobs(config, args.sample, args.seed) if args.pipeline == "real" else _plan(config, args)
    n_pieces = sum(len(job["pieces"]) for job in jobs)
    print(f"pipeline={args.pipeline}  pieces={n_pieces} from {len(jobs)} puzzles (seed={args.seed})")
    print(f"grid: {len(config.grid.scales)} scales x {len(config.grid.angles)} rotations, erode={config.erode}")
    print(f"piece root: {config.piece_root if args.pipeline != 'real' else config.real_cache_dir}")
    print(f"overview source: {config.puzzle_root if args.pipeline != 'real' else config.real_dataset_root}")

    start = time.time()
    records = run_all(config, jobs, args.workers)
    elapsed = time.time() - start
    stats = summarize(records, config.grid, args.high_ncc)
    stats["seconds"] = elapsed

    print_report(args.pipeline, stats, reference_name, reference)
    verdict = gate(stats, reference, args.high_ncc)
    print(f"\n=== verdict ({args.pipeline}) ===")
    for reason in verdict.reasons:
        print(f"    {reason}")
    print(f"\n{elapsed:.0f}s for {stats['n_pieces']} pieces")
    print(f"OVERALL: {'PASS' if verdict.passed else 'FAIL'}")

    if args.json is not None:
        _dump_json(
            args.json,
            args,
            {
                "pipeline": args.pipeline,
                "stats": stats,
                "reference": {"name": reference_name, **reference},
                "reasons": verdict.reasons,
                "overall_passed": verdict.passed,
            },
        )
    return 0 if verdict.passed else 1


def _plan(config: ProbeConfig, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Plan the synthetic jobs (thin wrapper so ``main`` stays readable).

    Args:
        config: Probe configuration.
        args: Parsed arguments.

    Returns:
        Job dicts.
    """
    return plan_synthetic_jobs(config, args.sample, args.seed)


if __name__ == "__main__":
    sys.exit(main())
