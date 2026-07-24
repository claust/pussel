#!/usr/bin/env python3
r"""Score how well the iOS app's glare-free stitching worked on one capture dump.

Consumes a DEBUG-build capture-dump directory (`reference.jpg`, `composite.jpg`,
0-4 `corner_N.jpg`, optional `metadata.json`; see `common.py`) and scores the
app's `composite.jpg` against `reference.jpg` on seven axes:

- **Global geometry**: SIFT + RANSAC homography between composite and reference
  (should be ~identity); inlier count/ratio and RMS/p95 pixel deviation from identity.
- **Local ghosting**: a grid of ~64px patches, phase-correlation shift per patch;
  median/p95 shift magnitude, plus a rendered heatmap. Patches whose region was
  substantially DARKENED (healed) rather than misaligned are excluded from the shift
  statistics -- see `compute_darkening_map` -- so a healed glare sheen can't masquerade
  as ghosting.
- **Edge doubling**: Canny edge-pixel-count ratio and mean gradient-magnitude ratio
  (composite/reference) -- doubled edges push both above ~1.
- **Sharpness**: variance-of-Laplacian ratio (composite/reference).
- **Glare reduction**: near-saturated (gray >= 250) pixel fraction in reference vs
  composite. Blind to matte-print glare, which desaturates the artwork without ever
  reaching outright saturation -- see glare healing below for that case.
- **Glare healing**: per-pixel darkening the composite achieved relative to the
  reference (`max(0, reference - composite)`, the app's only source of legitimate
  per-pixel change since a min-composite can only darken). This is the primary
  glare-reduction benefit metric -- it catches matte-print glare the saturation
  metric above misses, and a stitch can't win it by returning the reference unmodified.
  Pixels covered by a detected reference bright speck (see below) are excluded from
  this metric's darkening map, so a min-composite deleting fine bright detail (e.g. a
  star on a dark background, via a 1-3px misregistration picking a neighboring frame's
  darker sky pixel) doesn't get counted as "healing".
- **Bright detail preservation**: small bright specks (e.g. stars) detected
  independently in composite and reference via a white top-hat filter; reports each
  image's speck count and the composite/reference retention ratio, both whole-frame and
  restricted to patches NOT excluded as healed (since a speck actually lost to glare was
  unrecoverable regardless of stitch quality). Catches wholesale deletion of fine bright
  detail that the other six axes barely register (a 1-3px misalignment lets one frame's
  darker background pixel consistently win the darkest-pixel-wins comparison at a star's
  location).

`--quad "x1,y1 x2,y2 x3,y3 x4,y4"` (unit coordinates, clockwise from top-left) restricts
all seven axes to a region of interest (e.g. the puzzle itself) reported alongside the
unrestricted full-frame numbers -- useful since ghosting/edge stats are otherwise easily
dominated by textured background (e.g. carpet) patches that have nothing to do with the
puzzle.

Usage:
    cd network
    uv run python experiments/exp29_stitch_quality/score_stitch.py /path/to/dump
    uv run python experiments/exp29_stitch_quality/score_stitch.py /path/to/dump --out /path/to/report

    # Restrict stats to a region (e.g. the puzzle itself), reported alongside full-frame:
    uv run python experiments/exp29_stitch_quality/score_stitch.py /path/to/dump \
        --quad "0.1,0.05 0.9,0.05 0.9,0.95 0.1,0.95"
"""

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import (  # noqa: E402
    DARKENED_PIXEL_THRESHOLD,
    HEALED_PATCH_DARKENING_THRESHOLD,
    SPECK_TOPHAT_KERNEL_PX,
    CaptureDump,
    FeatureMatchResult,
    PatchShift,
    SpeckDetection,
    canny_edge_count,
    compute_darkening_map,
    detect_bright_specks,
    load_dump,
    match_sift_ransac,
    mean_gradient_magnitude,
    near_saturated_fraction,
    parse_quad,
    phase_correlation_grid,
    quad_to_mask,
    variance_of_laplacian,
)

DEFAULT_OUTPUT_SUBDIR = "score_stitch_output"

# Visual crop context around the worst patch in the flicker image, and its upscale factor.
FLICKER_CROP_MARGIN_PATCHES = 1
FLICKER_UPSCALE = 4

# Ghost-heatmap color scale: shift magnitudes at/above this many pixels saturate to the
# colormap's top end, so a handful of extreme outlier patches don't wash out the rest.
GHOST_HEATMAP_MAX_PX = 8.0


@dataclass(frozen=True)
class GlobalGeometryMetrics:
    """SIFT + RANSAC homography metrics between composite and reference.

    Attributes:
        n_keypoints_composite: SIFT keypoints detected in the composite (full frame --
            SIFT is not rerun per `--quad` region).
        n_keypoints_reference: SIFT keypoints detected in the reference (full frame).
        n_ratio_matches: Matches surviving Lowe's ratio test (full frame; the same
            denominator is used for `inlier_ratio` in both full-frame and region reports).
        n_inliers: RANSAC inlier matches (restricted to the region when one is given).
        inlier_ratio: `n_inliers / n_ratio_matches` (0 when there are no matches).
        identity_deviation_rms_px: RMS pixel distance between each RANSAC-inlier match's
            composite and reference coordinates (0 would mean perfect pixel alignment).
            None when there were too few matches to run RANSAC, or none fell in the region.
        identity_deviation_p95_px: 95th percentile of the same per-match distances. None
            under the same condition as `identity_deviation_rms_px`.
    """

    n_keypoints_composite: int
    n_keypoints_reference: int
    n_ratio_matches: int
    n_inliers: int
    inlier_ratio: float
    identity_deviation_rms_px: Optional[float]
    identity_deviation_p95_px: Optional[float]

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "n_keypoints_composite": self.n_keypoints_composite,
            "n_keypoints_reference": self.n_keypoints_reference,
            "n_ratio_matches": self.n_ratio_matches,
            "n_inliers": self.n_inliers,
            "inlier_ratio": self.inlier_ratio,
            "identity_deviation_rms_px": self.identity_deviation_rms_px,
            "identity_deviation_p95_px": self.identity_deviation_p95_px,
        }


def compute_global_geometry_from_match(
    result: FeatureMatchResult, mask: Optional[np.ndarray] = None
) -> GlobalGeometryMetrics:
    """Derive global-geometry metrics from an already-computed SIFT+RANSAC match result.

    Reuses one `FeatureMatchResult` for both the full-frame report and a `--quad` region
    report (subsetting by mask instead of rerunning SIFT+RANSAC per region), so the region
    numbers are a strict subset of the same underlying fit rather than an independent,
    potentially inconsistent one.

    Args:
        result: A `common.FeatureMatchResult` between composite (src) and reference (dst).
        mask: Optional region mask (working-size, nonzero = inside); when given, only
            RANSAC inliers whose composite-side coordinate falls inside it are counted.

    Returns:
        The computed `GlobalGeometryMetrics`.
    """
    if result.inlier_mask is None or result.n_inliers == 0:
        return GlobalGeometryMetrics(
            n_keypoints_composite=result.n_keypoints_src,
            n_keypoints_reference=result.n_keypoints_dst,
            n_ratio_matches=result.n_ratio_matches,
            n_inliers=0,
            inlier_ratio=0.0,
            identity_deviation_rms_px=None,
            identity_deviation_p95_px=None,
        )

    inlier_src = result.src_pts[result.inlier_mask]
    inlier_dst = result.dst_pts[result.inlier_mask]

    if mask is not None:
        in_region = _points_in_mask(inlier_src, mask)
        inlier_src = inlier_src[in_region]
        inlier_dst = inlier_dst[in_region]

    n_inliers = len(inlier_src)
    if n_inliers == 0:
        return GlobalGeometryMetrics(
            n_keypoints_composite=result.n_keypoints_src,
            n_keypoints_reference=result.n_keypoints_dst,
            n_ratio_matches=result.n_ratio_matches,
            n_inliers=0,
            inlier_ratio=0.0,
            identity_deviation_rms_px=None,
            identity_deviation_p95_px=None,
        )

    deviations = np.linalg.norm(inlier_dst - inlier_src, axis=1)
    return GlobalGeometryMetrics(
        n_keypoints_composite=result.n_keypoints_src,
        n_keypoints_reference=result.n_keypoints_dst,
        n_ratio_matches=result.n_ratio_matches,
        n_inliers=n_inliers,
        inlier_ratio=n_inliers / result.n_ratio_matches if result.n_ratio_matches else 0.0,
        identity_deviation_rms_px=float(np.sqrt(np.mean(deviations**2))),
        identity_deviation_p95_px=float(np.percentile(deviations, 95)),
    )


def compute_global_geometry(
    gray_composite: np.ndarray, gray_reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> GlobalGeometryMetrics:
    """Run SIFT+RANSAC and compute the global-geometry metrics in one call.

    Convenience wrapper for standalone use (e.g. tests); `score_dump` calls
    `match_sift_ransac` once and reuses the result via `compute_global_geometry_from_match`
    instead, so full-frame and `--quad`-region stats come from the same fit.

    Args:
        gray_composite: Composite image, single channel, working size.
        gray_reference: Reference image, single channel, same size as `gray_composite`.
        mask: Optional region mask, see `compute_global_geometry_from_match`.

    Returns:
        The computed `GlobalGeometryMetrics`.
    """
    return compute_global_geometry_from_match(match_sift_ransac(gray_composite, gray_reference), mask)


def _points_in_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Test which (x, y) points fall on a nonzero pixel of `mask`.

    Args:
        points: (N, 2) array of (x, y) pixel coordinates.
        mask: uint8/bool mask, working size.

    Returns:
        Boolean array (N,), True where the (rounded, clamped) point falls inside `mask`.
    """
    height, width = mask.shape[:2]
    xs = np.clip(points[:, 0].round().astype(int), 0, width - 1)
    ys = np.clip(points[:, 1].round().astype(int), 0, height - 1)
    return mask[ys, xs] > 0


@dataclass(frozen=True)
class WorstPatch:
    """The grid patch with the largest local composite-vs-reference shift.

    Attributes:
        row: Patch row index.
        col: Patch column index.
        x: Patch's left edge, in working-size pixels.
        y: Patch's top edge, in working-size pixels.
        shift_px: The patch's shift magnitude, in pixels.
    """

    row: int
    col: int
    x: int
    y: int
    shift_px: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {"row": self.row, "col": self.col, "x": self.x, "y": self.y, "shift_px": self.shift_px}


@dataclass(frozen=True)
class LocalGhostingMetrics:
    """Patch-grid phase-correlation ghosting metrics.

    Attributes:
        patch_size_px: Grid patch side length used.
        n_patches: Total number of patches considered (the whole grid, or the patches
            whose center falls in the `--quad` region when one is given).
        n_valid_patches: Of those, patches with enough texture for a trustworthy shift
            estimate (`PatchShift.valid`).
        healed_patches: Of the valid patches, how many were excluded from the shift
            statistics as "healed" (substantially darkened -- see `compute_darkening_map`)
            rather than misaligned.
        median_shift_px: Median shift magnitude over valid, non-healed patches. None if
            there were none.
        p95_shift_px: 95th-percentile shift magnitude over the same set. None if there
            were none.
        worst_patch: The valid, non-healed patch with the largest shift, or None if
            there were none.
    """

    patch_size_px: int
    n_patches: int
    n_valid_patches: int
    healed_patches: int
    median_shift_px: Optional[float]
    p95_shift_px: Optional[float]
    worst_patch: Optional[WorstPatch]

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "patch_size_px": self.patch_size_px,
            "n_patches": self.n_patches,
            "n_valid_patches": self.n_valid_patches,
            "healed_patches": self.healed_patches,
            "median_shift_px": self.median_shift_px,
            "p95_shift_px": self.p95_shift_px,
            "worst_patch": self.worst_patch.to_dict() if self.worst_patch else None,
        }


def compute_local_ghosting(
    gray_composite: np.ndarray,
    gray_reference: np.ndarray,
    darkening_map: np.ndarray,
    patch_size: int = 64,
    healed_darkening_threshold: float = HEALED_PATCH_DARKENING_THRESHOLD,
    mask: Optional[np.ndarray] = None,
) -> Tuple[LocalGhostingMetrics, List[PatchShift]]:
    """Compute the local-ghosting metrics for one composite/reference pair.

    Every patch is annotated `healed=True` when its mean darkening (from `darkening_map`)
    exceeds `healed_darkening_threshold`: phase correlation reacts to ANY appearance
    change, including a region the composite legitimately healed (glare removed), so
    those patches are excluded from the shift statistics and from `worst_patch` --
    otherwise the "worst" patch is often just the most dramatically healed one, not the
    most misaligned one.

    Args:
        gray_composite: Composite image, single channel, working size.
        gray_reference: Reference image, single channel, same size as `gray_composite`.
        darkening_map: Per-pixel darkening from `common.compute_darkening_map`, same size.
        patch_size: Grid patch side length in pixels.
        healed_darkening_threshold: Per-patch mean darkening above which a patch is
            excluded as healed rather than scored as ghosting.
        mask: Optional region mask (working-size, nonzero = inside); when given, only
            patches whose center falls inside it are considered at all.

    Returns:
        Tuple of (`LocalGhostingMetrics`, the annotated per-patch grid -- the full,
        unfiltered grid when `mask` is None; this is what `render_ghost_heatmap` and
        `save_worst_patch_flicker` need for the full-frame report).
    """
    raw_patches = phase_correlation_grid(gray_composite, gray_reference, patch_size)
    patches: List[PatchShift] = []
    for patch in raw_patches:
        patch_darkening = darkening_map[patch.y : patch.y + patch_size, patch.x : patch.x + patch_size]
        healed = bool(patch_darkening.size) and float(patch_darkening.mean()) > healed_darkening_threshold
        patches.append(replace(patch, healed=healed))

    considered = patches
    if mask is not None:
        considered = [p for p in patches if _mask_value(mask, p.x + patch_size // 2, p.y + patch_size // 2)]

    usable = [p for p in considered if p.valid and not p.healed]
    healed_count = sum(1 for p in considered if p.valid and p.healed)

    if not usable:
        metrics = LocalGhostingMetrics(patch_size, len(considered), 0, healed_count, None, None, None)
        return metrics, patches

    magnitudes = np.array([p.shift_px for p in usable])
    worst = max(usable, key=lambda p: p.shift_px)
    metrics = LocalGhostingMetrics(
        patch_size_px=patch_size,
        n_patches=len(considered),
        n_valid_patches=len(usable) + healed_count,
        healed_patches=healed_count,
        median_shift_px=float(np.median(magnitudes)),
        p95_shift_px=float(np.percentile(magnitudes, 95)),
        worst_patch=WorstPatch(worst.row, worst.col, worst.x, worst.y, worst.shift_px),
    )
    return metrics, patches


def _mask_value(mask: np.ndarray, x: int, y: int) -> bool:
    """Look up whether a single (clamped) pixel coordinate falls on a nonzero `mask` pixel."""
    height, width = mask.shape[:2]
    return bool(mask[min(y, height - 1), min(x, width - 1)] > 0)


@dataclass(frozen=True)
class EdgeDoublingMetrics:
    """Canny/gradient edge-strength comparison between composite and reference.

    Attributes:
        canny_edge_ratio: Canny edge-pixel count, composite / reference.
        gradient_magnitude_ratio: Mean Sobel gradient magnitude, composite / reference.
    """

    canny_edge_ratio: float
    gradient_magnitude_ratio: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {"canny_edge_ratio": self.canny_edge_ratio, "gradient_magnitude_ratio": self.gradient_magnitude_ratio}


def compute_edge_doubling(
    gray_composite: np.ndarray, gray_reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> EdgeDoublingMetrics:
    """Compute the edge-doubling metrics for one composite/reference pair.

    Args:
        gray_composite: Composite image, single channel.
        gray_reference: Reference image, single channel, same size as `gray_composite`.
        mask: Optional region mask, see `common.canny_edge_count` / `mean_gradient_magnitude`.

    Returns:
        The computed `EdgeDoublingMetrics`.
    """
    ref_edges = max(canny_edge_count(gray_reference, mask=mask), 1)
    ref_gradient = max(mean_gradient_magnitude(gray_reference, mask=mask), 1e-6)
    return EdgeDoublingMetrics(
        canny_edge_ratio=canny_edge_count(gray_composite, mask=mask) / ref_edges,
        gradient_magnitude_ratio=mean_gradient_magnitude(gray_composite, mask=mask) / ref_gradient,
    )


@dataclass(frozen=True)
class SharpnessMetrics:
    """Variance-of-Laplacian sharpness comparison between composite and reference.

    Attributes:
        composite_laplacian_variance: Variance of the Laplacian of the composite.
        reference_laplacian_variance: Variance of the Laplacian of the reference.
        laplacian_variance_ratio: `composite_laplacian_variance / reference_laplacian_variance`.
    """

    composite_laplacian_variance: float
    reference_laplacian_variance: float
    laplacian_variance_ratio: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "composite_laplacian_variance": self.composite_laplacian_variance,
            "reference_laplacian_variance": self.reference_laplacian_variance,
            "laplacian_variance_ratio": self.laplacian_variance_ratio,
        }


def compute_sharpness(
    gray_composite: np.ndarray, gray_reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> SharpnessMetrics:
    """Compute the sharpness metrics for one composite/reference pair.

    Args:
        gray_composite: Composite image, single channel.
        gray_reference: Reference image, single channel.
        mask: Optional region mask, see `common.variance_of_laplacian`.

    Returns:
        The computed `SharpnessMetrics`.
    """
    composite_var = variance_of_laplacian(gray_composite, mask)
    reference_var = max(variance_of_laplacian(gray_reference, mask), 1e-6)
    return SharpnessMetrics(
        composite_laplacian_variance=composite_var,
        reference_laplacian_variance=reference_var,
        laplacian_variance_ratio=composite_var / reference_var,
    )


@dataclass(frozen=True)
class GlareReductionMetrics:
    """Near-saturated-pixel comparison between composite and reference.

    Blind to matte-print glare that desaturates the artwork without ever reaching
    outright saturation -- see `GlareHealingMetrics` for the metric that catches that.

    Attributes:
        reference_saturated_fraction: Fraction of near-saturated pixels in the reference.
        composite_saturated_fraction: Fraction of near-saturated pixels in the composite.
        reduction_factor: `reference_saturated_fraction / composite_saturated_fraction`.
            1.0 when neither image has any glare; +inf when the composite eliminated
            glare the reference had.
    """

    reference_saturated_fraction: float
    composite_saturated_fraction: float
    reduction_factor: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "reference_saturated_fraction": self.reference_saturated_fraction,
            "composite_saturated_fraction": self.composite_saturated_fraction,
            "reduction_factor": self.reduction_factor,
        }


def compute_glare_reduction(
    gray_composite: np.ndarray, gray_reference: np.ndarray, mask: Optional[np.ndarray] = None
) -> GlareReductionMetrics:
    """Compute the near-saturation glare-reduction metrics for one composite/reference pair.

    Args:
        gray_composite: Composite image, single channel.
        gray_reference: Reference image, single channel.
        mask: Optional region mask, see `common.near_saturated_fraction`.

    Returns:
        The computed `GlareReductionMetrics`.
    """
    ref_frac = near_saturated_fraction(gray_reference, mask=mask)
    comp_frac = near_saturated_fraction(gray_composite, mask=mask)
    if comp_frac == 0.0:
        reduction_factor = 1.0 if ref_frac == 0.0 else float("inf")
    else:
        reduction_factor = ref_frac / comp_frac
    return GlareReductionMetrics(
        reference_saturated_fraction=ref_frac, composite_saturated_fraction=comp_frac, reduction_factor=reduction_factor
    )


@dataclass(frozen=True)
class GlareHealingMetrics:
    """Per-pixel darkening the composite achieved relative to the reference.

    This is the primary glare-reduction benefit metric. A min-composite can only ever pick
    a darker (or equal) pixel than the reference at any location -- it never brightens --
    so `max(0, reference - composite)` is an unambiguous signal of what the stitch actually
    changed, catching matte-print glare (a desaturating gray wash that never reaches
    outright saturation) that `GlareReductionMetrics` misses entirely.

    Attributes:
        darkened_fraction: Fraction of pixels darkened by more than
            `common.DARKENED_PIXEL_THRESHOLD` (~8/255).
        mean_darkening_over_darkened: Mean darkening magnitude, restricted to those
            darkened pixels. 0.0 if none were darkened.
        p95_darkening: 95th percentile of darkening over ALL pixels in the region (not
            just the darkened ones) -- the overall right tail of the distribution.
    """

    darkened_fraction: float
    mean_darkening_over_darkened: float
    p95_darkening: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "darkened_fraction": self.darkened_fraction,
            "mean_darkening_over_darkened": self.mean_darkening_over_darkened,
            "p95_darkening": self.p95_darkening,
        }


def compute_glare_healing(
    darkening_map: np.ndarray, threshold: float = DARKENED_PIXEL_THRESHOLD, mask: Optional[np.ndarray] = None
) -> GlareHealingMetrics:
    """Compute the glare-healing metrics from a darkening map.

    Args:
        darkening_map: Per-pixel darkening from `common.compute_darkening_map`.
        threshold: Per-pixel darkening above which a pixel counts as "darkened".
        mask: Optional region mask (working-size, nonzero = inside); restricts to that region.

    Returns:
        The computed `GlareHealingMetrics`. All-zero if `mask` selects no pixels.
    """
    values = darkening_map[mask > 0] if mask is not None else darkening_map.ravel()
    if values.size == 0:
        return GlareHealingMetrics(0.0, 0.0, 0.0)

    darkened = values > threshold
    darkened_fraction = float(np.count_nonzero(darkened)) / values.size
    mean_over_darkened = float(values[darkened].mean()) if np.any(darkened) else 0.0
    p95 = float(np.percentile(values, 95))
    return GlareHealingMetrics(darkened_fraction, mean_over_darkened, p95)


@dataclass(frozen=True)
class BrightDetailMetrics:
    """Small bright speck (e.g. stars) preservation between composite and reference.

    Specks are detected independently in each image (see `common.detect_bright_specks`);
    this reports a count-based retention ratio rather than tracking individual specks across
    images, since a 1-3px misregistration can shift a speck's exact location slightly while
    it's still clearly present -- exact spatial matching would conflate "moved a little" with
    "erased", which is not the failure mode this metric targets (a min-composite deleting
    fine bright detail wholesale because a neighboring frame's darker background pixel wins
    the darkest-pixel-wins comparison at that exact location).

    Attributes:
        reference_speck_count: Small bright specks detected in the reference (this region).
        composite_speck_count: Same, in the composite.
        retention_ratio: `composite_speck_count / reference_speck_count`. 1.0 when the
            reference had none either; +inf when the reference had none but the composite
            gained some (spurious detections, e.g. seam noise).
        reference_speck_count_excl_healed: Reference specks whose centroid falls outside any
            patch excluded as "healed" by `LocalGhostingMetrics` (see `compute_local_ghosting`).
        composite_speck_count_excl_healed: Same, for the composite.
        retention_ratio_excl_healed: Same ratio as `retention_ratio`, over the
            healed-patch-excluded counts. A speck actually lost to glare (inside a healed
            patch) was unrecoverable regardless of stitch quality, so counting its loss
            against the composite would penalize correct behavior -- this is the ratio that
            isolates deletion caused by misregistration rather than legitimate healing.
    """

    reference_speck_count: int
    composite_speck_count: int
    retention_ratio: float
    reference_speck_count_excl_healed: int
    composite_speck_count_excl_healed: int
    retention_ratio_excl_healed: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "reference_speck_count": self.reference_speck_count,
            "composite_speck_count": self.composite_speck_count,
            "retention_ratio": self.retention_ratio,
            "reference_speck_count_excl_healed": self.reference_speck_count_excl_healed,
            "composite_speck_count_excl_healed": self.composite_speck_count_excl_healed,
            "retention_ratio_excl_healed": self.retention_ratio_excl_healed,
        }


def _retention_ratio(composite_count: int, reference_count: int) -> float:
    """`composite_count / reference_count`, defined at zero the same way `GlareReductionMetrics` is.

    Args:
        composite_count: Speck count in the composite (or region/subset thereof).
        reference_count: Speck count in the reference (or region/subset thereof).

    Returns:
        The ratio; 1.0 when `reference_count` is 0 and `composite_count` is also 0 (nothing
        to retain, nothing lost); +inf when `reference_count` is 0 but `composite_count` isn't
        (specks gained, not lost).
    """
    if reference_count == 0:
        return 1.0 if composite_count == 0 else float("inf")
    return composite_count / reference_count


def compute_bright_detail(
    reference_specks: SpeckDetection,
    composite_specks: SpeckDetection,
    healed_pixel_mask: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> BrightDetailMetrics:
    """Compute the bright-detail-preservation metrics from already-detected specks.

    Reuses one `SpeckDetection` per image for both the full-frame report and a `--quad`
    region report (subsetting by centroid location instead of rerunning detection per
    region), mirroring how `compute_global_geometry_from_match` reuses one SIFT+RANSAC fit.

    Args:
        reference_specks: `common.detect_bright_specks` result for the reference image
            (full frame -- not rerun per `--quad` region).
        composite_specks: Same, for the composite image.
        healed_pixel_mask: Working-size mask, nonzero where a local-ghosting grid patch was
            excluded as "healed" (see `_healed_pixel_mask`); used to compute the
            `*_excl_healed` fields.
        mask: Optional region mask (working-size, nonzero = inside); when given, only specks
            whose centroid falls inside it are counted at all.

    Returns:
        The computed `BrightDetailMetrics`.
    """
    if mask is not None:
        ref_in_region = _points_in_mask(reference_specks.centroids, mask)
        comp_in_region = _points_in_mask(composite_specks.centroids, mask)
    else:
        ref_in_region = np.ones(reference_specks.count, dtype=bool)
        comp_in_region = np.ones(composite_specks.count, dtype=bool)

    ref_healed = _points_in_mask(reference_specks.centroids, healed_pixel_mask)
    comp_healed = _points_in_mask(composite_specks.centroids, healed_pixel_mask)

    ref_count = int(np.count_nonzero(ref_in_region))
    comp_count = int(np.count_nonzero(comp_in_region))
    ref_count_excl = int(np.count_nonzero(ref_in_region & ~ref_healed))
    comp_count_excl = int(np.count_nonzero(comp_in_region & ~comp_healed))

    return BrightDetailMetrics(
        reference_speck_count=ref_count,
        composite_speck_count=comp_count,
        retention_ratio=_retention_ratio(comp_count, ref_count),
        reference_speck_count_excl_healed=ref_count_excl,
        composite_speck_count_excl_healed=comp_count_excl,
        retention_ratio_excl_healed=_retention_ratio(comp_count_excl, ref_count_excl),
    )


def _healed_pixel_mask(patches: List[PatchShift], patch_size: int, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize which pixels fall in a local-ghosting grid patch marked `healed=True`.

    Used to compute the `*_excl_healed` bright-detail fields -- a speck lost purely to
    glare-healing (unrecoverable regardless of stitch quality) shouldn't count against
    retention.

    Args:
        patches: The full, unfiltered per-patch grid from `compute_local_ghosting` (as
            returned alongside `LocalGhostingMetrics` -- NOT a `--quad`-restricted subset;
            a patch's healed status doesn't depend on which region is being reported).
        patch_size: Grid patch side length in pixels (must match `patches`).
        shape: `(height, width)` of the working-size images.

    Returns:
        uint8 mask, 255 inside a healed patch, 0 elsewhere (including any trailing rows/
        columns the patch grid doesn't cover).
    """
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for patch in patches:
        if patch.valid and patch.healed:
            mask[patch.y : patch.y + patch_size, patch.x : patch.x + patch_size] = 255
    return mask


def _darkening_map_excluding_specks(darkening_map: np.ndarray, reference_speck_mask: np.ndarray) -> np.ndarray:
    """Zero out darkening attributed to deleted reference bright specks, before glare-healing stats.

    A min-composite can erase a small bright detail (e.g. a star on a dark background) simply
    because a 1-3px misregistration lets a neighboring frame's darker background pixel win
    the darkest-pixel-wins comparison at that exact location. Left alone, that reads as
    legitimate "darkening" to `compute_glare_healing` even though nothing was healed -- fine
    bright detail was just deleted. Zeroing the darkening map at every detected
    reference-speck pixel removes that false credit before the glare-healing stats are
    computed (local ghosting's healed-patch classification is unaffected -- it still uses the
    unmodified darkening map).

    The speck mask is dilated by `common.SPECK_TOPHAT_KERNEL_PX` first: `darkening_map` itself
    comes from two Gaussian-blurred images (`common.DARKENING_BLUR_SIGMA`), which spreads a
    deleted speck's brightness difference a couple pixels past the speck's own detected
    pixels -- without the margin, that thin blur halo survives as residual "darkening" right
    at the speck's rim.

    Args:
        darkening_map: Per-pixel darkening from `common.compute_darkening_map`.
        reference_speck_mask: `SpeckDetection.mask` for the REFERENCE image (nonzero = a
            detected speck pixel there).

    Returns:
        A copy of `darkening_map` with speck-pixel locations (and a small margin around them)
        zeroed.
    """
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SPECK_TOPHAT_KERNEL_PX, SPECK_TOPHAT_KERNEL_PX))
    excluded = cv2.dilate(reference_speck_mask, dilation_kernel)
    result = darkening_map.copy()
    result[excluded > 0] = 0.0
    return result


@dataclass(frozen=True)
class RegionMetrics:
    """The same seven metric axes as the full-frame report, computed over a `--quad` region.

    Attributes:
        n_pixels: Number of working-size pixels inside the region.
        global_geometry: See `GlobalGeometryMetrics`.
        local_ghosting: See `LocalGhostingMetrics`.
        edge_doubling: See `EdgeDoublingMetrics`.
        sharpness: See `SharpnessMetrics`.
        glare_reduction: See `GlareReductionMetrics`.
        glare_healing: See `GlareHealingMetrics`.
        bright_detail: See `BrightDetailMetrics`.
    """

    n_pixels: int
    global_geometry: GlobalGeometryMetrics
    local_ghosting: LocalGhostingMetrics
    edge_doubling: EdgeDoublingMetrics
    sharpness: SharpnessMetrics
    glare_reduction: GlareReductionMetrics
    glare_healing: GlareHealingMetrics
    bright_detail: BrightDetailMetrics

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return {
            "n_pixels": self.n_pixels,
            "global_geometry": self.global_geometry.to_dict(),
            "local_ghosting": self.local_ghosting.to_dict(),
            "edge_doubling": self.edge_doubling.to_dict(),
            "sharpness": self.sharpness.to_dict(),
            "glare_reduction": self.glare_reduction.to_dict(),
            "glare_healing": self.glare_healing.to_dict(),
            "bright_detail": self.bright_detail.to_dict(),
        }


@dataclass(frozen=True)
class StitchQualityReport:
    """Full stitch-quality report for one capture dump.

    Attributes:
        dump_dir: Source dump directory, as a string.
        metadata: Parsed `metadata.json` contents, or None if it was absent.
        corner_frame_count: Number of `corner_N.jpg` files found (0-4).
        global_geometry: Full-frame. See `GlobalGeometryMetrics`.
        local_ghosting: Full-frame. See `LocalGhostingMetrics`.
        edge_doubling: Full-frame. See `EdgeDoublingMetrics`.
        sharpness: Full-frame. See `SharpnessMetrics`.
        glare_reduction: Full-frame. See `GlareReductionMetrics`.
        glare_healing: Full-frame. See `GlareHealingMetrics`.
        bright_detail: Full-frame. See `BrightDetailMetrics`.
        quad: The `--quad` unit coordinates used for `region`, or None if not given.
        region: The same seven axes restricted to `quad`, or None if `--quad` wasn't given.
    """

    dump_dir: str
    metadata: Optional[dict]
    corner_frame_count: int
    global_geometry: GlobalGeometryMetrics
    local_ghosting: LocalGhostingMetrics
    edge_doubling: EdgeDoublingMetrics
    sharpness: SharpnessMetrics
    glare_reduction: GlareReductionMetrics
    glare_healing: GlareHealingMetrics
    bright_detail: BrightDetailMetrics
    quad: Optional[List[List[float]]]
    region: Optional[RegionMetrics]

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict, matching `metrics.json`'s on-disk shape."""
        return {
            "dump_dir": self.dump_dir,
            "metadata": self.metadata,
            "corner_frame_count": self.corner_frame_count,
            "global_geometry": self.global_geometry.to_dict(),
            "local_ghosting": self.local_ghosting.to_dict(),
            "edge_doubling": self.edge_doubling.to_dict(),
            "sharpness": self.sharpness.to_dict(),
            "glare_reduction": self.glare_reduction.to_dict(),
            "glare_healing": self.glare_healing.to_dict(),
            "bright_detail": self.bright_detail.to_dict(),
            "quad": self.quad,
            "region": self.region.to_dict() if self.region is not None else None,
        }


def score_dump(
    dump: CaptureDump, patch_size: int = 64, quad_unit: Optional[np.ndarray] = None
) -> Tuple[StitchQualityReport, List[PatchShift]]:
    """Compute the full stitch-quality report for a loaded capture dump.

    Args:
        dump: A `CaptureDump` loaded by `common.load_dump`.
        patch_size: Grid patch side length in pixels for the local-ghosting metrics.
        quad_unit: Optional (4, 2) unit-coordinate quad (see `common.parse_quad`); when
            given, `report.region` reports the same six axes restricted to it.

    Returns:
        Tuple of (the `StitchQualityReport`, the full-frame ghosting patch grid -- needed
        by `render_ghost_heatmap` and `save_worst_patch_flicker`).
    """
    gray_composite = cv2.cvtColor(dump.composite, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(dump.reference, cv2.COLOR_BGR2GRAY)
    height, width = gray_reference.shape[:2]

    match_result = match_sift_ransac(gray_composite, gray_reference)
    darkening_map = compute_darkening_map(gray_composite, gray_reference)

    global_geometry = compute_global_geometry_from_match(match_result)
    local_ghosting, patches = compute_local_ghosting(gray_composite, gray_reference, darkening_map, patch_size)
    edge_doubling = compute_edge_doubling(gray_composite, gray_reference)
    sharpness = compute_sharpness(gray_composite, gray_reference)
    glare_reduction = compute_glare_reduction(gray_composite, gray_reference)

    reference_specks = detect_bright_specks(gray_reference)
    composite_specks = detect_bright_specks(gray_composite)
    healed_pixel_mask = _healed_pixel_mask(patches, patch_size, (height, width))
    darkening_map_excl_specks = _darkening_map_excluding_specks(darkening_map, reference_specks.mask)

    glare_healing = compute_glare_healing(darkening_map_excl_specks)
    bright_detail = compute_bright_detail(reference_specks, composite_specks, healed_pixel_mask)

    region: Optional[RegionMetrics] = None
    quad_unit_list: Optional[List[List[float]]] = None
    if quad_unit is not None:
        mask = quad_to_mask(quad_unit, width, height)
        region_ghosting, _region_patches = compute_local_ghosting(
            gray_composite, gray_reference, darkening_map, patch_size, mask=mask
        )
        region = RegionMetrics(
            n_pixels=int(np.count_nonzero(mask)),
            global_geometry=compute_global_geometry_from_match(match_result, mask),
            local_ghosting=region_ghosting,
            edge_doubling=compute_edge_doubling(gray_composite, gray_reference, mask),
            sharpness=compute_sharpness(gray_composite, gray_reference, mask),
            glare_reduction=compute_glare_reduction(gray_composite, gray_reference, mask),
            glare_healing=compute_glare_healing(darkening_map_excl_specks, mask=mask),
            bright_detail=compute_bright_detail(reference_specks, composite_specks, healed_pixel_mask, mask=mask),
        )
        quad_unit_list = quad_unit.tolist()

    report = StitchQualityReport(
        dump_dir=str(dump.dump_dir),
        metadata=dump.metadata,
        corner_frame_count=len(dump.corners),
        global_geometry=global_geometry,
        local_ghosting=local_ghosting,
        edge_doubling=edge_doubling,
        sharpness=sharpness,
        glare_reduction=glare_reduction,
        glare_healing=glare_healing,
        bright_detail=bright_detail,
        quad=quad_unit_list,
        region=region,
    )
    return report, patches


def render_ghost_heatmap(dump: CaptureDump, patches: List[PatchShift], out_path: Path) -> None:
    """Render the per-patch shift-magnitude heatmap over the composite and save it.

    Healed patches (excluded from the shift statistics -- see `compute_local_ghosting`)
    are rendered as a translucent green overlay instead of a shift-magnitude color, so
    they read visually as "explained by healing", distinct from both real ghosting
    (inferno colors) and skipped-uniform patches (no overlay at all).

    Args:
        dump: The scored capture dump (used for the background image and grid size).
        patches: The raw per-patch grid from `compute_local_ghosting`.
        out_path: PNG output path.
    """
    if not patches:
        return
    n_rows = max(p.row for p in patches) + 1
    n_cols = max(p.col for p in patches) + 1
    shift_grid = np.full((n_rows, n_cols), np.nan)
    healed_grid = np.full((n_rows, n_cols), np.nan)
    for patch in patches:
        if patch.valid and patch.healed:
            healed_grid[patch.row, patch.col] = 1.0
        elif patch.valid:
            shift_grid[patch.row, patch.col] = min(patch.shift_px, GHOST_HEATMAP_MAX_PX)

    height, width = dump.composite.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 200, height / 200), dpi=200)
    ax.imshow(cv2.cvtColor(dump.composite, cv2.COLOR_BGR2RGB), extent=(0, width, height, 0))
    im = ax.imshow(
        shift_grid,
        extent=(0, width, height, 0),
        cmap="inferno",
        alpha=0.55,
        vmin=0,
        vmax=GHOST_HEATMAP_MAX_PX,
        interpolation="nearest",
    )
    ax.imshow(
        healed_grid, extent=(0, width, height, 0), cmap="Greens", alpha=0.45, vmin=0, vmax=1, interpolation="nearest"
    )
    ax.set_title("Local ghosting: per-patch shift (px); green = healed, excluded")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="shift (px), capped at %.0f" % GHOST_HEATMAP_MAX_PX)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_absdiff_heatmap(dump: CaptureDump, out_path: Path) -> None:
    """Save a colorized absolute-difference heatmap between composite and reference.

    Args:
        dump: The scored capture dump.
        out_path: PNG output path.
    """
    gray_composite = cv2.cvtColor(dump.composite, cv2.COLOR_BGR2GRAY)
    gray_reference = cv2.cvtColor(dump.reference, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_composite, gray_reference)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out_path), heatmap)


def save_worst_patch_flicker(dump: CaptureDump, ghosting: LocalGhostingMetrics, out_path: Path) -> None:
    """Save the composite and reference crops around the worst patch, side by side.

    Args:
        dump: The scored capture dump.
        ghosting: The `LocalGhostingMetrics` (used to locate the worst patch). Its
            `worst_patch` is already restricted to non-healed patches, so this should
            land on genuine ghosting rather than a healed glare region.
        out_path: PNG output path. Nothing is written if there was no valid worst patch.
    """
    if ghosting.worst_patch is None:
        return
    patch = ghosting.worst_patch
    margin = FLICKER_CROP_MARGIN_PATCHES * ghosting.patch_size_px
    height, width = dump.reference.shape[:2]
    x0 = max(0, patch.x - margin)
    y0 = max(0, patch.y - margin)
    x1 = min(width, patch.x + ghosting.patch_size_px + margin)
    y1 = min(height, patch.y + ghosting.patch_size_px + margin)

    def _labeled_crop(image: np.ndarray, label: str) -> np.ndarray:
        crop = image[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=FLICKER_UPSCALE, fy=FLICKER_UPSCALE, interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return crop

    composite_crop = _labeled_crop(dump.composite, "composite")
    reference_crop = _labeled_crop(dump.reference, "reference")
    separator = np.full((composite_crop.shape[0], 6, 3), (0, 0, 255), dtype=np.uint8)
    cv2.imwrite(str(out_path), cv2.hconcat([composite_crop, separator, reference_crop]))


def _print_metrics(
    geo: GlobalGeometryMetrics,
    ghost: LocalGhostingMetrics,
    edges: EdgeDoublingMetrics,
    sharp: SharpnessMetrics,
    glare: GlareReductionMetrics,
    healing: GlareHealingMetrics,
    bright_detail: BrightDetailMetrics,
) -> None:
    """Print one metrics block (shared by the full-frame and `--quad`-region sections)."""
    print("  Global geometry (SIFT + RANSAC)")
    print(f"    SIFT keypoints         composite={geo.n_keypoints_composite}  reference={geo.n_keypoints_reference}")
    print(f"    Ratio-test matches     {geo.n_ratio_matches}")
    print(f"    RANSAC inliers         {geo.n_inliers} ({100 * geo.inlier_ratio:.1f}%)")
    if geo.identity_deviation_rms_px is not None:
        print(
            f"    Identity deviation     RMS={geo.identity_deviation_rms_px:.2f}px"
            f"  p95={geo.identity_deviation_p95_px:.2f}px"
        )
    else:
        print("    Identity deviation     n/a (too few matches)")

    print(f"\n  Local ghosting ({ghost.patch_size_px}px patch grid, phase correlation)")
    print(f"    Valid patches          {ghost.n_valid_patches}/{ghost.n_patches}")
    print(f"    Healed (excluded)      {ghost.healed_patches}")
    if ghost.median_shift_px is not None:
        print(f"    Shift magnitude        median={ghost.median_shift_px:.2f}px  p95={ghost.p95_shift_px:.2f}px")
        wp = ghost.worst_patch
        assert wp is not None
        print(f"    Worst patch            row={wp.row} col={wp.col} (x={wp.x},y={wp.y})  shift={wp.shift_px:.2f}px")
    else:
        print("    Shift magnitude        n/a (no usable patches)")

    print("\n  Edge doubling")
    print(f"    Canny edge ratio (composite/reference)              {edges.canny_edge_ratio:.3f}")
    print(f"    Gradient magnitude ratio (composite/reference)      {edges.gradient_magnitude_ratio:.3f}")

    print("\n  Sharpness")
    print(f"    Laplacian variance ratio (composite/reference)      {sharp.laplacian_variance_ratio:.3f}")

    print("\n  Glare reduction (near-saturated px >=250, blurred)")
    print(f"    Reference               {100 * glare.reference_saturated_fraction:.3f}%")
    print(f"    Composite               {100 * glare.composite_saturated_fraction:.3f}%")
    reduction_str = "inf" if glare.reduction_factor == float("inf") else f"{glare.reduction_factor:.2f}x"
    print(f"    Reduction factor        {reduction_str}")

    print("\n  Glare healing (darkening = max(0, reference - composite), the benefit metric)")
    print(f"    Darkened fraction (>{DARKENED_PIXEL_THRESHOLD:.0f}/255)   {100 * healing.darkened_fraction:.2f}%")
    print(f"    Mean darkening over darkened patches   {healing.mean_darkening_over_darkened:.2f}/255")
    print(f"    p95 darkening (whole region)           {healing.p95_darkening:.2f}/255")

    def _ratio_str(ratio: float) -> str:
        return "inf" if ratio == float("inf") else f"{ratio:.3f}"

    print("\n  Bright detail preservation (small bright specks, e.g. stars)")
    print(
        f"    Speck count             reference={bright_detail.reference_speck_count}"
        f"  composite={bright_detail.composite_speck_count}"
    )
    print(f"    Retention ratio (composite/reference)              {_ratio_str(bright_detail.retention_ratio)}")
    print(
        f"    Speck count, excl. healed   reference={bright_detail.reference_speck_count_excl_healed}"
        f"  composite={bright_detail.composite_speck_count_excl_healed}"
    )
    print(
        f"    Retention ratio, excl. healed (composite/reference)   "
        f"{_ratio_str(bright_detail.retention_ratio_excl_healed)}"
    )


def print_report(report: StitchQualityReport) -> None:
    """Print a human-readable table of the report to stdout.

    Args:
        report: The `StitchQualityReport` to print.
    """
    print(f"\nStitch quality report: {report.dump_dir}")
    print("=" * 72)

    if report.metadata is not None:
        aligned = report.metadata.get("alignedFrameCount", "?")
        timestamp = report.metadata.get("timestamp", "?")
        print(f"metadata.json         alignedFrameCount={aligned}  timestamp={timestamp}")
    else:
        print("metadata.json          not found")
    print(f"corner frames found     {report.corner_frame_count}/4")

    print("\nFull frame")
    print("-" * 72)
    _print_metrics(
        report.global_geometry,
        report.local_ghosting,
        report.edge_doubling,
        report.sharpness,
        report.glare_reduction,
        report.glare_healing,
        report.bright_detail,
    )

    if report.region is not None:
        region = report.region
        print(f"\nRegion (--quad, {region.n_pixels} px)")
        print("-" * 72)
        _print_metrics(
            region.global_geometry,
            region.local_ghosting,
            region.edge_doubling,
            region.sharpness,
            region.glare_reduction,
            region.glare_healing,
            region.bright_detail,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Score glare-free stitch quality for one capture dump.")
    parser.add_argument("dump_dir", type=Path, help="Directory with reference.jpg, composite.jpg, corner_N.jpg, ...")
    parser.add_argument(
        "--out", type=Path, default=None, help=f"Output dir (default: <dump_dir>/{DEFAULT_OUTPUT_SUBDIR})"
    )
    parser.add_argument(
        "--quad",
        type=str,
        default=None,
        help='Restrict stats to a region, reported alongside full-frame: "x1,y1 x2,y2 x3,y3 x4,y4" '
        "unit coordinates (0-1), clockwise from top-left",
    )
    args = parser.parse_args()

    out_dir = args.out if args.out is not None else args.dump_dir / DEFAULT_OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    quad_unit = parse_quad(args.quad) if args.quad else None

    dump = load_dump(args.dump_dir)
    report, patches = score_dump(dump, quad_unit=quad_unit)

    render_ghost_heatmap(dump, patches, out_dir / "ghost_heatmap.png")
    save_absdiff_heatmap(dump, out_dir / "absdiff_heatmap.png")
    save_worst_patch_flicker(dump, report.local_ghosting, out_dir / "worst_patch_flicker.png")

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)

    print_report(report)
    print(
        f"\nDiagnostics written to {out_dir}/ (metrics.json, ghost_heatmap.png, absdiff_heatmap.png, "
        f"worst_patch_flicker.png)"
    )


if __name__ == "__main__":
    main()
