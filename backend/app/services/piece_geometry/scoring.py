"""Combined shape + spatial-color z-score for scan-lock accept/new/uncertain decisions.

Ported from ``network/experiments/exp28_piece_geometry/collision_study.py``
(M7) — keep algorithm changes in sync.

``z = z_shape + z_spatial``, where each raw distance is z-normalized by the
enroll gallery's own IMPOSTOR (distinct-piece) pairwise distance statistics.
Per-gallery normalization is what makes one threshold transfer across
galleries: each gallery's z is expressed in units of its own impostor
spread. When a puzzle's piece store is too small to estimate its own
impostor statistics reliably, `FALLBACK_STATS` (the M7-measured constants,
which were stable across all four north_star backgrounds) is used instead.

The `t_accept` / `t_new` scan-lock thresholds themselves live in
`app.config.Settings` (`PIECE_GEOMETRY_T_ACCEPT` / `PIECE_GEOMETRY_T_NEW`)
so they can be tuned via environment/config without a code change. See
`Settings` for the authoritative current defaults and the rationale: M7
froze the strictest accept point (-4.78), but the shipped default relaxes
`t_accept` to M7's FMR=1% ROC operating point so hands-free scan enrollment
does not send most genuine re-scans to the gray zone; `t_new` stays at the
M7-frozen -0.80.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Below this many enrolled pieces, a puzzle's own impostor statistics are too
# noisy to trust; fall back to the cross-gallery M7 constants instead.
MIN_GALLERY_FOR_STATS = 12

# M7-measured impostor distance statistics (mean/std), which were nearly
# identical across all four north_star enroll galleries (gray_fabric,
# red_carpet, cardboard, wood) — see exp28's collision_study.py output.
FALLBACK_SHAPE_MEAN = 0.079
FALLBACK_SHAPE_STD = 0.030
FALLBACK_SPATIAL_MEAN = 0.56
FALLBACK_SPATIAL_STD = 0.14


@dataclass(frozen=True)
class GalleryStats:
    """Impostor-distance normalization statistics for one enroll gallery.

    Attributes:
        shape_mean: Mean pairwise distinct-identity shape distance.
        shape_std: Std of the same.
        spatial_mean: Mean pairwise distinct-identity spatial color distance.
        spatial_std: Std of the same.
    """

    shape_mean: float
    shape_std: float
    spatial_mean: float
    spatial_std: float


FALLBACK_STATS = GalleryStats(
    shape_mean=FALLBACK_SHAPE_MEAN,
    shape_std=FALLBACK_SHAPE_STD,
    spatial_mean=FALLBACK_SPATIAL_MEAN,
    spatial_std=FALLBACK_SPATIAL_STD,
)


def combined_z(d_shape: float, d_spatial: float, stats: GalleryStats) -> float:
    """Combine shape + spatial-color distances into one z-normalized score.

    Args:
        d_shape: Raw rotation-invariant shape distance to a gallery piece.
        d_spatial: Raw spatial color chi-square distance to the same piece.
        stats: The enroll gallery's impostor normalization statistics.

    Returns:
        z = z_shape + z_spatial. Lower is more similar.
    """
    z_shape = (d_shape - stats.shape_mean) / max(stats.shape_std, 1e-9)
    z_spatial = (d_spatial - stats.spatial_mean) / max(stats.spatial_std, 1e-9)
    return z_shape + z_spatial


def gallery_impostor_stats(pairwise_shape: Sequence[float], pairwise_spatial: Sequence[float]) -> GalleryStats:
    """Compute impostor mean/std of shape and spatial distances within an enroll gallery.

    Args:
        pairwise_shape: All distinct-piece pairwise shape distances in the gallery.
        pairwise_spatial: The matching pairwise spatial-color distances (same pairing/order).

    Returns:
        The gallery's `GalleryStats`.
    """
    shape_arr = np.array(pairwise_shape, dtype=np.float64)
    spatial_arr = np.array(pairwise_spatial, dtype=np.float64)
    return GalleryStats(
        shape_mean=float(shape_arr.mean()),
        shape_std=float(shape_arr.std()),
        spatial_mean=float(spatial_arr.mean()),
        spatial_std=float(spatial_arr.std()),
    )
