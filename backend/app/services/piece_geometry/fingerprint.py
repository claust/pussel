"""Piece fingerprint: shape descriptor + spatial color descriptor, and piece-to-piece distances.

Ported from ``network/experiments/exp28_piece_geometry/fingerprint.py`` —
keep algorithm changes in sync.

A fingerprint identifies one PHYSICAL piece from one photo and combines two
signals:

1. SHAPE: the piece's 4 canonical edge polylines (chord-normalized frames
   from `app.services.piece_geometry.edges.canonicalize_edge`). Piece-to-piece
   shape distance is the mean per-edge L2 distance, minimized over the 4
   cyclic rotations of the edge order (robust to unknown photo orientation),
   gated by an exact edge-type-signature match under the winning rotation
   (falling back to the ungated minimum when no rotation's signature matches).
2. SPATIAL COLOR: the piece bbox divided into a 3x3 grid, each cell holding a
   gray-world-normalized a*b* histogram; cells with fewer than
   `SPATIAL_MIN_PIXELS` masked pixels are empty. Spatial distance = mean
   per-cell chi-square over cells non-empty on both sides, with the
   candidate's grid rotated consistently with the shape edge shift k.

A global gray-world a*b* histogram is also carried on the fingerprint for
completeness (exp28 M6 found it a weaker standalone signal than the spatial
descriptor); it is not part of the M7 z-sum score used for scan-lock
thresholding (see `app.services.piece_geometry.scoring`).
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from app.services.piece_geometry.edges import Edge

# Erosion radius (pixels) applied to the contour mask before sampling color,
# to strip the background halo/anti-aliased edge.
ERODE_PX = 5

# a*b*-only histogram (lighting-robustness hypothesis): 16 bins/channel -> 256-d.
AB_BINS = 16
# OpenCV's 8-bit BGR2LAB maps all three channels into [0, 255].
CHANNEL_RANGE = (0, 256)

# Spatial color descriptor: the piece's axis-aligned bbox is divided into a
# SPATIAL_GRID x SPATIAL_GRID grid; each cell gets a gray-world a*b*
# histogram with SPATIAL_AB_BINS bins/channel; cells with fewer than
# SPATIAL_MIN_PIXELS masked pixels are marked empty.
SPATIAL_GRID = 3
SPATIAL_AB_BINS = 8
SPATIAL_MIN_PIXELS = 50

EdgeTypes = Tuple[str, str, str, str]


def _spatial_rotation_permutations() -> np.ndarray:
    """Cell-index permutations of the spatial grid under CCW rotations.

    Row k gives, for each query cell index (row-major 3x3), the CANDIDATE
    cell index that occupies the same physical location once the candidate
    piece is rotated CCW by k*90 degrees - matching the shape edge shift k
    convention (query edge i pairs with candidate edge (i+k) mod 4).

    Returns:
        (4, SPATIAL_GRID**2) integer permutation array.
    """
    idx = np.arange(SPATIAL_GRID**2).reshape(SPATIAL_GRID, SPATIAL_GRID)
    return np.stack([np.rot90(idx, k).flatten() for k in range(4)], axis=0)


SPATIAL_ROT_PERMS = _spatial_rotation_permutations()


@dataclass(frozen=True)
class PieceFingerprint:
    """One physical piece's fingerprint from one photo.

    Attributes:
        edge_types: Edge type per edge, in contour traversal order.
        edges_canonical: (4, 100, 2) canonical edge polylines, contour order.
        spatial_hists: (9, SPATIAL_AB_BINS**2) per-grid-cell gray-world a*b*
            histograms (row-major 3x3 over the piece bbox), each
            L1-normalized or zero.
        spatial_nonempty: (9,) boolean flags, True where the cell has at
            least `SPATIAL_MIN_PIXELS` masked pixels.
        color_hist_ab_gw: 256-d L1-normalized global a*b* histogram after
            gray-world normalization (kept on the record for completeness;
            not used by the M7 z-sum score).
    """

    edge_types: EdgeTypes
    edges_canonical: np.ndarray
    spatial_hists: np.ndarray
    spatial_nonempty: np.ndarray
    color_hist_ab_gw: np.ndarray


def build_piece_mask(image_shape: Tuple[int, ...], contour: np.ndarray, erode_px: int = ERODE_PX) -> np.ndarray:
    """Rasterize a full-image-coordinate contour into an eroded binary mask.

    Args:
        image_shape: The source photo's (H, W[, C]) shape.
        contour: Nx2 contour points in the photo's own pixel coordinates.
        erode_px: Erosion radius in pixels, to strip the background halo.

    Returns:
        Binary mask (H, W) with values in {0, 255}.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32)], (255,))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    return cv2.erode(mask, kernel, iterations=1)


def _gray_world_lab_pixels(bgr_pixels: np.ndarray) -> np.ndarray:
    """Gray-world-normalize BGR pixels and convert them to L*a*b*.

    Scales each BGR channel so the pixel-set mean is neutral, clips to
    [0, 255], and converts the result to OpenCV 8-bit L*a*b*.

    Args:
        bgr_pixels: (N, 3) float BGR pixel values.

    Returns:
        (N, 3) L*a*b* pixel values after gray-world normalization.
    """
    channel_means = bgr_pixels.mean(axis=0)
    gray = float(channel_means.mean())
    scaled = bgr_pixels * (gray / np.maximum(channel_means, 1e-9))
    scaled_u8 = np.clip(scaled, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(scaled_u8, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)


def _ab_histogram(lab_pixels: np.ndarray) -> np.ndarray:
    """L1-normalized a*b*-only histogram (16x16) of L*a*b* pixels.

    Args:
        lab_pixels: (N, 3) L*a*b* pixel values in OpenCV 8-bit ranges.

    Returns:
        256-d L1-normalized histogram.
    """
    hist_ab, _, _ = np.histogram2d(lab_pixels[:, 1], lab_pixels[:, 2], bins=AB_BINS, range=(CHANNEL_RANGE,) * 2)
    hist_ab = hist_ab.flatten()
    return hist_ab / max(hist_ab.sum(), 1e-10)


def compute_global_ab_gw_histogram(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute the global gray-world a*b* histogram over masked pixels.

    Args:
        image_bgr: The full photo in OpenCV BGR channel order.
        mask: Binary mask (H, W), values in {0, 255}; nonzero pixels are sampled.

    Returns:
        256-d L1-normalized histogram (all-zero when the mask is empty).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.zeros(AB_BINS**2, dtype=np.float64)
    bgr_pixels = image_bgr[ys, xs].astype(np.float64)
    return _ab_histogram(_gray_world_lab_pixels(bgr_pixels))


def compute_spatial_color(
    image_bgr: np.ndarray, mask: np.ndarray, contour: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 3x3 spatial gray-world a*b* color descriptor over masked pixels.

    All masked pixels are gray-world normalized TOGETHER (one piece-level
    illuminant correction), then binned per grid cell into an a*b* histogram.
    Cells with fewer than `SPATIAL_MIN_PIXELS` masked pixels are marked empty.

    Args:
        image_bgr: The full photo in OpenCV BGR channel order.
        mask: Binary mask (H, W), values in {0, 255}; nonzero pixels are sampled.
        contour: Nx2 full contour in the photo's own pixel coordinates
            (defines the bbox the grid spans).

    Returns:
        Tuple of (per-cell histograms (9, SPATIAL_AB_BINS**2), row-major cell
        order, each L1-normalized or all-zero when empty; (9,) boolean
        non-empty flags).
    """
    n_cells = SPATIAL_GRID**2
    hists = np.zeros((n_cells, SPATIAL_AB_BINS**2), dtype=np.float64)
    nonempty = np.zeros(n_cells, dtype=bool)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return hists, nonempty

    lab_pixels = _gray_world_lab_pixels(image_bgr[ys, xs].astype(np.float64))

    x1, y1 = contour[:, 0].min(), contour[:, 1].min()
    x2, y2 = contour[:, 0].max(), contour[:, 1].max()
    cell_cols = np.clip(((xs - x1) * SPATIAL_GRID / max(x2 - x1, 1e-9)).astype(np.int64), 0, SPATIAL_GRID - 1)
    cell_rows = np.clip(((ys - y1) * SPATIAL_GRID / max(y2 - y1, 1e-9)).astype(np.int64), 0, SPATIAL_GRID - 1)
    cell_idx = cell_rows * SPATIAL_GRID + cell_cols

    for cell in range(n_cells):
        in_cell = cell_idx == cell
        if int(in_cell.sum()) < SPATIAL_MIN_PIXELS:
            continue
        cell_pixels = lab_pixels[in_cell]
        hist, _, _ = np.histogram2d(
            cell_pixels[:, 1], cell_pixels[:, 2], bins=SPATIAL_AB_BINS, range=(CHANNEL_RANGE,) * 2
        )
        hist = hist.flatten()
        hists[cell] = hist / max(hist.sum(), 1e-10)
        nonempty[cell] = True

    return hists, nonempty


def build_fingerprint(edges: List[Edge], image_bgr: np.ndarray, contour: np.ndarray) -> PieceFingerprint:
    """Compute a piece's shape + spatial-color fingerprint.

    Args:
        edges: The piece's 4 classified, canonicalized edges (contour order).
        image_bgr: The source photo in OpenCV BGR channel order.
        contour: Nx2 full contour in the photo's own pixel coordinates.

    Returns:
        The computed `PieceFingerprint`.
    """
    edge_types: EdgeTypes = tuple(e.edge_type for e in edges)  # type: ignore[assignment]
    edges_canonical = np.stack([e.canonical_polyline for e in edges], axis=0)
    mask = build_piece_mask(image_bgr.shape, contour)
    spatial_hists, spatial_nonempty = compute_spatial_color(image_bgr, mask, contour)
    color_hist_ab_gw = compute_global_ab_gw_histogram(image_bgr, mask)
    return PieceFingerprint(
        edge_types=edge_types,
        edges_canonical=edges_canonical,
        spatial_hists=spatial_hists,
        spatial_nonempty=spatial_nonempty,
        color_hist_ab_gw=color_hist_ab_gw,
    )


def shape_pair_distance(
    query_edges: np.ndarray,
    query_types: EdgeTypes,
    candidate_edges: np.ndarray,
    candidate_types: EdgeTypes,
) -> Tuple[float, int]:
    """Rotation-invariant same-piece shape distance between two pieces, with the winning shift.

    For each candidate cyclic shift k of the edge order, compares query edge
    i (no mate-flip, same traversal) to candidate edge (i+k) mod 4 via mean
    pointwise L2, and requires the shifted type signature to match the
    query's exactly to be an eligible k. Returns the minimum eligible
    per-edge-mean distance (falling back to the minimum over ALL k when no
    shift's type signature matches) together with its winning k, so a
    caller can keep a rotation-sensitive descriptor (spatial color)
    consistent with the shape alignment.

    Args:
        query_edges: (4, 100, 2) canonical edges, contour order.
        query_types: 4-tuple of edge types for the query.
        candidate_edges: (4, 100, 2) canonical edges, contour order.
        candidate_types: 4-tuple of edge types for the candidate.

    Returns:
        Tuple of (scalar shape distance, winning cyclic shift k in [0, 4)).
    """
    l2_by_k: List[float] = []
    gate_by_k: List[bool] = []
    for k in range(4):
        shift = [(i + k) % 4 for i in range(4)]
        shifted_types = tuple(candidate_types[j] for j in shift)
        gate_by_k.append(shifted_types == query_types)
        per_edge = [
            float(np.mean(np.linalg.norm(query_edges[i] - candidate_edges[shift[i]], axis=1))) for i in range(4)
        ]
        l2_by_k.append(float(np.mean(per_edge)))

    gated = [(dist, k) for k, (dist, ok) in enumerate(zip(l2_by_k, gate_by_k)) if ok]
    if gated:
        return min(gated, key=lambda pair: pair[0])
    best_k = int(np.argmin(l2_by_k))
    return l2_by_k[best_k], best_k


def spatial_pair_distance(query: PieceFingerprint, candidate: PieceFingerprint, k: int) -> float:
    """Spatial color chi-square distance between two pieces, rotation-consistent with shift k.

    Args:
        query: The query piece's fingerprint.
        candidate: The candidate piece's fingerprint.
        k: The cyclic edge shift chosen by `shape_pair_distance` for this pair.

    Returns:
        The mean per-cell chi-square distance over cells non-empty on both
        sides (after rotating the candidate's grid by k), or 1.0 (max
        chi-square distance) when no cell is non-empty on both sides.
    """
    perm = SPATIAL_ROT_PERMS[k % 4]
    candidate_hists = candidate.spatial_hists[perm]
    candidate_nonempty = candidate.spatial_nonempty[perm]
    valid = query.spatial_nonempty & candidate_nonempty
    if not valid.any():
        return 1.0
    p = query.spatial_hists[valid]
    q = candidate_hists[valid]
    per_cell = 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-10), axis=1)
    return float(per_cell.mean())


def chi_square_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Chi-square distance between two L1-normalized histograms.

    Args:
        p: First histogram.
        q: Second histogram, same shape as `p`.

    Returns:
        0.5 * sum((p - q)^2 / (p + q + eps)).
    """
    return float(0.5 * np.sum((p - q) ** 2 / (p + q + 1e-10)))
