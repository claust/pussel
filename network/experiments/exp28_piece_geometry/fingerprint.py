#!/usr/bin/env python3
"""M6: piece fingerprint (shape + color) library and CLI to build per-background galleries.

A fingerprint identifies one PHYSICAL piece from one photo (puzzle_id, row,
col, background) and combines two signals:

1. SHAPE: the piece's 4 canonical edge polylines (M4's chord-normalized
   frames, reused verbatim from `edge_match.canonicalize_edge` /
   `edge_match.dist_l2` - NOT the mate-flip transform, since this is
   same-piece re-identification, not edge-to-edge complementarity matching).
   Piece-to-piece shape distance is the mean per-edge L2 distance, minimized
   over the 4 cyclic rotations of the edge order (to be robust to the rare
   case where the upright-photo auto-orientation differs by 90 degrees
   between two photos of the same piece), gated by an exact edge-type
   signature match under the winning rotation (with an ungated fallback when
   no rotation's signature matches).
2. COLOR: a chi-square histogram distance over the piece's masked, eroded
   face pixels in three global variants - L*a*b* joint (lighting-sensitive),
   a*b* only (lighting-robust hypothesis), and a*b* after gray-world
   normalization of the masked pixels (illuminant-robust hypothesis: each
   BGR channel is scaled so the masked mean is neutral before conversion) -
   plus a SPATIAL descriptor: the piece bbox is divided into a 3x3 grid and
   each cell gets its own gray-world a*b* histogram, so pieces sharing a
   palette but differing in color LAYOUT stay separable. Spatial distance =
   mean per-cell chi-square over cells non-empty on both sides, with the
   candidate's grid rotated consistently with the shape edge shift k.

`build_gallery` loads all clean (non `corner_disagreement`) M3 piece records
for one background, loads each piece's photo + M1 full contour to compute
color histograms, and returns one `PieceFingerprint` per piece. The CLI
persists color histograms + piece references to
`outputs/fingerprints/{background}.json`; shape data is cheap to recompute
from the M3 piece records and is intentionally NOT duplicated into that
file. `load_gallery` reverses this: it reads a fingerprint JSON for the
color data and re-derives the canonical edge arrays from the M3 records.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/fingerprint.py
    uv run python experiments/exp28_piece_geometry/fingerprint.py --background wood
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from edge_match import canonicalize_edge

DEFAULT_DATASET_ROOT = Path("/Users/claus/Repos/pussel/network/datasets/north_star/v1")

BACKGROUNDS = ("red_carpet", "gray_fabric", "cardboard", "wood")
DIRECTIONS = ("N", "E", "S", "W")

# Erosion radius (pixels) applied to the M1 full contour mask before
# sampling color, to strip the background halo/anti-aliased edge.
ERODE_PX = 5

# L*a*b* joint histogram: 8 bins/channel -> 512-d.
LAB_BINS = 8
# a*b*-only histogram (lighting-robustness hypothesis): 16 bins/channel -> 256-d.
AB_BINS = 16
# OpenCV's 8-bit BGR2LAB maps all three channels into [0, 255].
CHANNEL_RANGE = (0, 256)

# Spatial color descriptor: the piece's axis-aligned bbox is divided into a
# SPATIAL_GRID x SPATIAL_GRID grid (upright photo frame); each cell gets a
# gray-world a*b* histogram with SPATIAL_AB_BINS bins/channel; cells with
# fewer than SPATIAL_MIN_PIXELS masked pixels are marked empty.
SPATIAL_GRID = 3
SPATIAL_AB_BINS = 8
SPATIAL_MIN_PIXELS = 50


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
    """One physical piece's fingerprint from one background photo.

    Attributes:
        puzzle_id: Puzzle identifier.
        row: Piece grid row.
        col: Piece grid column.
        background: Background surface name of the source photo.
        piece_file: Path to the piece photo, relative to the dataset root.
        edge_types: Edge type per direction, in N/E/S/W order.
        edges_canonical: (4, 100, 2) canonical edge polylines, N/E/S/W order,
            each via `edge_match.canonicalize_edge` (not mate-flipped).
        color_hist_lab: 512-d L1-normalized joint L*a*b* histogram (8 bins/channel).
        color_hist_ab: 256-d L1-normalized a*b*-only histogram (16 bins/channel).
        color_hist_ab_gw: 256-d L1-normalized a*b*-only histogram after
            gray-world normalization of the masked pixels.
        spatial_hists: (9, 64) per-grid-cell gray-world a*b* histograms
            (row-major 3x3 over the piece bbox), each L1-normalized or zero.
        spatial_nonempty: (9,) boolean flags, True where the cell has at
            least `SPATIAL_MIN_PIXELS` masked pixels.
    """

    puzzle_id: str
    row: int
    col: int
    background: str
    piece_file: str
    edge_types: Tuple[str, str, str, str]
    edges_canonical: np.ndarray
    color_hist_lab: np.ndarray
    color_hist_ab: np.ndarray
    color_hist_ab_gw: np.ndarray
    spatial_hists: np.ndarray
    spatial_nonempty: np.ndarray

    @property
    def identity(self) -> Tuple[str, int, int]:
        """The physical-piece identity key (puzzle_id, row, col), shared across backgrounds."""
        return (self.puzzle_id, self.row, self.col)


def canonical_edges_from_record(record: Dict) -> np.ndarray:
    """Compute the (4, 100, 2) canonical edge array (N/E/S/W order) from an M3 piece record.

    Args:
        record: A loaded `outputs/piece_records/{puzzle}/{stem}.json` dict.

    Returns:
        (4, 100, 2) float array of canonical (chord-normalized, unflipped) edge polylines.
    """
    arrays = []
    for direction in DIRECTIONS:
        polyline = np.array(record["edges"][direction]["polyline"], dtype=np.float64)
        arrays.append(canonicalize_edge(polyline))
    return np.stack(arrays, axis=0)


def type_signature_from_record(record: Dict) -> Tuple[str, str, str, str]:
    """Extract the (N, E, S, W) edge-type signature from an M3 piece record.

    Args:
        record: A loaded piece-record dict.

    Returns:
        4-tuple of edge types ("tab"/"blank"/"flat"), N/E/S/W order.
    """
    return tuple(record["edges"][direction]["type"] for direction in DIRECTIONS)  # type: ignore[return-value]


def load_full_contour(contours_dir: Path, puzzle_id: str, piece_stem: str) -> Optional[np.ndarray]:
    """Load a piece's clean rembg contour from the M1 outputs, if it exists.

    Args:
        contours_dir: The `outputs/contours` directory.
        puzzle_id: The piece's puzzle id.
        piece_stem: The piece's filename stem.

    Returns:
        Nx2 contour array in original image coordinates, or None when the
        contour JSON is missing, has no rembg contour, or is not clean.
    """
    contour_path = contours_dir / puzzle_id / f"{piece_stem}.json"
    if not contour_path.exists():
        return None
    with open(contour_path, encoding="utf-8") as handle:
        data = json.load(handle)
    rembg = data["methods"].get("rembg")
    if not rembg or not rembg.get("contour") or not rembg["quality"]["is_clean"]:
        return None
    return np.array(rembg["contour"], dtype=np.float64)


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
    cv2.fillPoly(mask, [contour.astype(np.int32)], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    return cv2.erode(mask, kernel, iterations=1)


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


def _gray_world_lab_pixels(bgr_pixels: np.ndarray) -> np.ndarray:
    """Gray-world-normalize BGR pixels and convert them to L*a*b*.

    Scales each BGR channel so the pixel-set mean is neutral (all channel
    means equal to their overall mean), clips to [0, 255], and converts the
    result to OpenCV 8-bit L*a*b*.

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


def compute_color_histograms(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the three L1-normalized color histograms over masked pixels.

    Variants: (a) joint L*a*b* 8x8x8, (b) a*b*-only 16x16, (c) a*b*-only
    16x16 after gray-world normalization of the masked pixels
    (illuminant-robustness hypothesis).

    Args:
        image_bgr: The full photo in OpenCV BGR channel order.
        mask: Binary mask (H, W), values in {0, 255}; nonzero pixels are sampled.

    Returns:
        Tuple of (512-d L*a*b* histogram, 256-d a*b* histogram, 256-d
        gray-world a*b* histogram), each L1-normalized.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        zeros_ab = np.zeros(AB_BINS**2, dtype=np.float64)
        return np.zeros(LAB_BINS**3, dtype=np.float64), zeros_ab, zeros_ab.copy()

    pixels = lab[ys, xs].astype(np.float64)

    hist_lab, _ = np.histogramdd(pixels, bins=(LAB_BINS, LAB_BINS, LAB_BINS), range=(CHANNEL_RANGE,) * 3)
    hist_lab = hist_lab.flatten()
    hist_lab = hist_lab / max(hist_lab.sum(), 1e-10)

    hist_ab = _ab_histogram(pixels)

    bgr_pixels = image_bgr[ys, xs].astype(np.float64)
    hist_ab_gw = _ab_histogram(_gray_world_lab_pixels(bgr_pixels))

    return hist_lab, hist_ab, hist_ab_gw


def compute_spatial_color(
    image_bgr: np.ndarray, mask: np.ndarray, contour: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 3x3 spatial gray-world a*b* color descriptor over masked pixels.

    The piece's axis-aligned bbox (from the full contour, upright photo
    frame) is divided into a `SPATIAL_GRID` x `SPATIAL_GRID` grid. All
    masked pixels are gray-world normalized TOGETHER (one piece-level
    illuminant correction, same transform as the global ab_gw histogram),
    then binned per grid cell into an a*b* histogram with
    `SPATIAL_AB_BINS` bins/channel. Cells with fewer than
    `SPATIAL_MIN_PIXELS` masked pixels are marked empty.

    Args:
        image_bgr: The full photo in OpenCV BGR channel order.
        mask: Binary mask (H, W), values in {0, 255}; nonzero pixels are sampled.
        contour: Nx2 full contour in the photo's own pixel coordinates
            (defines the bbox the grid spans).

    Returns:
        Tuple of (per-cell histograms (9, SPATIAL_AB_BINS**2), row-major
        cell order, each L1-normalized or all-zero when empty; (9,) boolean
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


def spatial_color_distance_matrix(
    query_hists: np.ndarray,
    query_nonempty: np.ndarray,
    gallery_hists: np.ndarray,
    gallery_nonempty: np.ndarray,
    k_per_candidate: np.ndarray,
) -> np.ndarray:
    """Spatial color distance from one query piece to a gallery, rotation-consistent.

    Each candidate's 3x3 grid is rotated by its shape-chosen edge shift k
    (via `SPATIAL_ROT_PERMS`), then the distance is the mean per-cell
    chi-square over cells non-empty on BOTH sides. Candidates sharing no
    non-empty cell with the query get the maximum chi-square distance (1.0).

    Args:
        query_hists: (9, D) query cell histograms.
        query_nonempty: (9,) query non-empty flags.
        gallery_hists: (N, 9, D) gallery cell histograms.
        gallery_nonempty: (N, 9) gallery non-empty flags.
        k_per_candidate: (N,) edge shift k chosen by the shape distance for
            each candidate (all zeros for the fixed-orientation protocol).

    Returns:
        (N,) spatial color distances.
    """
    n = gallery_hists.shape[0]
    out = np.full(n, 1.0, dtype=np.float64)
    for k in np.unique(k_per_candidate):
        group = np.where(k_per_candidate == k)[0]
        perm = SPATIAL_ROT_PERMS[int(k) % 4]
        cand_hists = gallery_hists[group][:, perm, :]  # (G, 9, D)
        cand_nonempty = gallery_nonempty[group][:, perm]  # (G, 9)
        q = query_hists[None, :, :]
        per_cell = 0.5 * np.sum((q - cand_hists) ** 2 / (q + cand_hists + 1e-10), axis=2)  # (G, 9)
        valid = query_nonempty[None, :] & cand_nonempty  # (G, 9)
        n_valid = valid.sum(axis=1)
        sums = np.where(valid, per_cell, 0.0).sum(axis=1)
        has_valid = n_valid > 0
        out[group[has_valid]] = sums[has_valid] / n_valid[has_valid]
    return out


def chi_square_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Chi-square distance between two L1-normalized histograms.

    Args:
        p: First histogram.
        q: Second histogram, same shape as `p`.

    Returns:
        0.5 * sum((p - q)^2 / (p + q + eps)).
    """
    return float(0.5 * np.sum((p - q) ** 2 / (p + q + 1e-10)))


def chi_square_distance_matrix(query_hist: np.ndarray, gallery_hists: np.ndarray) -> np.ndarray:
    """Vectorized chi-square distance from one query histogram to a gallery of histograms.

    Args:
        query_hist: (D,) histogram.
        gallery_hists: (N, D) stacked histograms.

    Returns:
        (N,) chi-square distances.
    """
    p = query_hist[None, :]
    q = gallery_hists
    return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-10), axis=1)


def shape_pair_distance(
    query_edges: np.ndarray,
    query_types: Tuple[str, str, str, str],
    candidate_edges: np.ndarray,
    candidate_types: Tuple[str, str, str, str],
    rotation_invariant: bool = True,
) -> float:
    """Scalar same-piece shape distance between two pieces' canonical edge sets.

    For each candidate cyclic shift k of the edge order (k=0 only when
    `rotation_invariant` is False), compares query edge i (same traversal,
    no mate-flip) to candidate edge (i+k) mod 4 via `edge_match`-style mean
    pointwise L2, and requires the shifted type signature to match the
    query's exactly to be an eligible k. Returns the minimum eligible
    per-edge-mean distance, falling back to the minimum over ALL k when no
    shift's type signature matches.

    Args:
        query_edges: (4, 100, 2) canonical edges, N/E/S/W order.
        query_types: (N, E, S, W) edge types for the query.
        candidate_edges: (4, 100, 2) canonical edges, N/E/S/W order.
        candidate_types: (N, E, S, W) edge types for the candidate.
        rotation_invariant: When False, only k=0 is considered.

    Returns:
        The scalar shape distance.
    """
    ks = range(4) if rotation_invariant else (0,)
    l2_by_k = []
    gate_by_k = []
    for k in ks:
        shift = [(i + k) % 4 for i in range(4)]
        shifted_types = tuple(candidate_types[j] for j in shift)
        gate_by_k.append(shifted_types == query_types)
        per_edge = [
            float(np.mean(np.linalg.norm(query_edges[i] - candidate_edges[shift[i]], axis=1))) for i in range(4)
        ]
        l2_by_k.append(float(np.mean(per_edge)))
    if any(gate_by_k):
        return min(dist for dist, ok in zip(l2_by_k, gate_by_k) if ok)
    return min(l2_by_k)


def shape_distance_matrix(
    query_edges: np.ndarray,
    query_types: Tuple[str, str, str, str],
    gallery_edges: np.ndarray,
    gallery_types: np.ndarray,
    rotation_invariant: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized same-piece shape distance from one query piece to a gallery of pieces.

    Same semantics as `shape_pair_distance`, computed for the whole gallery
    at once via numpy broadcasting. Also returns the winning cyclic shift k
    per candidate, so downstream rotation-sensitive descriptors (the spatial
    color grid) can stay consistent with the shape alignment.

    Args:
        query_edges: (4, 100, 2) canonical edges, N/E/S/W order.
        query_types: (N, E, S, W) edge types for the query.
        gallery_edges: (N, 4, 100, 2) canonical edges for N gallery pieces.
        gallery_types: (N, 4) string array of gallery edge types, N/E/S/W order.
        rotation_invariant: When False, only k=0 is considered.

    Returns:
        Tuple of ((N,) shape distances, (N,) winning shift k per candidate).
    """
    ks = range(4) if rotation_invariant else (0,)
    n = gallery_edges.shape[0]
    query_types_arr = np.array(query_types)
    l2_stack = np.empty((len(ks), n), dtype=np.float64)
    gate_stack = np.empty((len(ks), n), dtype=bool)
    for row, k in enumerate(ks):
        shift = [(i + k) % 4 for i in range(4)]
        shifted_edges = gallery_edges[:, shift, :, :]  # (N, 4, 100, 2)
        diffs = query_edges[None, :, :, :] - shifted_edges
        per_point = np.linalg.norm(diffs, axis=3)  # (N, 4, 100)
        per_edge_mean = per_point.mean(axis=2)  # (N, 4)
        l2_stack[row] = per_edge_mean.mean(axis=1)  # (N,)
        shifted_types = gallery_types[:, shift]  # (N, 4)
        gate_stack[row] = np.all(shifted_types == query_types_arr[None, :], axis=1)

    any_gate = gate_stack.any(axis=0)
    masked = np.where(gate_stack, l2_stack, np.inf)
    ks_arr = np.array(list(ks))
    best_k = np.where(any_gate, ks_arr[np.argmin(masked, axis=0)], ks_arr[np.argmin(l2_stack, axis=0)])
    best_gated = masked.min(axis=0)
    best_all = l2_stack.min(axis=0)
    return np.where(any_gate, best_gated, best_all), best_k


def build_gallery(records_dir: Path, contours_dir: Path, dataset_root: Path, background: str) -> List[PieceFingerprint]:
    """Build fingerprints for every clean, non-disagreement piece record of one background.

    "Clean record" = an M3 piece-record JSON exists for this piece x
    background AND `corner_disagreement` is False (M4 found this gate worth
    ~8 top-1 points on shape alone; M6 applies the same filter to both the
    shape and color signal).

    Args:
        records_dir: The `outputs/piece_records` directory.
        contours_dir: The `outputs/contours` directory (M1 full contours,
            used to rasterize the color-sampling mask).
        dataset_root: The north_star v1 dataset root (for photo pixels).
        background: Background name to filter to.

    Returns:
        One `PieceFingerprint` per eligible piece, in file order.
    """
    fingerprints: List[PieceFingerprint] = []
    for path in sorted(records_dir.glob("*/*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        if record["background"] != background or record["corner_disagreement"]:
            continue

        piece_stem = Path(record["piece_file"]).stem
        contour = load_full_contour(contours_dir, record["puzzle_id"], piece_stem)
        if contour is None:
            continue

        image_path = dataset_root / record["piece_file"]
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        mask = build_piece_mask(image.shape, contour)
        hist_lab, hist_ab, hist_ab_gw = compute_color_histograms(image, mask)
        spatial_hists, spatial_nonempty = compute_spatial_color(image, mask, contour)

        fingerprints.append(
            PieceFingerprint(
                puzzle_id=record["puzzle_id"],
                row=record["row"],
                col=record["col"],
                background=background,
                piece_file=record["piece_file"],
                edge_types=type_signature_from_record(record),
                edges_canonical=canonical_edges_from_record(record),
                color_hist_lab=hist_lab,
                color_hist_ab=hist_ab,
                color_hist_ab_gw=hist_ab_gw,
                spatial_hists=spatial_hists,
                spatial_nonempty=spatial_nonempty,
            )
        )
    return fingerprints


def save_fingerprints(fingerprints: List[PieceFingerprint], path: Path) -> None:
    """Persist a gallery's color histograms + piece references (not shape data) to JSON.

    Args:
        fingerprints: The gallery to persist.
        path: Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "background": fingerprints[0].background if fingerprints else None,
        "n_pieces": len(fingerprints),
        "pieces": [
            {
                "puzzle_id": fp.puzzle_id,
                "row": fp.row,
                "col": fp.col,
                "piece_file": fp.piece_file,
                "edge_types": list(fp.edge_types),
                "color_hist_lab": fp.color_hist_lab.tolist(),
                "color_hist_ab": fp.color_hist_ab.tolist(),
                "color_hist_ab_gw": fp.color_hist_ab_gw.tolist(),
                "spatial_hists": fp.spatial_hists.tolist(),
                "spatial_nonempty": fp.spatial_nonempty.tolist(),
            }
            for fp in fingerprints
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)


def load_gallery(fingerprint_path: Path, records_dir: Path) -> List[PieceFingerprint]:
    """Reload a persisted gallery, re-deriving shape data from the M3 piece records.

    Args:
        fingerprint_path: Path to an `outputs/fingerprints/{background}.json` file.
        records_dir: The `outputs/piece_records` directory.

    Returns:
        One `PieceFingerprint` per persisted piece.
    """
    with open(fingerprint_path, encoding="utf-8") as handle:
        data = json.load(handle)
    background = data["background"]

    fingerprints: List[PieceFingerprint] = []
    for piece in data["pieces"]:
        piece_stem = Path(piece["piece_file"]).stem
        record_path = records_dir / piece["puzzle_id"] / f"{piece_stem}.json"
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
        fingerprints.append(
            PieceFingerprint(
                puzzle_id=piece["puzzle_id"],
                row=piece["row"],
                col=piece["col"],
                background=background,
                piece_file=piece["piece_file"],
                edge_types=tuple(piece["edge_types"]),  # type: ignore[arg-type]
                edges_canonical=canonical_edges_from_record(record),
                color_hist_lab=np.array(piece["color_hist_lab"], dtype=np.float64),
                color_hist_ab=np.array(piece["color_hist_ab"], dtype=np.float64),
                color_hist_ab_gw=np.array(piece["color_hist_ab_gw"], dtype=np.float64),
                spatial_hists=np.array(piece["spatial_hists"], dtype=np.float64),
                spatial_nonempty=np.array(piece["spatial_nonempty"], dtype=bool),
            )
        )
    return fingerprints


def main() -> None:
    """CLI entry point: build and persist fingerprints for one or all backgrounds."""
    parser = argparse.ArgumentParser(description="Build piece fingerprints (shape + color) per background.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--records-dir", type=Path, default=Path(__file__).parent / "outputs" / "piece_records")
    parser.add_argument("--contours-dir", type=Path, default=Path(__file__).parent / "outputs" / "contours")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--background", type=str, default=None, choices=BACKGROUNDS)
    args = parser.parse_args()

    backgrounds = [args.background] if args.background else list(BACKGROUNDS)
    fingerprints_dir = args.output_dir / "fingerprints"

    n_total_pieces = len({(r.puzzle_id, r.row, r.col) for r in _distinct_pieces(args.records_dir)})
    for background in backgrounds:
        fingerprints = build_gallery(args.records_dir, args.contours_dir, args.dataset_root, background)
        out_path = fingerprints_dir / f"{background}.json"
        save_fingerprints(fingerprints, out_path)
        print(f"{background:12s}: {len(fingerprints):3d}/{n_total_pieces} clean pieces fingerprinted -> {out_path}")


def _distinct_pieces(records_dir: Path) -> List["_Identity"]:
    """Enumerate distinct (puzzle_id, row, col) physical pieces across all piece records.

    Args:
        records_dir: The `outputs/piece_records` directory.

    Returns:
        List of lightweight identity holders (one per distinct physical piece).
    """
    seen: Dict[Tuple[str, int, int], "_Identity"] = {}
    for path in sorted(records_dir.glob("*/*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        key = (record["puzzle_id"], record["row"], record["col"])
        seen.setdefault(key, _Identity(*key))
    return list(seen.values())


@dataclass(frozen=True)
class _Identity:
    """Lightweight (puzzle_id, row, col) identity holder for counting distinct physical pieces."""

    puzzle_id: str
    row: int
    col: int


if __name__ == "__main__":
    main()
