#!/usr/bin/env python3
"""M6 evaluation: cross-background piece re-identification via shape + color rank fusion.

Protocol: enroll every clean (non `corner_disagreement`) piece from ONE
background (the gallery), then query with the SAME physical pieces
photographed on each OTHER background - 4 x 3 = 12 (enroll, query) cells.
For each query, nearest neighbor is found over the entire cross-puzzle
gallery (the hard, primary setting; a secondary pass restricts the
candidate pool to the query's own puzzle).

Fingerprint fusion: M6's first iteration showed a linear z-score blend of
shape and color distances degenerates to shape-only (the sweep chose w=1);
iteration 2 fixed the combiner (reciprocal rank fusion) and the illuminant
sensitivity (gray-world a*b*), reaching 86.1% headline top-1 with the
residual failures dominated by rank-2/3 near-misses among same-palette
pieces - which a GLOBAL histogram cannot separate. This iteration adds a
SPATIAL color descriptor (3x3 grid of per-cell gray-world a*b* histograms
over the piece bbox, rotation-consistent with the shape edge shift k) and
compares three fusions:

- "rrf_ab": RRF(shape, global ab_gw), c=5 - iteration 2's frozen champion,
  the baseline to beat.
- "rrf_sp": RRF(shape, spatial color), c swept over {5, 10}.
- "rrf3": RRF(shape, global ab_gw, spatial color), same c for all terms,
  c swept over {5, 10}.

(rerank with K=5, iteration 2's frozen shortlist re-ranker, stays as an
ablation row.) All hyperparameters are chosen ONCE using only the 3
validation cells where the query background is "cardboard", then frozen -
the 9 remaining cells are the headline, leakage-free numbers.

Reports (print + `outputs/reid_eval.json`):
    a. Ablation (shape-only, color variants incl. spatial, rerank, all RRF
       fusions): top-1/top-5, aggregated over all 12 cells and over the 9
       non-validation cells.
    b. 12-cell (enroll x query) top-1 matrix for the winning fusion method.
    c. Fixed-orientation vs rotation-invariant shape base, winner top-1.
    d. Genuine/impostor stats for the winner + failure concentration by
       query background and puzzle; plus an easier-context line excluding
       wood-as-query cells (the real app scans on a gray mat).
    e. Failure review sheet for the winner's worst cell:
       `outputs/review_reid/failures_{enroll}_{query}.png`.
    A secondary ablation table restricted to within-puzzle galleries.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/fingerprint.py   # build galleries first
    uv run python experiments/exp28_piece_geometry/eval_reid.py
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from common import crop_with_margin, load_metadata  # noqa: E402
from fingerprint import (  # noqa: E402
    BACKGROUNDS,
    PieceFingerprint,
    build_gallery,
    chi_square_distance_matrix,
    load_gallery,
    save_fingerprints,
    shape_distance_matrix,
    spatial_color_distance_matrix,
)

# Frozen from iteration 2's validation sweep: the rerank shortlist size and
# the champion RRF(shape, ab_gw) c - the baseline the new candidates must beat.
RERANK_K = 5.0
RRF_AB_C = 5.0
# Iteration 3 fusion candidates (validated on cardboard-as-query cells only).
RRF_SPATIAL_C_SWEEP = (5.0, 10.0)
COLOR_VARIANTS = ("ab", "ab_gw")
VALIDATION_QUERY_BACKGROUND = "cardboard"

# Scalar offset placed on out-of-top-K candidates in the rerank fusion
# distance; chi-square color distances are <= 1, so any offset > 1 keeps
# every out-of-K candidate strictly behind every in-K candidate.
RERANK_OUT_OF_K_OFFSET = 10.0

MAX_FAILURES_PER_SHEET = 12
ABLATION_ROWS = (
    ("shape_ri", "shape-only"),
    ("color_lab", "color-lab"),
    ("color_ab", "color-ab"),
    ("color_ab_gw", "color-ab-gw"),
    ("color_spatial", "spatial-color"),
    ("fused_rerank", "shape+color-rerank"),
    ("fused_rrf_ab", "RRF(shape,ab_gw)"),
    ("fused_rrf_sp", "RRF(shape,spatial)"),
    ("fused_rrf3", "RRF(shape,ab_gw,spatial)"),
)

FUSION_DISPLAY = {
    "rerank": "shape+color-rerank",
    "rrf_ab": "RRF(shape,ab_gw)",
    "rrf_sp": "RRF(shape,spatial)",
    "rrf3": "RRF(shape,ab_gw,spatial)",
}


@dataclass
class GalleryArrays:
    """Stacked, vectorization-ready arrays for one background's enrolled gallery.

    Attributes:
        identities: (puzzle_id, row, col) per gallery piece, aligned with all other arrays.
        puzzle_ids: Puzzle id per gallery piece, as a numpy string array (for puzzle-restricted pooling).
        piece_files: Photo path (relative to dataset root) per gallery piece.
        edges: (N, 4, 100, 2) canonical edge polylines.
        types: (N, 4) string array of edge types, N/E/S/W order.
        hist_lab: (N, 512) L*a*b* joint histograms.
        hist_ab: (N, 256) a*b*-only histograms.
        hist_ab_gw: (N, 256) gray-world a*b* histograms.
        spatial_hists: (N, 9, 64) per-cell spatial gray-world a*b* histograms.
        spatial_nonempty: (N, 9) spatial-cell non-empty flags.
    """

    identities: List[Tuple[str, int, int]]
    puzzle_ids: np.ndarray
    piece_files: List[str]
    edges: np.ndarray
    types: np.ndarray
    hist_lab: np.ndarray
    hist_ab: np.ndarray
    hist_ab_gw: np.ndarray
    spatial_hists: np.ndarray
    spatial_nonempty: np.ndarray


@dataclass
class QueryBase:
    """Cached raw distance arrays from one query to one cell's candidate pool.

    Attributes:
        query: The query fingerprint.
        true_local: Index of the true identity within the pool arrays.
        pool_idx: Pool indices into the gallery's global arrays.
        d_shape_ri: (P,) rotation-invariant shape distances.
        d_shape_fixed: (P,) fixed-orientation (k=0) shape distances.
        d_color: {"lab"/"ab"/"ab_gw": (P,) chi-square color distances}.
        d_spatial: (P,) spatial color distances, each candidate's grid
            rotated by the rotation-invariant shape distance's winning k.
        d_spatial_fixed: (P,) spatial color distances at k=0 everywhere
            (the fixed-orientation protocol).
    """

    query: PieceFingerprint
    true_local: int
    pool_idx: np.ndarray
    d_shape_ri: np.ndarray
    d_shape_fixed: np.ndarray
    d_color: Dict[str, np.ndarray]
    d_spatial: np.ndarray
    d_spatial_fixed: np.ndarray


@dataclass
class CellBase:
    """All cached query bases for one (enroll, query) cell.

    Attributes:
        enroll_bg: Gallery background.
        query_bg: Query background.
        queries: One `QueryBase` per query whose identity exists in the gallery.
        n_skipped: Queries skipped because their identity is not enrolled.
    """

    enroll_bg: str
    query_bg: str
    queries: List[QueryBase]
    n_skipped: int


@dataclass
class QueryOutcome:
    """Final per-query ranks and winner-fusion stats for one cell.

    Attributes:
        identity: The query's (puzzle_id, row, col) identity.
        enroll_bg: Gallery background.
        query_bg: Query background.
        pool_size: Number of gallery candidates scored.
        ranks: Per-metric 1-based rank of the true identity (keys: the
            `ABLATION_ROWS` metrics plus shape_fixed, fused_winner, and
            fused_winner_fixed).
        winner_true_dist: Winner-fusion distance to the true gallery match.
        winner_best_impostor_dist: Winner-fusion distance to the best impostor.
        query_piece_file: Query photo path.
        true_piece_file: True gallery match's photo path.
        best_impostor_piece_file: Best-impostor gallery entry's photo path.
        best_impostor_identity: Best-impostor gallery entry's identity.
    """

    identity: Tuple[str, int, int]
    enroll_bg: str
    query_bg: str
    pool_size: int
    ranks: Dict[str, int]
    winner_true_dist: float
    winner_best_impostor_dist: float
    query_piece_file: str
    true_piece_file: str
    best_impostor_piece_file: str
    best_impostor_identity: Tuple[str, int, int]


def build_arrays(fingerprints: List[PieceFingerprint]) -> GalleryArrays:
    """Stack a list of piece fingerprints into vectorization-ready arrays.

    Args:
        fingerprints: One background's enrolled gallery.

    Returns:
        The stacked `GalleryArrays`.
    """
    return GalleryArrays(
        identities=[fp.identity for fp in fingerprints],
        puzzle_ids=np.array([fp.puzzle_id for fp in fingerprints]),
        piece_files=[fp.piece_file for fp in fingerprints],
        edges=np.stack([fp.edges_canonical for fp in fingerprints], axis=0),
        types=np.array([list(fp.edge_types) for fp in fingerprints]),
        hist_lab=np.stack([fp.color_hist_lab for fp in fingerprints], axis=0),
        hist_ab=np.stack([fp.color_hist_ab for fp in fingerprints], axis=0),
        hist_ab_gw=np.stack([fp.color_hist_ab_gw for fp in fingerprints], axis=0),
        spatial_hists=np.stack([fp.spatial_hists for fp in fingerprints], axis=0),
        spatial_nonempty=np.stack([fp.spatial_nonempty for fp in fingerprints], axis=0),
    )


def compute_cell_base(
    gallery: GalleryArrays,
    query_fingerprints: List[PieceFingerprint],
    enroll_bg: str,
    query_bg: str,
    restrict_puzzle: bool = False,
) -> CellBase:
    """Compute and cache every query's raw distance arrays for one cell.

    All fusion variants and ablation metrics are pure functions of these
    arrays, so the (comparatively expensive) shape/color distance
    computation happens exactly once per cell regardless of how many
    hyperparameter combinations are evaluated.

    Args:
        gallery: The enroll background's stacked gallery.
        query_fingerprints: The query background's fingerprints.
        enroll_bg: Enroll background name.
        query_bg: Query background name.
        restrict_puzzle: When True, restrict the candidate pool to the
            query's own puzzle (the secondary, easier setting).

    Returns:
        The cell's cached `CellBase`.
    """
    identity_to_idx = {identity: i for i, identity in enumerate(gallery.identities)}
    queries: List[QueryBase] = []
    n_skipped = 0

    for query in query_fingerprints:
        if query.identity not in identity_to_idx:
            n_skipped += 1
            continue
        true_idx = identity_to_idx[query.identity]

        if restrict_puzzle:
            pool_idx = np.where(gallery.puzzle_ids == query.puzzle_id)[0]
        else:
            pool_idx = np.arange(len(gallery.identities))
        true_local = int(np.where(pool_idx == true_idx)[0][0])

        edges_pool = gallery.edges[pool_idx]
        types_pool = gallery.types[pool_idx]
        spatial_pool = gallery.spatial_hists[pool_idx]
        spatial_nonempty_pool = gallery.spatial_nonempty[pool_idx]

        d_shape_ri, best_k_ri = shape_distance_matrix(
            query.edges_canonical, query.edge_types, edges_pool, types_pool, True
        )
        d_shape_fixed, _ = shape_distance_matrix(query.edges_canonical, query.edge_types, edges_pool, types_pool, False)
        queries.append(
            QueryBase(
                query=query,
                true_local=true_local,
                pool_idx=pool_idx,
                d_shape_ri=d_shape_ri,
                d_shape_fixed=d_shape_fixed,
                d_color={
                    "lab": chi_square_distance_matrix(query.color_hist_lab, gallery.hist_lab[pool_idx]),
                    "ab": chi_square_distance_matrix(query.color_hist_ab, gallery.hist_ab[pool_idx]),
                    "ab_gw": chi_square_distance_matrix(query.color_hist_ab_gw, gallery.hist_ab_gw[pool_idx]),
                },
                d_spatial=spatial_color_distance_matrix(
                    query.spatial_hists, query.spatial_nonempty, spatial_pool, spatial_nonempty_pool, best_k_ri
                ),
                d_spatial_fixed=spatial_color_distance_matrix(
                    query.spatial_hists,
                    query.spatial_nonempty,
                    spatial_pool,
                    spatial_nonempty_pool,
                    np.zeros(len(pool_idx), dtype=np.int64),
                ),
            )
        )

    return CellBase(enroll_bg=enroll_bg, query_bg=query_bg, queries=queries, n_skipped=n_skipped)


def ranks_from_distances(distances: np.ndarray) -> np.ndarray:
    """1-based rank of every candidate under ascending-distance ordering.

    Args:
        distances: (P,) candidate distances.

    Returns:
        (P,) integer ranks (1 = closest).
    """
    order = np.argsort(distances, kind="stable")
    ranks = np.empty(len(distances), dtype=np.int64)
    ranks[order] = np.arange(1, len(distances) + 1)
    return ranks


def fused_distance_array(
    base: QueryBase, method: str, param: float, color_variant: str, fixed_shape: bool = False
) -> np.ndarray:
    """Compute a fused distance-like scalar per candidate for one fusion method.

    Lower is better in all cases:
    - "rerank": shape top-K candidates get their color distance (so color
      alone orders the shortlist); everything else gets
      `RERANK_OUT_OF_K_OFFSET` + its shape rank, preserving shape order
      strictly behind the shortlist.
    - "rrf_ab" / "rrf_sp" / "rrf3": the negated reciprocal-rank-fusion score
      -(sum over terms of 1/(rank + c)), where the terms are shape + global
      color, shape + spatial color, or all three respectively.

    Args:
        base: The query's cached distance arrays.
        method: One of `FUSION_DISPLAY`'s keys.
        param: K for "rerank", c for the RRF methods.
        color_variant: Which global color distance feeds the fusion.
        fixed_shape: When True, use the fixed-orientation shape (and
            spatial-color) base.

    Returns:
        (P,) fused distances.
    """
    d_shape = base.d_shape_fixed if fixed_shape else base.d_shape_ri
    d_spatial = base.d_spatial_fixed if fixed_shape else base.d_spatial
    d_color = base.d_color[color_variant]
    shape_ranks = ranks_from_distances(d_shape)

    if method == "rerank":
        in_k = shape_ranks <= param
        return np.where(in_k, d_color, RERANK_OUT_OF_K_OFFSET + shape_ranks.astype(np.float64))

    rrf_terms = {"rrf_ab": (d_color,), "rrf_sp": (d_spatial,), "rrf3": (d_color, d_spatial)}
    if method not in rrf_terms:
        raise ValueError(f"Unknown fusion method: {method}")
    fused = -1.0 / (shape_ranks + param)
    for term in rrf_terms[method]:
        fused = fused - 1.0 / (ranks_from_distances(term) + param)
    return fused


def rank_of_true(distances: np.ndarray, true_idx: int) -> int:
    """1-based rank of the true index under ascending-distance nearest-neighbor search.

    Args:
        distances: (P,) candidate distances.
        true_idx: Index of the true match within `distances`.

    Returns:
        1 + the number of candidates strictly closer than the true match.
    """
    return 1 + int(np.sum(distances < distances[true_idx]))


def fused_true_rank(base: QueryBase, method: str, param: float, color_variant: str, fixed_shape: bool = False) -> int:
    """1-based fused rank of one query's true identity.

    Args:
        base: The query's cached distance arrays.
        method: One of `FUSION_DISPLAY`'s keys.
        param: K for "rerank", c for the RRF methods.
        color_variant: Which global color distance feeds the fusion.
        fixed_shape: When True, use the fixed-orientation shape base.

    Returns:
        The fused rank.
    """
    fused = fused_distance_array(base, method, param, color_variant, fixed_shape)
    return rank_of_true(fused, base.true_local)


def _top_k_ranks(ranks: List[int], k: int) -> float:
    """Fraction of ranks within the top k.

    Args:
        ranks: 1-based ranks.
        k: Rank cutoff.

    Returns:
        The top-k fraction, or nan for an empty list.
    """
    if not ranks:
        return float("nan")
    return float(np.mean(np.array(ranks) <= k))


def validation_selection(
    cells: Dict[Tuple[str, str], CellBase],
) -> Tuple[str, str, float, Dict[str, float], Dict[str, float]]:
    """Choose the color variant and the winning fusion on the validation cells only.

    Validation cells = the 3 cells whose query background is
    `VALIDATION_QUERY_BACKGROUND`. The global color variant ("ab" vs
    "ab_gw") is chosen by color-only top-1; then the iteration 3 candidates
    - the frozen baseline RRF(shape, ab_gw) c=5, RRF(shape, spatial) and
    RRF(shape, ab_gw, spatial) with c swept - are scored with that variant
    and the best combination wins.

    Args:
        cells: All 12 cells' cached bases, keyed by (enroll_bg, query_bg).

    Returns:
        Tuple of (chosen variant, chosen method, chosen param,
        {variant: color-only top1} scores, {"method=param": top1} sweep curve).
    """
    validation = [cell for (_e, q), cell in cells.items() if q == VALIDATION_QUERY_BACKGROUND]
    bases = [base for cell in validation for base in cell.queries]

    variant_scores = {
        variant: _top_k_ranks([rank_of_true(b.d_color[variant], b.true_local) for b in bases], 1)
        for variant in COLOR_VARIANTS
    }
    chosen_variant = max(variant_scores, key=lambda v: variant_scores[v])

    sweep: Dict[str, float] = {}
    candidates: List[Tuple[str, float]] = [("rrf_ab", RRF_AB_C)]
    candidates += [("rrf_sp", c) for c in RRF_SPATIAL_C_SWEEP]
    candidates += [("rrf3", c) for c in RRF_SPATIAL_C_SWEEP]
    for method, param in candidates:
        ranks = [fused_true_rank(b, method, param, chosen_variant) for b in bases]
        sweep[f"{method}={param:g}"] = _top_k_ranks(ranks, 1)

    chosen_method, chosen_param = max(candidates, key=lambda mp: sweep[f"{mp[0]}={mp[1]:g}"])
    return chosen_variant, chosen_method, chosen_param, variant_scores, sweep


def finalize_outcomes(
    cells: Dict[Tuple[str, str], CellBase],
    gallery_arrays: Dict[str, GalleryArrays],
    color_variant: str,
    winner_method: str,
    winner_param: float,
    rrf_sp_c: float,
    rrf3_c: float,
) -> Dict[Tuple[str, str], List[QueryOutcome]]:
    """Compute every cell's final per-query outcomes with frozen hyperparameters.

    Args:
        cells: All cells' cached bases.
        gallery_arrays: enroll background -> stacked gallery arrays.
        color_variant: The frozen global color variant.
        winner_method: The frozen winning fusion method (a `FUSION_DISPLAY` key).
        winner_param: The winner's frozen K/c.
        rrf_sp_c: Frozen c for the ablation's RRF(shape,spatial) row.
        rrf3_c: Frozen c for the ablation's RRF(shape,ab_gw,spatial) row.

    Returns:
        {(enroll_bg, query_bg): [QueryOutcome, ...]}.
    """
    outcomes: Dict[Tuple[str, str], List[QueryOutcome]] = {}
    for key, cell in cells.items():
        gallery = gallery_arrays[cell.enroll_bg]
        cell_outcomes: List[QueryOutcome] = []
        for base in cell.queries:
            fused_winner = fused_distance_array(base, winner_method, winner_param, color_variant)
            ranks = {
                "shape_ri": rank_of_true(base.d_shape_ri, base.true_local),
                "shape_fixed": rank_of_true(base.d_shape_fixed, base.true_local),
                "color_lab": rank_of_true(base.d_color["lab"], base.true_local),
                "color_ab": rank_of_true(base.d_color["ab"], base.true_local),
                "color_ab_gw": rank_of_true(base.d_color["ab_gw"], base.true_local),
                "color_spatial": rank_of_true(base.d_spatial, base.true_local),
                "fused_rerank": fused_true_rank(base, "rerank", RERANK_K, color_variant),
                "fused_rrf_ab": fused_true_rank(base, "rrf_ab", RRF_AB_C, color_variant),
                "fused_rrf_sp": fused_true_rank(base, "rrf_sp", rrf_sp_c, color_variant),
                "fused_rrf3": fused_true_rank(base, "rrf3", rrf3_c, color_variant),
                "fused_winner": rank_of_true(fused_winner, base.true_local),
                "fused_winner_fixed": fused_true_rank(base, winner_method, winner_param, color_variant, True),
            }

            masked = fused_winner.copy()
            masked[base.true_local] = np.inf
            best_imp_local = int(np.argmin(masked))
            best_imp_global = int(base.pool_idx[best_imp_local])
            true_global = int(base.pool_idx[base.true_local])

            cell_outcomes.append(
                QueryOutcome(
                    identity=base.query.identity,
                    enroll_bg=cell.enroll_bg,
                    query_bg=cell.query_bg,
                    pool_size=len(base.pool_idx),
                    ranks=ranks,
                    winner_true_dist=float(fused_winner[base.true_local]),
                    winner_best_impostor_dist=float(fused_winner[best_imp_local]),
                    query_piece_file=base.query.piece_file,
                    true_piece_file=gallery.piece_files[true_global],
                    best_impostor_piece_file=gallery.piece_files[best_imp_global],
                    best_impostor_identity=gallery.identities[best_imp_global],
                )
            )
        outcomes[key] = cell_outcomes
    return outcomes


def ablation_table(outcomes: List[QueryOutcome]) -> Dict[str, Dict[str, float]]:
    """Aggregate top-1/top-5 per ablation metric over a pool of outcomes.

    Args:
        outcomes: Query outcomes (typically pooled across cells).

    Returns:
        {display_name: {"top1": ..., "top5": ...}}.
    """
    table: Dict[str, Dict[str, float]] = {}
    for metric, display in ABLATION_ROWS:
        ranks = [o.ranks[metric] for o in outcomes]
        table[display] = {"top1": _top_k_ranks(ranks, 1), "top5": _top_k_ranks(ranks, 5)}
    return table


def genuine_impostor_stats(outcomes: List[QueryOutcome]) -> Dict[str, Optional[float]]:
    """Genuine vs. best-impostor winner-fusion distance stats over a pool of outcomes.

    Args:
        outcomes: Query outcomes.

    Returns:
        Dict with median genuine, median best-impostor, the impostor 5th
        percentile, and overlap (fraction of genuine distances above that
        percentile) - the same shape as `eval_matching.genuine_impostor_stats`.
    """
    if not outcomes:
        return {"median_genuine": None, "median_best_impostor": None, "impostor_p5": None, "overlap_frac": None}
    genuine = np.array([o.winner_true_dist for o in outcomes])
    impostor_best = np.array([o.winner_best_impostor_dist for o in outcomes])
    impostor_p5 = float(np.percentile(impostor_best, 5))
    overlap = float(np.mean(genuine > impostor_p5))
    return {
        "median_genuine": float(np.median(genuine)),
        "median_best_impostor": float(np.median(impostor_best)),
        "impostor_p5": impostor_p5,
        "overlap_frac": overlap,
    }


def failure_concentration(outcomes: List[QueryOutcome], top_n: int = 8) -> Dict[str, object]:
    """Break down winner-fusion top-1 failures by query background and by puzzle.

    Args:
        outcomes: Query outcomes (typically pooled across all 12 cells).
        top_n: How many puzzles to report.

    Returns:
        Dict with failure counts by query background and puzzle plus totals.
    """
    failures = [o for o in outcomes if o.ranks["fused_winner"] != 1]
    by_bg = Counter(o.query_bg for o in failures)
    by_puzzle = Counter(o.identity[0] for o in failures)
    return {
        "by_query_background": sorted(by_bg.items(), key=lambda kv: -kv[1]),
        "by_puzzle": sorted(by_puzzle.items(), key=lambda kv: -kv[1])[:top_n],
        "n_failures": len(failures),
        "n_total": len(outcomes),
    }


def load_bbox_lookup(dataset_root: Path) -> Dict[str, Tuple[int, int, int, int]]:
    """Map each piece photo's relative path to its bounding box, from `metadata.csv`.

    Args:
        dataset_root: The north_star v1 dataset root.

    Returns:
        {piece_file: (x1, y1, x2, y2)}.
    """
    return {r.piece_file: r.bbox for r in load_metadata(dataset_root)}


def render_failure_sheet(
    outcomes: List[QueryOutcome],
    enroll_bg: str,
    query_bg: str,
    dataset_root: Path,
    bbox_lookup: Dict[str, Tuple[int, int, int, int]],
    output_dir: Path,
) -> Optional[Path]:
    """Render a query/wrong-top-1/true-match crop comparison sheet for one cell's failures.

    Args:
        outcomes: This cell's query outcomes.
        enroll_bg: Enroll background name.
        query_bg: Query background name.
        dataset_root: The north_star v1 dataset root (for photo pixels).
        bbox_lookup: `load_bbox_lookup` output.
        output_dir: The experiment outputs directory.

    Returns:
        The written PNG path, or None when the cell has no failures.
    """
    failures = [o for o in outcomes if o.ranks["fused_winner"] != 1][:MAX_FAILURES_PER_SHEET]
    if not failures:
        return None

    def load_crop(piece_file: str) -> np.ndarray:
        image = cv2.imread(str(dataset_root / piece_file))
        crop, _ = crop_with_margin(image, bbox_lookup[piece_file])
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    n = len(failures)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    fig.suptitle(f"Worst-cell winner-fusion top-1 failures: enroll={enroll_bg}  query={query_bg}")
    for row, outcome in enumerate(failures):
        puzzle_id, piece_row, piece_col = outcome.identity
        imp_puzzle, imp_row, imp_col = outcome.best_impostor_identity
        cells = (
            (outcome.query_piece_file, f"query\n{puzzle_id} r{piece_row}c{piece_col} ({query_bg})"),
            (
                outcome.best_impostor_piece_file,
                f"wrong top-1\n{imp_puzzle} r{imp_row}c{imp_col}\nd={outcome.winner_best_impostor_dist:.3f}",
            ),
            (
                outcome.true_piece_file,
                f"true match (rank {outcome.ranks['fused_winner']})\nd={outcome.winner_true_dist:.3f}",
            ),
        )
        for col, (piece_file, title) in enumerate(cells):
            ax = axes[row][col]
            ax.imshow(load_crop(piece_file))
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    review_dir = output_dir / "review_reid"
    review_dir.mkdir(parents=True, exist_ok=True)
    out_path = review_dir / f"failures_{enroll_bg}_{query_bg}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _print_ablation(title: str, table: Dict[str, Dict[str, float]]) -> None:
    """Print an ablation top-1/top-5 table.

    Args:
        title: Table heading.
        table: `ablation_table` output.
    """
    print(f"\n{title}")
    print(f"{'metric':<22}{'top1':>8}{'top5':>8}")
    for _metric, display in ABLATION_ROWS:
        row = table[display]
        print(f"{display:<22}{row['top1'] * 100:7.1f}%{row['top5'] * 100:7.1f}%")


def main() -> None:  # noqa: C901  (evaluation driver, sequential report sections)
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate cross-background piece re-identification.")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/Users/claus/Repos/pussel/network/datasets/north_star/v1")
    )
    parser.add_argument("--records-dir", type=Path, default=Path(__file__).parent / "outputs" / "piece_records")
    parser.add_argument("--contours-dir", type=Path, default=Path(__file__).parent / "outputs" / "contours")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--no-sheets", action="store_true", help="Skip rendering the failure review sheet")
    args = parser.parse_args()

    fingerprints_dir = args.output_dir / "fingerprints"
    total_pieces = len({(r.puzzle_id, r.row, r.col) for r in load_metadata(args.dataset_root)})

    gallery_fps: Dict[str, List[PieceFingerprint]] = {}
    for background in BACKGROUNDS:
        fp_path = fingerprints_dir / f"{background}.json"
        if not fp_path.exists():
            print(f"{fp_path} missing - building it now.")
            built = build_gallery(args.records_dir, args.contours_dir, args.dataset_root, background)
            save_fingerprints(built, fp_path)
        gallery_fps[background] = load_gallery(fp_path, args.records_dir)
        print(f"{background:12s}: {len(gallery_fps[background]):3d}/{total_pieces} clean pieces enrolled (coverage)")

    galleries = {bg: build_arrays(fps) for bg, fps in gallery_fps.items()}

    cell_keys = [(e, q) for e in BACKGROUNDS for q in BACKGROUNDS if e != q]
    cells = {(e, q): compute_cell_base(galleries[e], gallery_fps[q], e, q) for e, q in cell_keys}
    cells_puzzle = {
        (e, q): compute_cell_base(galleries[e], gallery_fps[q], e, q, restrict_puzzle=True) for e, q in cell_keys
    }

    variant, method, param, variant_scores, sweep = validation_selection(cells)
    print(
        f"\nColor variant chosen on validation "
        f"(query={VALIDATION_QUERY_BACKGROUND}, color-only top-1): {variant_scores}"
    )
    print(f"Winner variant: {variant}")
    print(f"Fusion sweep on validation cells (top-1): {sweep}")
    print(f"Frozen winner: method={method}, param={param:g}, color variant={variant}")

    rrf_sp_c = max(RRF_SPATIAL_C_SWEEP, key=lambda c: sweep[f"rrf_sp={c:g}"])
    rrf3_c = max(RRF_SPATIAL_C_SWEEP, key=lambda c: sweep[f"rrf3={c:g}"])
    print(
        f"Frozen ablation-row params: rerank K={RERANK_K:g} (iter 2), rrf_ab c={RRF_AB_C:g} (iter 2), "
        f"rrf_sp c={rrf_sp_c:g}, rrf3 c={rrf3_c:g}"
    )

    outcomes_by_cell = finalize_outcomes(cells, galleries, variant, method, param, rrf_sp_c, rrf3_c)
    outcomes_puzzle_by_cell = finalize_outcomes(cells_puzzle, galleries, variant, method, param, rrf_sp_c, rrf3_c)
    all_outcomes = [o for outs in outcomes_by_cell.values() for o in outs]
    all_outcomes_puzzle = [o for outs in outcomes_puzzle_by_cell.values() for o in outs]
    n_skipped_total = sum(cell.n_skipped for cell in cells.values())

    print(f"\n{len(all_outcomes)} queries across 12 cells ({n_skipped_total} skipped: true identity not in gallery)")

    # (a) Ablation, cross-puzzle hard gallery.
    table_all = ablation_table(all_outcomes)
    _print_ablation("=== Ablation (all 12 cells, cross-puzzle gallery) ===", table_all)

    final_9 = [o for o in all_outcomes if o.query_bg != VALIDATION_QUERY_BACKGROUND]
    table_final9 = ablation_table(final_9)
    _print_ablation("=== Ablation (9 cells excluding the validation cells) - HEADLINE ===", table_final9)

    winner_display = FUSION_DISPLAY[method]
    headline_top1 = table_final9[winner_display]["top1"]
    headline_ranks_9 = [o.ranks["fused_winner"] for o in final_9]
    headline_top5 = _top_k_ranks(headline_ranks_9, 5)

    # Easier-context line: exclude wood-as-query cells from the headline set
    # (the production app scans on a gray mat in one session, so wood-query
    # cells overstate the real difficulty). NOT the headline number.
    final_6_no_wood = [o for o in final_9 if o.query_bg != "wood"]
    no_wood_ranks = [o.ranks["fused_winner"] for o in final_6_no_wood]
    print(
        f"\nEasier-context (leakage-free, EXCLUDING wood-as-query; {len(final_6_no_wood)} queries, 6 cells): "
        f"winner top-1 {_top_k_ranks(no_wood_ranks, 1) * 100:.1f}%, top-5 {_top_k_ranks(no_wood_ranks, 5) * 100:.1f}% "
        f"- context only, not the headline"
    )

    # (b) 12-cell winner matrix.
    print(f"\n=== 12-cell winner ({winner_display}) top-1 matrix (rows=enroll, cols=query) ===")
    print("enroll\\query".ljust(14) + "".join(f"{bg:>14}" for bg in BACKGROUNDS))
    matrix: Dict[str, Dict[str, float]] = {}
    for enroll_bg in BACKGROUNDS:
        row_vals = []
        matrix[enroll_bg] = {}
        for query_bg in BACKGROUNDS:
            if enroll_bg == query_bg:
                row_vals.append("--".rjust(14))
                continue
            top1 = _top_k_ranks([o.ranks["fused_winner"] for o in outcomes_by_cell[(enroll_bg, query_bg)]], 1)
            matrix[enroll_bg][query_bg] = top1
            row_vals.append(f"{top1 * 100:13.1f}%")
        print(enroll_bg.ljust(14) + "".join(row_vals))

    # (c) Fixed-orientation vs rotation-invariant shape base for the winner.
    top1_ri = _top_k_ranks([o.ranks["fused_winner"] for o in all_outcomes], 1)
    top1_fixed = _top_k_ranks([o.ranks["fused_winner_fixed"] for o in all_outcomes], 1)
    print(
        f"\nFixed-orientation vs rotation-invariant shape base (winner top-1, all 12 cells): "
        f"fixed={top1_fixed * 100:.1f}%  rotation-invariant={top1_ri * 100:.1f}%"
    )

    # (d) Genuine/impostor stats + failure concentration.
    gi_stats = genuine_impostor_stats(all_outcomes)
    print(
        f"\nGenuine vs impostor (winner fusion distance): median genuine={gi_stats['median_genuine']:.4f}, "
        f"median best-impostor={gi_stats['median_best_impostor']:.4f}, impostor p5={gi_stats['impostor_p5']:.4f}, "
        f"overlap={gi_stats['overlap_frac'] * 100:.1f}% of genuine above impostor p5"
    )
    failure_stats = failure_concentration(all_outcomes)
    print(f"\nFailure concentration ({failure_stats['n_failures']}/{failure_stats['n_total']} winner top-1 misses):")
    print(f"  by query background: {failure_stats['by_query_background']}")
    print(f"  by puzzle (top 8):   {failure_stats['by_puzzle']}")

    # (e) Worst-cell failure review sheet.
    worst_enroll, worst_query = min(((e, q) for e in matrix for q in matrix[e]), key=lambda eq: matrix[eq[0]][eq[1]])
    worst_top1 = matrix[worst_enroll][worst_query]
    print(f"\nWorst cell (winner top-1): enroll={worst_enroll} query={worst_query} -> {worst_top1 * 100:.1f}%")

    sheet_path: Optional[Path] = None
    if not args.no_sheets:
        bbox_lookup = load_bbox_lookup(args.dataset_root)
        sheet_path = render_failure_sheet(
            outcomes_by_cell[(worst_enroll, worst_query)],
            worst_enroll,
            worst_query,
            args.dataset_root,
            bbox_lookup,
            args.output_dir,
        )
        if sheet_path:
            print(f"Wrote {sheet_path}")

    # Secondary: within-puzzle gallery.
    table_puzzle = ablation_table(all_outcomes_puzzle)
    _print_ablation("=== Secondary: ablation with WITHIN-PUZZLE gallery only (all 12 cells) ===", table_puzzle)
    winner_puzzle_ranks = [o.ranks["fused_winner"] for o in all_outcomes_puzzle]
    print(
        f"Within-puzzle winner ({winner_display}): top-1 {_top_k_ranks(winner_puzzle_ranks, 1) * 100:.1f}%, "
        f"top-5 {_top_k_ranks(winner_puzzle_ranks, 5) * 100:.1f}%"
    )

    criterion_met = headline_top1 >= 0.95
    print(
        f"\nPlan criterion (winner top-1 >= 95%, headline 9 non-validation cells): "
        f"top-1 {headline_top1 * 100:.1f}% / top-5 {headline_top5 * 100:.1f}% "
        f"-> {'MET' if criterion_met else 'NOT MET'}"
    )

    eval_path = args.output_dir / "reid_eval.json"
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "coverage": {bg: len(gallery_fps[bg]) for bg in BACKGROUNDS},
                "total_physical_pieces": total_pieces,
                "chosen_color_variant": variant,
                "chosen_fusion": {"method": method, "param": param},
                "frozen_ablation_params": {
                    "rerank_k": RERANK_K,
                    "rrf_ab_c": RRF_AB_C,
                    "rrf_sp_c": rrf_sp_c,
                    "rrf3_c": rrf3_c,
                },
                "validation_color_variant_scores": variant_scores,
                "validation_fusion_sweep": sweep,
                "n_queries": len(all_outcomes),
                "n_skipped_no_gallery_match": n_skipped_total,
                "ablation_all_12_cells": table_all,
                "ablation_final_9_cells": table_final9,
                "ablation_within_puzzle": table_puzzle,
                "winner_matrix_12_cell": matrix,
                "fixed_vs_rotation_invariant_winner_top1": {"fixed": top1_fixed, "rotation_invariant": top1_ri},
                "genuine_impostor_winner": gi_stats,
                "failure_concentration": failure_stats,
                "worst_cell": {
                    "enroll": worst_enroll,
                    "query": worst_query,
                    "top1": worst_top1,
                    "review_sheet": str(sheet_path) if sheet_path else None,
                },
                "headline_top1_final_9_cells": headline_top1,
                "headline_top5_final_9_cells": headline_top5,
                "easier_context_no_wood_query": {
                    "n_cells": 6,
                    "n_queries": len(final_6_no_wood),
                    "top1": _top_k_ranks(no_wood_ranks, 1),
                    "top5": _top_k_ranks(no_wood_ranks, 5),
                },
                "plan_criterion_met_final_9_cells": criterion_met,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {eval_path}")


if __name__ == "__main__":
    main()
