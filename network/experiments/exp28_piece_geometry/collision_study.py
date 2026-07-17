#!/usr/bin/env python3
"""M7: uniqueness/collision study - accept/reject thresholds in DISTANCE space.

M6's retrieval winner (reciprocal rank fusion) is rank-based: its scores
depend on gallery composition and are not valid absolute similarities for
thresholding. M7 therefore works in raw distance space, using M6's two
strong component distances - rotation-invariant canonical-polyline shape L2
and 3x3 gray-world a*b* spatial color chi-square - combined into one
thresholdable score:

    z = (d_shape - mu_s) / sd_s + (d_spatial - mu_sp) / sd_sp

where (mu, sd) are the mean/std of IMPOSTOR (distinct-identity) pairwise
distances WITHIN the enroll gallery. Per-gallery normalization is what
makes one threshold transferable across galleries: each gallery's z is
expressed in units of its own impostor spread, so a threshold learned on
one enroll background applies to another without re-tuning.

Analyses (all 12 M6 enroll x query cells; thresholds are FROZEN on the 3
cardboard-as-query validation cells and error rates reported on the other
9, consistent with M6's selection discipline):

1. Genuine/impostor extraction: per query, the genuine z (to its true
   identity) and the best-impostor z (min over all wrong identities); full
   sample arrays dumped to `outputs/collision_samples.npz`.
2. Verification analysis ("is the top-1 match actually this piece?"):
   threshold sweep reporting FMR (wrong top-1 accepted) and FNR (correct
   top-1 rejected) as fractions of all queries, EER, FNR at FMR = 1% and
   0.1%, ROC data + plot.
3. New-piece detection (enrollment dedupe): removing the query's identity
   from the gallery makes its best match an impostor by construction, so
   the best-impostor samples ARE the new-piece score distribution;
   false-merge rate = fraction below the accept threshold.
4. Two-threshold recommendation: accept if z < t_accept (validation
   best-impostor 1st percentile), declare new if z > t_new (validation
   genuine 99th percentile), gray zone in between; gray-zone occupancy
   reported for both genuine and new-piece samples.
5. Per-puzzle die-cut collision study (shape only): within each puzzle x
   gray_fabric, all distinct-piece pairwise shape distances; a collision is
   a pair below the median GENUINE cross-background shape distance (two
   different pieces as alike as the same piece re-photographed). Pair
   sheets rendered for the 2 tightest puzzles (their nearest pairs,
   whether or not any pair crosses the collision bar).

Outputs: `outputs/collision_eval.json`, `outputs/collision_samples.npz`,
plots in `outputs/collision_plots/`.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/fingerprint.py     # galleries first
    uv run python experiments/exp28_piece_geometry/collision_study.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from common import crop_with_margin  # noqa: E402
from eval_reid import (  # noqa: E402
    GalleryArrays,
    build_arrays,
    compute_cell_base,
    load_bbox_lookup,
)
from fingerprint import (  # noqa: E402
    BACKGROUNDS,
    PieceFingerprint,
    load_gallery,
    shape_distance_matrix,
    spatial_color_distance_matrix,
)

VALIDATION_QUERY_BACKGROUND = "cardboard"

# Background used for the within-puzzle die-cut collision study (the best
# segmented background, so shape distances reflect die-cut geometry, not
# segmentation noise).
COLLISION_BACKGROUND = "gray_fabric"

# Validation quantiles defining the frozen threshold pair: t_accept admits
# at most this fraction of new-piece (best-impostor) scores, t_new rejects
# at most this fraction of genuine scores.
ACCEPT_QUANTILE = 0.01
NEW_QUANTILE = 0.99

N_THRESHOLD_GRID = 2001
N_COLLISION_SHEETS = 2
N_COLLISION_PAIRS_PER_SHEET = 4


@dataclass
class ZStats:
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


@dataclass
class Samples:
    """Aligned genuine/best-impostor score samples over all evaluated queries.

    Attributes:
        enroll_bg: Enroll background per query.
        query_bg: Query background per query.
        puzzle: Puzzle id per query.
        row: Piece grid row per query.
        col: Piece grid col per query.
        z_genuine: Combined z to the query's true identity.
        z_best_impostor: Minimum combined z over all wrong identities.
        d_shape_genuine: Raw genuine shape distance.
        d_spatial_genuine: Raw genuine spatial color distance.
    """

    enroll_bg: np.ndarray
    query_bg: np.ndarray
    puzzle: np.ndarray
    row: np.ndarray
    col: np.ndarray
    z_genuine: np.ndarray
    z_best_impostor: np.ndarray
    d_shape_genuine: np.ndarray
    d_spatial_genuine: np.ndarray


def gallery_z_stats(arrays: GalleryArrays, fingerprints: List[PieceFingerprint]) -> ZStats:
    """Impostor mean/std of shape and spatial distances within one enroll gallery.

    Every within-gallery pair is a distinct-identity (impostor) pair, since
    a gallery holds one entry per physical piece. Spatial distances use the
    shape distance's winning rotation k per pair, matching query-time
    scoring.

    Args:
        arrays: The gallery's stacked arrays.
        fingerprints: The gallery's fingerprints (for typed edge signatures).

    Returns:
        The gallery's `ZStats`.
    """
    shape_parts: List[np.ndarray] = []
    spatial_parts: List[np.ndarray] = []
    n = len(fingerprints)
    for i, fp in enumerate(fingerprints):
        rest = np.arange(n) != i
        d_shape, best_k = shape_distance_matrix(
            fp.edges_canonical, fp.edge_types, arrays.edges[rest], arrays.types[rest], True
        )
        d_spatial = spatial_color_distance_matrix(
            fp.spatial_hists,
            fp.spatial_nonempty,
            arrays.spatial_hists[rest],
            arrays.spatial_nonempty[rest],
            best_k,
        )
        shape_parts.append(d_shape)
        spatial_parts.append(d_spatial)
    shape_all = np.concatenate(shape_parts)
    spatial_all = np.concatenate(spatial_parts)
    return ZStats(
        shape_mean=float(shape_all.mean()),
        shape_std=float(shape_all.std()),
        spatial_mean=float(spatial_all.mean()),
        spatial_std=float(spatial_all.std()),
    )


def collect_samples(
    galleries: Dict[str, GalleryArrays],
    gallery_fps: Dict[str, List[PieceFingerprint]],
    stats_by_bg: Dict[str, ZStats],
) -> Samples:
    """Compute genuine and best-impostor combined z for every query in all 12 cells.

    Args:
        galleries: enroll background -> stacked gallery arrays.
        gallery_fps: background -> fingerprints (used as both galleries and queries).
        stats_by_bg: enroll background -> z-normalization stats.

    Returns:
        The pooled `Samples`.
    """
    rows: List[Tuple[str, str, str, int, int, float, float, float, float]] = []
    for enroll_bg in BACKGROUNDS:
        stats = stats_by_bg[enroll_bg]
        for query_bg in BACKGROUNDS:
            if enroll_bg == query_bg:
                continue
            cell = compute_cell_base(galleries[enroll_bg], gallery_fps[query_bg], enroll_bg, query_bg)
            for base in cell.queries:
                z = (base.d_shape_ri - stats.shape_mean) / max(stats.shape_std, 1e-9) + (
                    base.d_spatial - stats.spatial_mean
                ) / max(stats.spatial_std, 1e-9)
                z_genuine = float(z[base.true_local])
                masked = z.copy()
                masked[base.true_local] = np.inf
                z_best_imp = float(masked.min())
                puzzle_id, row, col = base.query.identity
                rows.append(
                    (
                        enroll_bg,
                        query_bg,
                        puzzle_id,
                        row,
                        col,
                        z_genuine,
                        z_best_imp,
                        float(base.d_shape_ri[base.true_local]),
                        float(base.d_spatial[base.true_local]),
                    )
                )

    return Samples(
        enroll_bg=np.array([r[0] for r in rows]),
        query_bg=np.array([r[1] for r in rows]),
        puzzle=np.array([r[2] for r in rows]),
        row=np.array([r[3] for r in rows], dtype=np.int64),
        col=np.array([r[4] for r in rows], dtype=np.int64),
        z_genuine=np.array([r[5] for r in rows]),
        z_best_impostor=np.array([r[6] for r in rows]),
        d_shape_genuine=np.array([r[7] for r in rows]),
        d_spatial_genuine=np.array([r[8] for r in rows]),
    )


def verification_curves(z_genuine: np.ndarray, z_best_impostor: np.ndarray) -> Dict[str, np.ndarray]:
    """Operational FMR/FNR threshold sweep for top-1 verification.

    The system returns the z-nearest candidate; its score is
    min(z_genuine, z_best_impostor) and it is correct iff the genuine score
    wins. FMR(t) = fraction of ALL queries whose top-1 is wrong AND
    accepted (score < t); FNR(t) = fraction whose top-1 is correct but
    rejected (score >= t).

    Args:
        z_genuine: (Q,) genuine combined z per query.
        z_best_impostor: (Q,) best-impostor combined z per query.

    Returns:
        {"thresholds", "fmr", "fnr"} arrays over an even threshold grid.
    """
    z_top1 = np.minimum(z_genuine, z_best_impostor)
    top1_correct = z_genuine <= z_best_impostor
    lo = float(z_top1.min()) - 0.5
    hi = float(z_top1.max()) + 0.5
    thresholds = np.linspace(lo, hi, N_THRESHOLD_GRID)
    fmr = np.array([np.mean(~top1_correct & (z_top1 < t)) for t in thresholds])
    fnr = np.array([np.mean(top1_correct & (z_top1 >= t)) for t in thresholds])
    return {"thresholds": thresholds, "fmr": fmr, "fnr": fnr}


def eer_and_operating_points(curves: Dict[str, np.ndarray]) -> Dict[str, float]:
    """EER and FNR at fixed FMR operating points from a threshold sweep.

    Args:
        curves: `verification_curves` output.

    Returns:
        Dict with eer, eer_threshold, and fnr/threshold at FMR = 1% and 0.1%.
    """
    thresholds = curves["thresholds"]
    fmr = curves["fmr"]
    fnr = curves["fnr"]
    eer_idx = int(np.argmin(np.abs(fmr - fnr)))
    out = {
        "eer": float((fmr[eer_idx] + fnr[eer_idx]) / 2.0),
        "eer_threshold": float(thresholds[eer_idx]),
    }
    for target, label in ((0.01, "1pct"), (0.001, "0p1pct")):
        ok = np.where(fmr <= target)[0]
        idx = int(ok[-1]) if len(ok) else 0
        out[f"fnr_at_fmr_{label}"] = float(fnr[idx])
        out[f"threshold_at_fmr_{label}"] = float(thresholds[idx])
    return out


def threshold_report(
    z_genuine: np.ndarray, z_best_impostor: np.ndarray, t_accept: float, t_new: float
) -> Dict[str, float]:
    """Error rates of a frozen (t_accept, t_new) pair on one sample set.

    Args:
        z_genuine: (Q,) genuine z (the re-scan scenario's correct-match scores).
        z_best_impostor: (Q,) best-impostor z (the new-piece scenario's scores).
        t_accept: Accept threshold (z below -> lock onto the match).
        t_new: New-piece threshold (z above -> declare unseen piece).

    Returns:
        Dict with FMR/FNR at t_accept, false-merge and false-new rates, and
        gray-zone occupancy for genuine and new-piece samples.
    """
    z_top1 = np.minimum(z_genuine, z_best_impostor)
    top1_correct = z_genuine <= z_best_impostor
    return {
        "fmr_at_t_accept": float(np.mean(~top1_correct & (z_top1 < t_accept))),
        "fnr_at_t_accept": float(np.mean(top1_correct & (z_top1 >= t_accept))),
        "false_merge_rate": float(np.mean(z_best_impostor < t_accept)),
        "false_new_rate": float(np.mean(z_genuine > t_new)),
        "gray_zone_genuine": float(np.mean((z_genuine >= t_accept) & (z_genuine <= t_new))),
        "gray_zone_new_piece": float(np.mean((z_best_impostor >= t_accept) & (z_best_impostor <= t_new))),
    }


@dataclass
class CollisionPair:
    """One distinct-piece pair and its shape distance within a puzzle.

    Attributes:
        idx_a: First piece's index into the gallery arrays.
        idx_b: Second piece's index.
        distance: Rotation-invariant shape distance between the two pieces.
    """

    idx_a: int
    idx_b: int
    distance: float


def puzzle_pair_distances(
    arrays: GalleryArrays, fingerprints: List[PieceFingerprint], puzzle_id: str
) -> List[CollisionPair]:
    """All distinct-piece pairwise shape distances within one puzzle, ascending.

    Args:
        arrays: The collision background's stacked gallery.
        fingerprints: The same gallery's fingerprints.
        puzzle_id: Puzzle to analyze.

    Returns:
        Every distinct pair as a `CollisionPair`, sorted by ascending distance.
    """
    idx = [i for i, fp in enumerate(fingerprints) if fp.puzzle_id == puzzle_id]
    pairs: List[CollisionPair] = []
    for pos, i in enumerate(idx):
        others = idx[pos + 1 :]
        if not others:
            continue
        others_arr = np.array(others)
        d, _ = shape_distance_matrix(
            fingerprints[i].edges_canonical,
            fingerprints[i].edge_types,
            arrays.edges[others_arr],
            arrays.types[others_arr],
            True,
        )
        pairs.extend(CollisionPair(idx_a=i, idx_b=others[j], distance=float(dist)) for j, dist in enumerate(d))
    pairs.sort(key=lambda p: p.distance)
    return pairs


def render_collision_sheet(
    pairs: List[CollisionPair],
    fingerprints: List[PieceFingerprint],
    puzzle_id: str,
    dataset_root: Path,
    bbox_lookup: Dict[str, Tuple[int, int, int, int]],
    plots_dir: Path,
) -> Path:
    """Render side-by-side crops of a puzzle's nearest distinct-piece pairs.

    Args:
        pairs: The puzzle's pairs, ascending by distance (nearest first).
        fingerprints: The collision background's gallery fingerprints.
        puzzle_id: Puzzle being rendered.
        dataset_root: The north_star v1 dataset root.
        bbox_lookup: `load_bbox_lookup` output.
        plots_dir: The collision plots directory.

    Returns:
        The written PNG path.
    """
    shown = pairs[:N_COLLISION_PAIRS_PER_SHEET]
    fig, axes = plt.subplots(len(shown), 2, figsize=(7, 3.2 * len(shown)), squeeze=False)
    fig.suptitle(f"Nearest die-cut shape pairs: {puzzle_id} ({COLLISION_BACKGROUND})")
    for row, pair in enumerate(shown):
        for col, piece_idx in enumerate((pair.idx_a, pair.idx_b)):
            fp = fingerprints[piece_idx]
            image = cv2.imread(str(dataset_root / fp.piece_file))
            crop, _ = crop_with_margin(image, bbox_lookup[fp.piece_file])
            ax = axes[row][col]
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            ax.set_title(f"r{fp.row}c{fp.col}" + (f"   shape d={pair.distance:.4f}" if col == 0 else ""), fontsize=9)
            ax.axis("off")
    out_path = plots_dir / f"collisions_{puzzle_id}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_histograms(samples: Samples, plots_dir: Path) -> Path:
    """Plot genuine vs best-impostor z histograms, one panel per query background.

    Args:
        samples: The pooled samples.
        plots_dir: The collision plots directory.

    Returns:
        The written PNG path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, query_bg in zip(axes.flat, BACKGROUNDS):
        mask = samples.query_bg == query_bg
        bins = np.linspace(
            min(samples.z_genuine[mask].min(), samples.z_best_impostor[mask].min()),
            max(samples.z_genuine[mask].max(), samples.z_best_impostor[mask].max()),
            60,
        )
        ax.hist(samples.z_genuine[mask], bins=bins, alpha=0.6, label="genuine", color="tab:green")
        ax.hist(samples.z_best_impostor[mask], bins=bins, alpha=0.6, label="best impostor", color="tab:red")
        ax.set_title(f"query background: {query_bg}")
        ax.set_xlabel("combined z")
        ax.legend()
    fig.tight_layout()
    out_path = plots_dir / "genuine_impostor_hist.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_roc(curves: Dict[str, np.ndarray], points: Dict[str, float], plots_dir: Path) -> Path:
    """Plot the FNR-vs-FMR trade-off curve with the EER and operating points marked.

    Args:
        curves: `verification_curves` output (9 test cells).
        points: `eer_and_operating_points` output.
        plots_dir: The collision plots directory.

    Returns:
        The written PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    positive = curves["fmr"] > 0
    ax.plot(curves["fmr"][positive], curves["fnr"][positive], "b-")
    ax.scatter([points["eer"]], [points["eer"]], color="k", zorder=3, label=f"EER = {points['eer'] * 100:.2f}%")
    for target, label in ((0.01, "1pct"), (0.001, "0p1pct")):
        fnr = points[f"fnr_at_fmr_{label}"]
        ax.scatter([target], [fnr], zorder=3, label=f"FNR@FMR={target * 100:g}% -> {fnr * 100:.2f}%")
    ax.set_xscale("log")
    ax.set_xlabel("FMR (wrong top-1 accepted, fraction of all queries)")
    ax.set_ylabel("FNR (correct top-1 rejected, fraction of all queries)")
    ax.set_title("Top-1 verification trade-off, combined z (9 non-validation cells)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    out_path = plots_dir / "roc.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_collision_rates(rates: Dict[str, float], plots_dir: Path) -> Path:
    """Bar chart of per-puzzle die-cut collision rates.

    Args:
        rates: {puzzle_id: collision rate}.
        plots_dir: The collision plots directory.

    Returns:
        The written PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    names = sorted(rates, key=lambda p: -rates[p])
    ax.bar(range(len(names)), [rates[p] * 100 for p in names], color="tab:blue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("collision rate (% of distinct pairs)")
    ax.set_title(f"Die-cut shape collision rate per puzzle ({COLLISION_BACKGROUND})")
    fig.tight_layout()
    out_path = plots_dir / "collision_rates.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def main() -> None:  # noqa: C901  (evaluation driver, sequential report sections)
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Uniqueness/collision study in distance space.")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/Users/claus/Repos/pussel/network/datasets/north_star/v1")
    )
    parser.add_argument("--records-dir", type=Path, default=Path(__file__).parent / "outputs" / "piece_records")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--no-sheets", action="store_true", help="Skip rendering plots and collision sheets")
    args = parser.parse_args()

    plots_dir = args.output_dir / "collision_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gallery_fps: Dict[str, List[PieceFingerprint]] = {}
    for background in BACKGROUNDS:
        fp_path = args.output_dir / "fingerprints" / f"{background}.json"
        if not fp_path.exists():
            raise SystemExit(f"{fp_path} missing - run fingerprint.py first.")
        gallery_fps[background] = load_gallery(fp_path, args.records_dir)
        print(f"{background:12s}: {len(gallery_fps[background]):3d} pieces enrolled")
    galleries = {bg: build_arrays(fps) for bg, fps in gallery_fps.items()}

    print("\nPer-gallery impostor z-normalization stats (shape mean/std, spatial mean/std):")
    stats_by_bg: Dict[str, ZStats] = {}
    for bg in BACKGROUNDS:
        stats = gallery_z_stats(galleries[bg], gallery_fps[bg])
        stats_by_bg[bg] = stats
        print(
            f"  {bg:12s}: shape {stats.shape_mean:.4f}/{stats.shape_std:.4f}   "
            f"spatial {stats.spatial_mean:.4f}/{stats.spatial_std:.4f}"
        )

    samples = collect_samples(galleries, gallery_fps, stats_by_bg)
    npz_path = args.output_dir / "collision_samples.npz"
    np.savez(
        npz_path,
        enroll_bg=samples.enroll_bg,
        query_bg=samples.query_bg,
        puzzle=samples.puzzle,
        row=samples.row,
        col=samples.col,
        z_genuine=samples.z_genuine,
        z_best_impostor=samples.z_best_impostor,
        d_shape_genuine=samples.d_shape_genuine,
        d_spatial_genuine=samples.d_spatial_genuine,
    )
    print(f"\n{len(samples.z_genuine)} query samples across 12 cells -> {npz_path}")

    top1_z = float(np.mean(samples.z_genuine <= samples.z_best_impostor))
    print(f"Context: top-1 accuracy of the thresholdable z score itself (all 12 cells): {top1_z * 100:.1f}%")

    is_validation = samples.query_bg == VALIDATION_QUERY_BACKGROUND
    val_genuine = samples.z_genuine[is_validation]
    val_impostor = samples.z_best_impostor[is_validation]
    test_genuine = samples.z_genuine[~is_validation]
    test_impostor = samples.z_best_impostor[~is_validation]

    # (2) Verification threshold sweep on the 9 test cells.
    curves = verification_curves(test_genuine, test_impostor)
    points = eer_and_operating_points(curves)
    print(
        f"\nVerification (9 non-validation cells): EER = {points['eer'] * 100:.2f}% (z = {points['eer_threshold']:.3f})"
    )
    print(
        f"  FNR @ FMR=1%:   {points['fnr_at_fmr_1pct'] * 100:.2f}%  (z = {points['threshold_at_fmr_1pct']:.3f})\n"
        f"  FNR @ FMR=0.1%: {points['fnr_at_fmr_0p1pct'] * 100:.2f}%  (z = {points['threshold_at_fmr_0p1pct']:.3f})"
    )

    # (3+4) Two-threshold recommendation, frozen on validation.
    t_accept = float(np.quantile(val_impostor, ACCEPT_QUANTILE))
    t_new = float(np.quantile(val_genuine, NEW_QUANTILE))
    print(
        f"\nFrozen thresholds (validation query={VALIDATION_QUERY_BACKGROUND}): "
        f"t_accept = {t_accept:.3f} (impostor p{ACCEPT_QUANTILE * 100:g}), "
        f"t_new = {t_new:.3f} (genuine p{NEW_QUANTILE * 100:g})"
    )
    if t_accept >= t_new:
        print("  NOTE: t_accept >= t_new - the distributions separate cleanly enough that no gray zone is needed.")
    report = threshold_report(test_genuine, test_impostor, t_accept, t_new)
    print("Frozen-threshold error rates on the 9 non-validation cells:")
    print(
        f"  accept (z < t_accept):  FMR = {report['fmr_at_t_accept'] * 100:.2f}%   "
        f"FNR = {report['fnr_at_t_accept'] * 100:.2f}%"
    )
    print(
        f"  new-piece scenario:     false-merge = {report['false_merge_rate'] * 100:.2f}%   "
        f"false-new (genuine > t_new) = {report['false_new_rate'] * 100:.2f}%"
    )
    print(
        f"  gray zone occupancy:    genuine {report['gray_zone_genuine'] * 100:.2f}%   "
        f"new-piece {report['gray_zone_new_piece'] * 100:.2f}%"
    )

    # (5) Per-puzzle die-cut collision study (shape only).
    shape_bar = float(np.median(samples.d_shape_genuine))
    print(f"\nDie-cut collision bar = median genuine cross-background shape distance = {shape_bar:.4f}")
    collision_fps = gallery_fps[COLLISION_BACKGROUND]
    collision_arrays = galleries[COLLISION_BACKGROUND]
    puzzle_ids = sorted({fp.puzzle_id for fp in collision_fps})
    collision_table: Dict[str, Dict[str, object]] = {}
    collision_rates: Dict[str, float] = {}
    all_pairs: Dict[str, List[CollisionPair]] = {}
    print(f"\n{'puzzle':<28}{'pairs':>7}{'collisions':>12}{'rate':>8}{'min d':>9}{'/bar':>7}  nearest pair")
    for puzzle_id in puzzle_ids:
        pairs = puzzle_pair_distances(collision_arrays, collision_fps, puzzle_id)
        all_pairs[puzzle_id] = pairs
        n_total = len(pairs)
        n_collisions = sum(1 for p in pairs if p.distance < shape_bar)
        rate = n_collisions / n_total if n_total else 0.0
        collision_rates[puzzle_id] = rate
        nearest = pairs[0] if pairs else None
        nearest_desc = "-"
        min_d = float("nan")
        if nearest is not None:
            fp_a = collision_fps[nearest.idx_a]
            fp_b = collision_fps[nearest.idx_b]
            nearest_desc = f"r{fp_a.row}c{fp_a.col} vs r{fp_b.row}c{fp_b.col}"
            min_d = nearest.distance
        collision_table[puzzle_id] = {
            "n_pairs": n_total,
            "n_collisions": n_collisions,
            "collision_rate": rate,
            "min_pair_distance": min_d,
            "min_over_bar": min_d / shape_bar if shape_bar > 0 else float("nan"),
            "nearest_pair": nearest_desc,
        }
        print(
            f"{puzzle_id:<28}{n_total:>7}{n_collisions:>12}{rate * 100:>7.1f}%"
            f"{min_d:>9.4f}{min_d / shape_bar:>6.1f}x  {nearest_desc}"
        )

    overall_pairs = sum(int(t["n_pairs"]) for t in collision_table.values())  # type: ignore[call-overload]
    overall_collisions = sum(int(t["n_collisions"]) for t in collision_table.values())  # type: ignore[call-overload]
    print(
        f"{'OVERALL':<28}{overall_pairs:>7}{overall_collisions:>12}"
        f"{overall_collisions / max(overall_pairs, 1) * 100:>7.1f}%"
    )

    sheet_paths: List[str] = []
    if not args.no_sheets:
        plot_histograms(samples, plots_dir)
        plot_roc(curves, points, plots_dir)
        plot_collision_rates(collision_rates, plots_dir)
        bbox_lookup = load_bbox_lookup(args.dataset_root)
        # Sheets for the 2 tightest puzzles (smallest nearest-pair distance);
        # with zero collisions these show the closest NEAR-collisions, which
        # is what needs visual verification.
        tightest = sorted(
            (p for p in puzzle_ids if all_pairs[p]),
            key=lambda p: all_pairs[p][0].distance,
        )[:N_COLLISION_SHEETS]
        for puzzle_id in tightest:
            path = render_collision_sheet(
                all_pairs[puzzle_id], collision_fps, puzzle_id, args.dataset_root, bbox_lookup, plots_dir
            )
            sheet_paths.append(str(path))
            print(f"Wrote {path}")
        print(f"Plots in {plots_dir}")

    eval_path = args.output_dir / "collision_eval.json"
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_samples": len(samples.z_genuine),
                "z_stats_per_gallery": {
                    bg: {
                        "shape_mean": s.shape_mean,
                        "shape_std": s.shape_std,
                        "spatial_mean": s.spatial_mean,
                        "spatial_std": s.spatial_std,
                    }
                    for bg, s in stats_by_bg.items()
                },
                "top1_accuracy_z_score_all_cells": top1_z,
                "verification_9_cells": points,
                "frozen_thresholds": {"t_accept": t_accept, "t_new": t_new},
                "threshold_report_9_cells": report,
                "collision_bar_median_genuine_shape": shape_bar,
                "collision_background": COLLISION_BACKGROUND,
                "per_puzzle_collisions": collision_table,
                "overall_collision_rate": overall_collisions / max(overall_pairs, 1),
                "collision_sheets": sheet_paths,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {eval_path}")


if __name__ == "__main__":
    main()
