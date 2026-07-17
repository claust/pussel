#!/usr/bin/env python3
"""M4 evaluation: rank the true mating edge among all type-compatible edges.

Protocol, per puzzle x background (within one background only):
- Queries: every interior edge (grid position says it has a true neighbor)
  of every piece with a clean M3 record.
- Candidate pool: ALL type-compatible edges of all OTHER pieces in the same
  puzzle x background (not restricted to the mating compass direction).
- True mate: the neighboring piece's facing edge (E of (r,c) <-> W of
  (r,c+1), S <-> N of (r+1,c)). Queries whose neighbor record is missing are
  skipped and counted separately; a true mate excluded by the tab<->blank
  type gate counts as rank = pool size + 1.

Reports per metric (l2, chamfer, scalar6, l2_chord): top-1/3/5 %, median
rank, mean pool size - overall, per background, per puzzle. The main
configuration EXCLUDES records flagged corner_disagreement; a second pass
INCLUDES them to quantify the quality gate. Also reports genuine vs
impostor distance stats and the median best-impostor/true-mate margin for
the best metric, renders review sheets to `outputs/review_matching/`, and
writes everything to `outputs/matching_eval.json`.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/eval_matching.py
    uv run python experiments/exp28_piece_geometry/eval_matching.py --puzzle bambi --max-sheets 2
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
from edge_match import (  # noqa: E402
    COMPATIBLE_TYPES,
    canonicalize_edge,
    chord_penalty,
    dist_chamfer,
    dist_l2,
    dist_scalar6,
    edge_features,
    flip_edge,
    mirror_features,
)

DIRECTIONS = ("N", "E", "S", "W")
NEIGHBOR_OF = {"N": (-1, 0, "S"), "E": (0, 1, "W"), "S": (1, 0, "N"), "W": (0, -1, "E")}
METRICS = ("l2", "chamfer", "scalar6", "l2_chord")
MAX_QUERIES_PER_SHEET = 12


@dataclass
class EdgeEntry:
    """One edge of one piece record, with all match-ready representations.

    Attributes:
        puzzle_id: The puzzle this edge belongs to.
        background: The background of the photo.
        row: Piece grid row.
        col: Piece grid column.
        direction: Edge direction (N/E/S/W).
        edge_type: Predicted type (tab/blank/flat).
        chord_px: Chord length in original-image pixels.
        canonical: 100x2 canonical polyline.
        flipped: 100x2 mate-frame polyline (`flip_edge` of `canonical`).
        features: 6-feature vector.
        features_mirrored: Mirrored 6-feature vector.
        disagreement: The piece's corner_disagreement flag.
    """

    puzzle_id: str
    background: str
    row: int
    col: int
    direction: str
    edge_type: str
    chord_px: float
    canonical: np.ndarray
    flipped: np.ndarray
    features: np.ndarray
    features_mirrored: np.ndarray
    disagreement: bool


@dataclass
class QueryResult:
    """Outcome of ranking one query edge's true mate.

    Attributes:
        query: The query edge.
        mate: The true mate edge.
        pool_size: Number of candidates scored.
        ranks: Per-metric rank of the true mate (1-based; pool_size + 1 when
            the mate was gated out of the pool).
        mate_distances: Per-metric distance to the true mate (None when gated out).
        best_impostor_distances: Per-metric best (lowest) impostor distance.
        best_impostors: Per-metric best-scoring impostor entry.
    """

    query: EdgeEntry
    mate: EdgeEntry
    pool_size: int
    ranks: Dict[str, int]
    mate_distances: Dict[str, Optional[float]]
    best_impostor_distances: Dict[str, Optional[float]]
    best_impostors: Dict[str, Optional[EdgeEntry]]


def load_edge_entries(records_dir: Path, puzzle: Optional[str]) -> Tuple[List[EdgeEntry], Dict[str, Any]]:
    """Load M3 piece records and precompute match representations for every edge.

    Args:
        records_dir: The `outputs/piece_records` directory.
        puzzle: Optional substring filter on puzzle_id.

    Returns:
        Tuple of (all edge entries, {puzzle_id: (rows, cols)}).
    """
    entries: List[EdgeEntry] = []
    grid_dims: Dict[str, Any] = {}
    for path in sorted(records_dir.glob("*/*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        if puzzle and puzzle not in record["puzzle_id"]:
            continue
        grid_dims[record["puzzle_id"]] = (record["rows"], record["cols"])
        for direction in DIRECTIONS:
            edge = record["edges"][direction]
            polyline = np.array(edge["polyline"], dtype=np.float64)
            canonical = canonicalize_edge(polyline)
            features = edge_features(canonical, edge["chord_length_px"])
            entries.append(
                EdgeEntry(
                    puzzle_id=record["puzzle_id"],
                    background=record["background"],
                    row=record["row"],
                    col=record["col"],
                    direction=direction,
                    edge_type=edge["type"],
                    chord_px=edge["chord_length_px"],
                    canonical=canonical,
                    flipped=flip_edge(canonical),
                    features=features,
                    features_mirrored=mirror_features(features),
                    disagreement=record["corner_disagreement"],
                )
            )
    return entries, grid_dims


def _distance(metric: str, query: EdgeEntry, candidate: EdgeEntry) -> float:
    """Compute one metric's distance from a query edge to a candidate edge.

    Args:
        metric: One of `METRICS`.
        query: The query edge.
        candidate: The candidate edge (its flipped/mirrored forms are used).

    Returns:
        The distance.
    """
    if metric == "l2":
        return dist_l2(query.canonical, candidate.flipped)
    if metric == "chamfer":
        return dist_chamfer(query.canonical, candidate.flipped)
    if metric == "scalar6":
        return dist_scalar6(query.features, candidate.features_mirrored)
    if metric == "l2_chord":
        return dist_l2(query.canonical, candidate.flipped) + chord_penalty(query.chord_px, candidate.chord_px)
    raise ValueError(f"Unknown metric: {metric}")


def _is_interior(direction: str, row: int, col: int, rows: int, cols: int) -> bool:
    """Whether an edge has a true neighbor per the grid position.

    Args:
        direction: Edge direction.
        row: Piece row.
        col: Piece col.
        rows: Puzzle row count.
        cols: Puzzle col count.

    Returns:
        True when the edge is interior.
    """
    dr, dc, _ = NEIGHBOR_OF[direction]
    return 0 <= row + dr < rows and 0 <= col + dc < cols


def run_queries(entries: List[EdgeEntry], grid_dims: Dict[str, Any]) -> Tuple[List[QueryResult], int, int]:
    """Rank every interior query edge's true mate within its puzzle x background pool.

    Args:
        entries: All edge entries for this configuration.
        grid_dims: {puzzle_id: (rows, cols)}.

    Returns:
        Tuple of (query results, queries skipped because the neighbor record
        is missing, queries whose mate was type-gated out of the pool).
    """
    by_group: Dict[Tuple[str, str], Dict[Tuple[int, int, str], EdgeEntry]] = defaultdict(dict)
    for entry in entries:
        by_group[(entry.puzzle_id, entry.background)][(entry.row, entry.col, entry.direction)] = entry

    results: List[QueryResult] = []
    n_skipped_missing = 0
    n_gated_out = 0

    for (puzzle_id, _background), group in sorted(by_group.items()):
        rows, cols = grid_dims[puzzle_id]
        for (row, col, direction), query in sorted(group.items()):
            if not _is_interior(direction, row, col, rows, cols):
                continue
            dr, dc, mate_dir = NEIGHBOR_OF[direction]
            mate = group.get((row + dr, col + dc, mate_dir))
            if mate is None:
                n_skipped_missing += 1
                continue

            compatible = COMPATIBLE_TYPES[query.edge_type]
            pool = [
                entry
                for entry in group.values()
                if (entry.row, entry.col) != (row, col) and entry.edge_type in compatible
            ]
            mate_in_pool = any(entry is mate for entry in pool)
            if not mate_in_pool:
                n_gated_out += 1

            ranks: Dict[str, int] = {}
            mate_distances: Dict[str, Optional[float]] = {}
            best_impostor_distances: Dict[str, Optional[float]] = {}
            best_impostors: Dict[str, Optional[EdgeEntry]] = {}
            for metric in METRICS:
                distances = [(_distance(metric, query, entry), entry) for entry in pool]
                impostors = [(d, e) for d, e in distances if e is not mate]
                best_imp = min(impostors, key=lambda pair: pair[0]) if impostors else None
                best_impostor_distances[metric] = best_imp[0] if best_imp else None
                best_impostors[metric] = best_imp[1] if best_imp else None
                if mate_in_pool:
                    d_mate = next(d for d, e in distances if e is mate)
                    mate_distances[metric] = d_mate
                    ranks[metric] = 1 + sum(1 for d, e in distances if e is not mate and d < d_mate)
                else:
                    mate_distances[metric] = None
                    ranks[metric] = len(pool) + 1

            results.append(
                QueryResult(
                    query=query,
                    mate=mate,
                    pool_size=len(pool),
                    ranks=ranks,
                    mate_distances=mate_distances,
                    best_impostor_distances=best_impostor_distances,
                    best_impostors=best_impostors,
                )
            )

    return results, n_skipped_missing, n_gated_out


def summarize(results: List[QueryResult], key_fn: Callable[[QueryResult], str]) -> Dict[str, Dict[str, Any]]:
    """Aggregate rank statistics per metric, grouped by a scope key.

    Args:
        results: Query results.
        key_fn: Maps a result to its scope name ("overall", background, puzzle...).

    Returns:
        {scope: {metric: {top1, top3, top5, median_rank, n}, pool: mean pool size}}.
    """
    by_scope: Dict[str, List[QueryResult]] = defaultdict(list)
    for result in results:
        by_scope[key_fn(result)].append(result)

    out: Dict[str, Dict[str, Any]] = {}
    for scope, scoped in by_scope.items():
        scope_stats: Dict[str, Any] = {
            "n_queries": len(scoped),
            "mean_pool": float(np.mean([r.pool_size for r in scoped])),
        }
        for metric in METRICS:
            ranks = np.array([r.ranks[metric] for r in scoped])
            scope_stats[metric] = {
                "top1": float(np.mean(ranks <= 1)),
                "top3": float(np.mean(ranks <= 3)),
                "top5": float(np.mean(ranks <= 5)),
                "median_rank": float(np.median(ranks)),
            }
        out[scope] = scope_stats
    return out


def genuine_impostor_stats(results: List[QueryResult], metric: str) -> Dict[str, Any]:
    """Genuine vs impostor distance distributions for one metric.

    Args:
        results: Query results.
        metric: The metric to analyze.

    Returns:
        Dict with genuine/impostor medians, the impostor 5th percentile,
        overlap (% genuine above that percentile), and the median
        best-impostor/true-mate distance margin.
    """
    genuine = np.array([r.mate_distances[metric] for r in results if r.mate_distances[metric] is not None])
    impostor_best = np.array(
        [r.best_impostor_distances[metric] for r in results if r.best_impostor_distances[metric] is not None]
    )
    margins = np.array(
        [
            r.best_impostor_distances[metric] / r.mate_distances[metric]
            for r in results
            if r.mate_distances[metric] and r.best_impostor_distances[metric] is not None
        ]
    )
    impostor_p5 = float(np.percentile(impostor_best, 5)) if len(impostor_best) else float("nan")
    overlap = float(np.mean(genuine > impostor_p5)) if len(genuine) else float("nan")
    return {
        "median_genuine": float(np.median(genuine)) if len(genuine) else None,
        "median_best_impostor": float(np.median(impostor_best)) if len(impostor_best) else None,
        "impostor_p5": impostor_p5,
        "overlap_frac": overlap,
        "median_margin": float(np.median(margins)) if len(margins) else None,
    }


def render_sheets(results: List[QueryResult], metric: str, output_dir: Path, max_sheets: Optional[int]) -> None:
    """Render review sheets: query vs flipped true mate vs flipped best impostor.

    Args:
        results: Query results (main configuration).
        metric: The metric whose best impostor is drawn.
        output_dir: The experiment outputs directory.
        max_sheets: Optional cap on sheets rendered.
    """
    review_dir = output_dir / "review_matching"
    review_dir.mkdir(parents=True, exist_ok=True)

    by_group: Dict[Tuple[str, str], List[QueryResult]] = defaultdict(list)
    for result in results:
        by_group[(result.query.puzzle_id, result.query.background)].append(result)

    n_sheets = 0
    for (puzzle_id, background), group in sorted(by_group.items()):
        if max_sheets is not None and n_sheets >= max_sheets:
            break
        shown = group[:MAX_QUERIES_PER_SHEET]
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        fig.suptitle(f"{puzzle_id} / {background} - metric {metric}")
        for ax in axes.flat:
            ax.axis("off")
        for ax, result in zip(axes.flat, shown):
            query = result.query
            ax.plot(query.canonical[:, 0], query.canonical[:, 1], "b-", linewidth=2, label="query")
            ax.plot(result.mate.flipped[:, 0], result.mate.flipped[:, 1], "g-", linewidth=1.5, label="true mate")
            impostor = result.best_impostors[metric]
            if impostor is not None:
                ax.plot(impostor.flipped[:, 0], impostor.flipped[:, 1], "r--", linewidth=1.2, label="best impostor")
            d_mate = result.mate_distances[metric]
            d_imp = result.best_impostor_distances[metric]
            title = (
                f"r{query.row}c{query.col} {query.direction} rank={result.ranks[metric]}\n"
                f"mate={d_mate:.3f} imp={d_imp:.3f}"
                if d_mate is not None and d_imp is not None
                else f"r{query.row}c{query.col} {query.direction} rank={result.ranks[metric]}"
            )
            ax.set_title(title, fontsize=8)
            ax.set_aspect("equal")
            ax.invert_yaxis()  # image convention: tabs (outward) point up visually
            ax.axis("on")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        out_path = review_dir / f"{puzzle_id}_{background}.png"
        fig.savefig(out_path, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path} ({len(shown)} queries)")
        n_sheets += 1


def _print_metric_table(title: str, stats: Dict[str, Dict[str, Any]], scopes: List[str]) -> None:
    """Print a per-scope x per-metric rank table.

    Args:
        title: Table heading.
        stats: Output of `summarize`.
        scopes: Scope names to print, in order.
    """
    print(f"\n{title}")
    print(f"{'scope':<26}{'metric':<10}{'top1':>7}{'top3':>7}{'top5':>7}{'med.rank':>10}{'pool':>7}{'n':>6}")
    for scope in scopes:
        if scope not in stats:
            continue
        scope_stats = stats[scope]
        for metric in METRICS:
            m = scope_stats[metric]
            print(
                f"{scope:<26}{metric:<10}{m['top1'] * 100:6.1f}%{m['top3'] * 100:6.1f}%{m['top5'] * 100:6.1f}%"
                f"{m['median_rank']:10.1f}{scope_stats['mean_pool']:7.1f}{scope_stats['n_queries']:6d}"
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate edge-mate ranking within puzzle x background pools.")
    parser.add_argument("--records-dir", type=Path, default=Path(__file__).parent / "outputs" / "piece_records")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    parser.add_argument("--max-sheets", type=int, default=None)
    parser.add_argument("--no-sheets", action="store_true", help="Skip rendering review sheets")
    args = parser.parse_args()

    entries, grid_dims = load_edge_entries(args.records_dir, args.puzzle)
    if not entries:
        print("No piece records found - run edge_split.py first.")
        return

    clean_entries = [e for e in entries if not e.disagreement]
    print(f"Loaded {len(entries)} edges ({len(clean_entries)} from records without corner_disagreement)")

    results, n_missing, n_gated = run_queries(clean_entries, grid_dims)
    print(
        f"Main config (exclude disagreement): {len(results)} queries, "
        f"{n_missing} skipped (neighbor record missing), {n_gated} mates type-gated out"
    )

    overall = summarize(results, lambda r: "overall")
    per_bg = summarize(results, lambda r: r.query.background)
    per_puzzle = summarize(results, lambda r: f"{r.query.puzzle_id}|{r.query.background}")

    _print_metric_table("=== Overall (exclude corner_disagreement) ===", overall, ["overall"])
    _print_metric_table("=== Per background ===", per_bg, sorted(per_bg))

    best_metric = max(METRICS, key=lambda m: overall["overall"][m]["top1"])
    print(f"\nBest metric by overall top-1: {best_metric}")

    print(f"\n=== Per puzzle x background ({best_metric}) ===")
    print(f"{'puzzle|background':<44}{'top1':>7}{'top3':>7}{'med.rank':>10}{'pool':>7}{'n':>6}")
    for scope in sorted(per_puzzle):
        s = per_puzzle[scope]
        m = s[best_metric]
        print(
            f"{scope:<44}{m['top1'] * 100:6.1f}%{m['top3'] * 100:6.1f}%{m['median_rank']:10.1f}"
            f"{s['mean_pool']:7.1f}{s['n_queries']:6d}"
        )

    gi_stats = genuine_impostor_stats(results, best_metric)
    print(
        f"\nGenuine vs impostor ({best_metric}): median genuine={gi_stats['median_genuine']:.4f}, "
        f"median best-impostor={gi_stats['median_best_impostor']:.4f}, impostor p5={gi_stats['impostor_p5']:.4f}, "
        f"overlap={gi_stats['overlap_frac'] * 100:.1f}% of genuine above impostor p5, "
        f"median margin (best impostor / true mate) = {gi_stats['median_margin']:.2f}x"
    )

    results_incl, n_missing_i, n_gated_i = run_queries(entries, grid_dims)
    overall_incl = summarize(results_incl, lambda r: "overall")
    print("\n=== Disagreement-gate delta (overall top-1) ===")
    for metric in METRICS:
        excl = overall["overall"][metric]["top1"] * 100
        incl = overall_incl["overall"][metric]["top1"] * 100
        print(
            f"  {metric:<10} exclude: {excl:5.1f}% ({overall['overall']['n_queries']} q)   "
            f"include: {incl:5.1f}% ({overall_incl['overall']['n_queries']} q)   delta: {incl - excl:+.1f}"
        )

    if not args.no_sheets:
        render_sheets(results, best_metric, args.output_dir, args.max_sheets)

    eval_path = args.output_dir / "matching_eval.json"
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_exclude_disagreement": {
                    "n_queries": len(results),
                    "skipped_missing_neighbor": n_missing,
                    "mates_type_gated_out": n_gated,
                    "overall": overall,
                    "per_background": per_bg,
                    "per_puzzle_background": per_puzzle,
                },
                "config_include_disagreement": {
                    "n_queries": len(results_incl),
                    "skipped_missing_neighbor": n_missing_i,
                    "mates_type_gated_out": n_gated_i,
                    "overall": overall_incl,
                    "per_background": summarize(results_incl, lambda r: r.query.background),
                },
                "best_metric": best_metric,
                "genuine_impostor": gi_stats,
            },
            handle,
            indent=2,
        )
    print(f"\nWrote {eval_path}")


if __name__ == "__main__":
    main()
