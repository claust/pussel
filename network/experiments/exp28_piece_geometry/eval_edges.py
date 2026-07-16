#!/usr/bin/env python3
"""M3 evaluation: score edge classifications against near-free grid ground truth.

Evaluates the `edge_split.py` piece records three ways:

1. **Flat accuracy** (absolute GT): an edge is truly flat iff it faces the
   puzzle border (row 0 -> N, last row -> S, col 0 -> W, last col -> E).
   Reports the flat-vs-nonflat confusion overall, per background, and per
   puzzle. This is the headline >=98% criterion.
2. **Cross-background consistency**: a physical piece's 4-edge type
   signature (N,E,S,W) must be identical across its clean records.
3. **Neighbor complementarity** (within each background): (r,c) East must
   complement (r,c+1) West (tab<->blank), and (r,c) South <-> (r+1,c) North.

Also renders per-puzzle x background review sheets (flat=yellow, tab=green,
blank=red arcs, white corner dots, red cell border on corner_disagreement)
to `outputs/review_edges/`, prints a final summary block, and writes the
same numbers to `outputs/edge_eval.json`.

Usage:
    cd network
    uv run python experiments/exp28_piece_geometry/eval_edges.py
    uv run python experiments/exp28_piece_geometry/eval_edges.py --puzzle bambi --max-sheets 2
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from common import PieceRecord, assemble_grid_sheet, crop_with_margin, load_metadata, make_titled_cell

DEFAULT_DATASET_ROOT = Path("/Users/claus/Repos/pussel/network/datasets/north_star/v1")
DIRECTIONS = ("N", "E", "S", "W")
EDGE_COLORS = {  # BGR
    "flat": (0, 255, 255),
    "tab": (0, 200, 0),
    "blank": (0, 0, 220),
}
CELL_WIDTH = 260
DISAGREEMENT_BORDER = (0, 0, 255)


def _gt_flat(record: Dict[str, Any], direction: str) -> bool:
    """Whether an edge is truly flat, from the piece's grid position.

    Args:
        record: A piece record (needs row/col/rows/cols).
        direction: One of N/E/S/W.

    Returns:
        True iff the edge faces the puzzle border.
    """
    if direction == "N":
        return record["row"] == 0
    if direction == "S":
        return record["row"] == record["rows"] - 1
    if direction == "W":
        return record["col"] == 0
    return record["col"] == record["cols"] - 1


def load_piece_records(records_dir: Path, puzzle: Optional[str]) -> List[Dict[str, Any]]:
    """Load all piece-record JSONs written by edge_split.py.

    Args:
        records_dir: The `outputs/piece_records` directory.
        puzzle: Optional substring filter on puzzle_id.

    Returns:
        The parsed records.
    """
    records: List[Dict[str, Any]] = []
    for path in sorted(records_dir.glob("*/*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        if puzzle and puzzle not in record["puzzle_id"]:
            continue
        records.append(record)
    return records


def flat_confusion(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute the flat-vs-nonflat confusion, overall / per background / per puzzle.

    Args:
        records: All piece records.

    Returns:
        Dict mapping scope name ("overall", "bg:<background>", "puzzle:<id>")
        to {tp, fn, fp, tn, accuracy}.
    """
    counters: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        for direction in DIRECTIONS:
            gt = _gt_flat(record, direction)
            pred = record["edges"][direction]["type"] == "flat"
            key = ("tp" if pred else "fn") if gt else ("fp" if pred else "tn")
            counters["overall"][key] += 1
            counters[f"bg:{record['background']}"][key] += 1
            counters[f"puzzle:{record['puzzle_id']}"][key] += 1

    out: Dict[str, Dict[str, Any]] = {}
    for scope, counts in counters.items():
        total = sum(counts.values())
        correct = counts["tp"] + counts["tn"]
        out[scope] = {**{k: counts[k] for k in ("tp", "fn", "fp", "tn")}, "accuracy": correct / total}
    return out


def cross_background_consistency(records: List[Dict[str, Any]]) -> Tuple[float, int, int, List[Tuple[str, int]]]:
    """Check that each physical piece's edge-type signature agrees across backgrounds.

    Args:
        records: All piece records.

    Returns:
        Tuple of (fraction of multi-record pieces fully consistent, number of
        consistent pieces, number of pieces compared, most common
        inconsistency patterns as (description, count)).
    """
    by_piece: Dict[Tuple[str, int, int], List[Tuple[str, ...]]] = defaultdict(list)
    for record in records:
        signature = tuple(record["edges"][d]["type"] for d in DIRECTIONS)
        by_piece[(record["puzzle_id"], record["row"], record["col"])].append(signature)

    n_compared = 0
    n_consistent = 0
    patterns: Counter = Counter()
    for signatures in by_piece.values():
        if len(signatures) < 2:
            continue
        n_compared += 1
        if len(set(signatures)) == 1:
            n_consistent += 1
            continue
        majority = Counter(signatures).most_common(1)[0][0]
        for sig in set(signatures):
            if sig == majority:
                continue
            for d, (a, b) in zip(DIRECTIONS, zip(majority, sig)):
                if a != b:
                    patterns[f"{d}: {a}->{b}"] += 1
    rate = n_consistent / n_compared if n_compared else 1.0
    return rate, n_consistent, n_compared, patterns.most_common(5)


COMPLEMENT = {("tab", "blank"), ("blank", "tab")}


def neighbor_complementarity(records: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """Check tab<->blank complementarity between adjacent pieces, per background.

    Args:
        records: All piece records.

    Returns:
        Dict mapping background to (complementary pairs, compared pairs).
    """
    by_key: Dict[Tuple[str, str], Dict[Tuple[int, int], Dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_key[(record["puzzle_id"], record["background"])][(record["row"], record["col"])] = record

    results: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for (_puzzle_id, background), grid in by_key.items():
        for (row, col), record in grid.items():
            east_neighbor = grid.get((row, col + 1))
            if east_neighbor is not None:
                pair = (record["edges"]["E"]["type"], east_neighbor["edges"]["W"]["type"])
                results[background][1] += 1
                if pair in COMPLEMENT:
                    results[background][0] += 1
            south_neighbor = grid.get((row + 1, col))
            if south_neighbor is not None:
                pair = (record["edges"]["S"]["type"], south_neighbor["edges"]["N"]["type"])
                results[background][1] += 1
                if pair in COMPLEMENT:
                    results[background][0] += 1
    return {bg: (v[0], v[1]) for bg, v in results.items()}


def render_sheets(
    dataset_root: Path,
    records: List[Dict[str, Any]],
    meta_by_file: Dict[str, PieceRecord],
    output_dir: Path,
    max_sheets: Optional[int],
) -> None:
    """Render per-puzzle x background edge-type review sheets.

    Args:
        dataset_root: The north_star dataset root.
        records: All piece records.
        meta_by_file: Metadata rows keyed by piece_file (for bbox/margin crop).
        output_dir: The experiment outputs directory.
        max_sheets: Optional cap on the number of sheets rendered.
    """
    review_dir = output_dir / "review_edges"
    review_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["puzzle_id"], record["background"])].append(record)

    n_sheets = 0
    for (puzzle_id, background), group in sorted(groups.items()):
        if max_sheets is not None and n_sheets >= max_sheets:
            break
        cells: Dict[Tuple[int, int], np.ndarray] = {}
        for record in group:
            meta = meta_by_file[record["piece_file"]]
            image = cv2.imread(str(dataset_root / record["piece_file"]))
            crop, offset = crop_with_margin(image, meta.bbox, margin_frac=0.15)
            offset_arr = np.array(offset, dtype=np.float64)

            for direction in DIRECTIONS:
                edge = record["edges"][direction]
                polyline = (np.array(edge["polyline"]) - offset_arr).astype(np.int32)
                cv2.polylines(crop, [polyline], isClosed=False, color=EDGE_COLORS[edge["type"]], thickness=3)
            for corner in np.array(record["corners"]) - offset_arr:
                cv2.circle(crop, (int(corner[0]), int(corner[1])), 6, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            if record["corner_disagreement"]:
                cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), DISAGREEMENT_BORDER, 6)

            label = f"r{record['row']:02d}c{record['col']:02d}"
            cells[(record["row"], record["col"])] = make_titled_cell(crop, label, CELL_WIDTH)

        rows = group[0]["rows"]
        cols = group[0]["cols"]
        sheet = assemble_grid_sheet(cells, rows, cols, CELL_WIDTH)
        out_path = review_dir / f"{puzzle_id}_{background}.png"
        cv2.imwrite(str(out_path), sheet)
        print(f"Wrote {out_path} ({sheet.shape[1]}x{sheet.shape[0]}, {len(group)} pieces)")
        n_sheets += 1


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate edge classifications against grid ground truth.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--records-dir", type=Path, default=Path(__file__).parent / "outputs" / "piece_records")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--puzzle", type=str, default=None, help="Substring filter on puzzle_id")
    parser.add_argument("--max-sheets", type=int, default=None)
    parser.add_argument("--no-sheets", action="store_true", help="Skip rendering review sheets")
    args = parser.parse_args()

    records = load_piece_records(args.records_dir, args.puzzle)
    if not records:
        print("No piece records found - run edge_split.py first.")
        return
    print(f"Loaded {len(records)} piece records ({4 * len(records)} edges)")

    metadata = load_metadata(args.dataset_root)
    meta_by_file = {m.piece_file: m for m in metadata}

    if not args.no_sheets:
        render_sheets(args.dataset_root, records, meta_by_file, args.output_dir, args.max_sheets)

    confusion = flat_confusion(records)
    consistency_rate, n_consistent, n_compared, patterns = cross_background_consistency(records)
    complementarity = neighbor_complementarity(records)

    print("\n================ M3 edge evaluation summary ================")
    overall = confusion["overall"]
    print(
        f"Flat accuracy (overall): {overall['accuracy'] * 100:.2f}%  "
        f"[tp={overall['tp']} fn={overall['fn']} fp={overall['fp']} tn={overall['tn']}]"
    )
    print("Flat accuracy per background:")
    for scope in sorted(s for s in confusion if s.startswith("bg:")):
        c = confusion[scope]
        print(f"  {scope[3:]:<12} {c['accuracy'] * 100:6.2f}%  [tp={c['tp']} fn={c['fn']} fp={c['fp']} tn={c['tn']}]")
    print("Flat accuracy per puzzle:")
    for scope in sorted(s for s in confusion if s.startswith("puzzle:")):
        c = confusion[scope]
        print(f"  {scope[7:]:<24} {c['accuracy'] * 100:6.2f}%  [fn={c['fn']} fp={c['fp']}]")
    print(
        f"Cross-background consistency: {consistency_rate * 100:.2f}% "
        f"({n_consistent}/{n_compared} pieces fully consistent)"
    )
    if patterns:
        print(f"  Most common inconsistencies (majority->minority): {patterns}")
    print("Neighbor complementarity per background:")
    comp_summary = {}
    for background, (good, total) in sorted(complementarity.items()):
        pct = 100.0 * good / total if total else 0.0
        comp_summary[background] = {"complementary": good, "pairs": total, "rate": good / total if total else None}
        print(f"  {background:<12} {pct:6.2f}%  ({good}/{total} adjacent pairs)")
    print("=============================================================")

    eval_path = args.output_dir / "edge_eval.json"
    with open(eval_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_records": len(records),
                "flat_confusion": confusion,
                "cross_background_consistency": {
                    "rate": consistency_rate,
                    "consistent": n_consistent,
                    "compared": n_compared,
                    "top_inconsistencies": patterns,
                },
                "neighbor_complementarity": comp_summary,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {eval_path}")


if __name__ == "__main__":
    main()
