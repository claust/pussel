#!/usr/bin/env python3
"""Evaluate an exp27 checkpoint on the north_star v1 real-photo benchmark.

Runs the exact exp25 protocol (same segmented-crop cache, overview auto-crop,
clockwise applied-rotation convention, row-major grid binning, metrics) on a
trained ``FrozenFeatureModel``. One plain forward per sample: the model's
rotation head already scores all 4 candidate rotations internally (by
re-encoding the rotated piece images), so no external rotation search is
needed — this is the matched protocol to exp25's "cnn" method.

NORTH-STAR DISCIPLINE: this touches the real-photo test set. Run it ONCE per
trained checkpoint (the val-selected one), not per idea.

Run from the network/ directory:
    uv run python -m experiments.exp27_pretrained_features.north_star_eval \
        --dataset-root datasets/north_star/v1 \
        --checkpoint experiments/exp27_pretrained_features/outputs/frozen_features/checkpoint_best_state_dict.pt
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from PIL import Image

from ..exp25_north_star_eval.evaluate import (
    BACKGROUNDS,
    build_piece_cache,
    cache_name,
    cell_index,
    compute_metrics,
    crop_overview,
    load_metadata,
    rotate_cw,
    rotation_confusion,
)
from .model import DEFAULT_ENCODER, FrozenFeatureModel, PositionHeadType

METHOD = "exp27"

# exp25/exp27 north_star v1 reference results (both-correct).
REFERENCE_BOTH = {"sift_else_ncc": 0.767, "ncc": 0.489, "dino_dense_zero_shot": 0.492, "cnn_exp20": 0.148}


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_inference(
    model: FrozenFeatureModel,
    samples: list[dict[str, Any]],
    records_by_puzzle: dict[str, list[dict[str, Any]]],
    puzzle_tensors: dict[str, torch.Tensor],
    cache_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Run batched model inference and record predictions on the samples in place.

    Args:
        model: Loaded exp27 model in eval mode.
        samples: Per-sample dicts (updated in place with preds/times).
        records_by_puzzle: Metadata records grouped by puzzle.
        puzzle_tensors: Prepared overview tensor per puzzle.
        cache_dir: Segmented piece crop cache.
        device: Device to run on.
        args: Parsed CLI namespace (batch/piece sizes).
    """
    crop_cache: dict[str, np.ndarray] = {}
    start = time.time()
    with torch.no_grad():
        for batch_start in range(0, len(samples), args.batch_size):
            batch = samples[batch_start : batch_start + args.batch_size]
            piece_inputs = []
            puzzle_inputs = []
            for s in batch:
                if s["piece_file"] not in crop_cache:
                    rec = next(r for r in records_by_puzzle[s["puzzle_id"]] if r["piece_file"] == s["piece_file"])
                    with Image.open(Path(cache_dir) / cache_name(rec)) as img:
                        crop = np.array(img.convert("RGB"))
                    crop_cache[s["piece_file"]] = cv2.resize(
                        crop, (args.piece_size, args.piece_size), interpolation=cv2.INTER_AREA
                    )
                observed = rotate_cw(crop_cache[s["piece_file"]], s["applied_idx"])
                piece_inputs.append(torch.from_numpy(observed.copy()).permute(2, 0, 1).float() / 255.0)
                puzzle_inputs.append(puzzle_tensors[s["puzzle_id"]])
            pieces = torch.stack(piece_inputs).to(device)
            puzzles = torch.stack(puzzle_inputs).to(device)
            positions, rotation_logits, _ = model(pieces, puzzles)
            positions_np = positions.cpu().float().numpy()
            rotations_np = rotation_logits.argmax(dim=1).cpu().numpy()

            for s, pos, rot in zip(batch, positions_np, rotations_np):
                pred_cell = cell_index(float(pos[0]), float(pos[1]), s["rows"], s["cols"])
                s["preds"][METHOD] = (pred_cell, int(rot))
            if (batch_start // args.batch_size + 1) % 20 == 0:
                print(f"  [{batch_start + len(batch)}/{len(samples)}] {time.time() - start:.0f}s", flush=True)
    per_sample = (time.time() - start) / len(samples)
    for s in samples:
        s["times"][METHOD] = per_sample


def main() -> None:
    """Run the exp27 north-star evaluation and write results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Raw state_dict checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Segmented crop cache (default: sibling of v1)")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER)
    parser.add_argument("--position-head", choices=("dense", "pooled"), default="dense")
    parser.add_argument("--adapter-dim", type=int, default=256)
    parser.add_argument("--piece-size", type=int, default=224)
    parser.add_argument("--puzzle-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=16, help="Samples per forward (each = 5 encoder passes)")
    parser.add_argument("--limit-puzzles", type=int, default=0, help="Evaluate only the first N puzzles (0 = all)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "north_star_results.json")
    args = parser.parse_args()
    cache_dir = args.cache_dir or args.dataset_root.parent / "v1_eval_cache"

    records = load_metadata(args.dataset_root)
    records_by_puzzle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        records_by_puzzle[rec["puzzle_id"]].append(rec)
    puzzle_ids = sorted(records_by_puzzle)
    if args.limit_puzzles:
        puzzle_ids = puzzle_ids[: args.limit_puzzles]
        records = [r for r in records if r["puzzle_id"] in set(puzzle_ids)]
        records_by_puzzle = {pid: records_by_puzzle[pid] for pid in puzzle_ids}
    print(f"Puzzles: {len(puzzle_ids)} | piece photos: {len(records)} | samples: {len(records) * 4}")

    print("Building/loading segmented piece cache (exp25 pipeline)...")
    build_piece_cache(records, args.dataset_root, cache_dir)

    device = get_device()
    print(f"Loading exp27 model ({args.encoder}, {args.position_head} head) on {device}...")
    model = FrozenFeatureModel(
        encoder_name=args.encoder,
        adapter_dim=args.adapter_dim,
        position_head=cast(PositionHeadType, args.position_head),
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Cropped, squared overview tensor per puzzle (mirrors exp25's CNN prep,
    # at exp27's input size).
    puzzle_tensors: dict[str, torch.Tensor] = {}
    for puzzle_id in puzzle_ids:
        with Image.open(args.dataset_root / puzzle_id / "overview.jpg") as img:
            overview = np.array(img.convert("RGB"))
        x1, y1, x2, y2 = crop_overview(overview)
        resized = cv2.resize(overview[y1:y2, x1:x2], (args.puzzle_size, args.puzzle_size), interpolation=cv2.INTER_AREA)
        puzzle_tensors[puzzle_id] = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

    samples: list[dict[str, Any]] = []
    for puzzle_id in puzzle_ids:
        for rec in records_by_puzzle[puzzle_id]:
            for applied_idx in range(4):
                samples.append(
                    {
                        "puzzle_id": puzzle_id,
                        "piece_file": rec["piece_file"],
                        "background": rec["background"],
                        "rows": rec["rows"],
                        "cols": rec["cols"],
                        "applied_idx": applied_idx,
                        "true_cell": rec["true_cell"],
                        "true_rotation_idx": (rec["base_rotation_idx"] + applied_idx) % 4,
                        "preds": {},
                        "times": {},
                    }
                )

    run_inference(model, samples, records_by_puzzle, puzzle_tensors, cache_dir, device, args)

    is_4x4 = {pid for pid in puzzle_ids if records_by_puzzle[pid][0]["rows"] == records_by_puzzle[pid][0]["cols"] == 4}
    result = {
        "overall": compute_metrics(samples, METHOD),
        "grid_4x4_subset": compute_metrics([s for s in samples if s["puzzle_id"] in is_4x4], METHOD),
        "per_background": {
            bg: compute_metrics([s for s in samples if s["background"] == bg], METHOD) for bg in BACKGROUNDS
        },
        "per_puzzle": {
            pid: compute_metrics([s for s in samples if s["puzzle_id"] == pid], METHOD) for pid in puzzle_ids
        },
        "rotation_confusion_true_x_pred": rotation_confusion(samples, METHOD),
        "runtime_ms_per_sample": sum(s["times"][METHOD] for s in samples) / len(samples) * 1000,
    }
    overall = result["overall"]
    print(
        f"\n{METHOD}: cell={overall['cell_accuracy']:.1%} rot={overall['rotation_accuracy']:.1%} "
        f"both={overall['both_accuracy']:.1%} ({result['runtime_ms_per_sample']:.0f} ms/sample)"
    )

    output = {
        "experiment": "exp27_pretrained_features north_star v1 evaluation",
        "checkpoint": str(args.checkpoint),
        "encoder": args.encoder,
        "position_head": args.position_head,
        "n_samples": len(samples),
        "protocol": {
            "piece_prep": "exp25 cache (bbox crop -> rembg -> largest component on black -> pad square)",
            "piece_size": args.piece_size,
            "puzzle_size": args.puzzle_size,
            "rotation": "single forward; the model re-encodes all 4 rotations internally",
        },
        "reference_both_accuracy": REFERENCE_BOTH,
        "results": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
