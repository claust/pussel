#!/usr/bin/env python3
"""Exp27 stage 0: zero-shot DINOv2 feature probe on the north_star v1 benchmark.

Measures what a frozen, self-supervised ViT (DINOv2-S/14) is worth on the real
photographed-piece task with NO training at all, before committing to the full
exp27 training run. Reuses the exp25 protocol verbatim: same segmented-crop
cache (bbox crop -> rembg -> largest component on black -> pad square), same
overview auto-crop, same clockwise applied-rotation convention, same row-major
grid binning, same metrics.

Two zero-shot methods, both evaluated by encoding all 4 rotated piece images
(ViT features are not rotation-equivariant, so rotating a feature map is not a
substitute for encoding the rotated image):

- dino_mean: piece descriptor = foreground-masked mean patch token; cosine map
  against the puzzle patch-token grid; cell-sized window smoothing; peak gives
  position, best-scoring rotation gives rotation.
- dino_dense: the piece's full patch-token grid, foreground-weighted, resized
  to a cell-sized window and cross-correlated against the puzzle token grid
  via convolution (masked NCC in feature space), over a small scale sweep.

Rotation bookkeeping: the observed piece for applied rotation ``a`` is
``rotate_cw(crop, a)`` and candidate ``c`` un-rotates it counter-clockwise by
``c`` (exp25 convention), so the candidate image equals the base crop rotated
clockwise by ``(a - c) % 4``. Rotations here are lossless np.rot90 calls, so we
encode the 4 clockwise rotations of the base crop once, pick the best net
rotation ``r*``, and read out the prediction for applied ``a`` as
``c* = (a - r*) % 4``.

Run from the network/ directory:
    uv run python -m experiments.exp27_pretrained_features.zero_shot_probe \
        --dataset-root datasets/north_star/v1
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm  # type: ignore[import-untyped]
import torch
import torch.nn.functional as F
from PIL import Image

from ..exp25_north_star_eval.evaluate import (
    BACKGROUNDS,
    MASK_THRESHOLD,
    build_piece_cache,
    cache_name,
    cell_index,
    compute_metrics,
    crop_overview,
    load_metadata,
    rotate_cw,
    rotation_confusion,
)

METHOD_NAMES = ("dino_mean", "dino_dense")

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
PATCH_SIZE = 14
PIECE_SIDE = 224  # 16x16 patch tokens
PUZZLE_SIDE = 448  # 32x32 patch tokens

# Real pieces with tabs span ~1.0-1.5 cells (exp25 NCC needed a scale sweep).
DENSE_SCALES = (1.0, 1.25, 1.5)

# exp25 north_star v1 reference results (both-correct), for the results JSON.
EXP25_REFERENCE_BOTH = {"sift_else_ncc": 0.767, "ncc": 0.489, "cnn_rotsearch": 0.180, "cnn": 0.148}


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DinoEncoder:
    """Frozen DINOv2 encoder that returns L2-normalized patch-token grids."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        """Load the pretrained model and its normalization constants.

        Args:
            model_name: timm model name (a ViT with 14px patches).
            device: Device to run on.
        """
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, dynamic_img_size=True)
        self.model.eval().to(device)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))

    @torch.inference_mode()
    def encode(self, images: list[np.ndarray]) -> torch.Tensor:
        """Encode a batch of same-sized RGB uint8 images into patch-token grids.

        Args:
            images: RGB uint8 arrays, all HxWx3 with H and W multiples of 14.

        Returns:
            Tensor of shape (N, H/14, W/14, C) with unit-norm tokens.
        """
        side_h, side_w = images[0].shape[:2]
        batch = torch.from_numpy(np.stack(images)).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float() / 255.0
        batch = (batch - self.mean) / self.std
        feats = self.model.forward_features(batch)
        tokens = feats[:, self.num_prefix_tokens :, :]
        grid_h, grid_w = side_h // PATCH_SIZE, side_w // PATCH_SIZE
        tokens = tokens.reshape(len(images), grid_h, grid_w, -1)
        return F.normalize(tokens, dim=-1)


def foreground_weights(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Per-patch foreground weight for a black-backed piece crop.

    Args:
        rgb: RGB uint8 array with side lengths that are multiples of 14.
        device: Device for the returned tensor.

    Returns:
        Tensor of shape (H/14, W/14) with each patch's foreground fraction.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = torch.from_numpy((gray > MASK_THRESHOLD).astype(np.float32)).to(device)
    return F.avg_pool2d(mask.unsqueeze(0).unsqueeze(0), PATCH_SIZE).squeeze(0).squeeze(0)


def window_peak(score_map: torch.Tensor, kh: int, kw: int, grid: int) -> tuple[float, float, float]:
    """Find the best cell-sized window in a similarity map by average pooling.

    Args:
        score_map: 2D similarity map over puzzle patch positions.
        kh: Window height in patches.
        kw: Window width in patches.
        grid: Side length of the (square) puzzle patch grid.

    Returns:
        (score, nx, ny): best window's mean score and its center normalized to [0, 1].
    """
    pooled = F.avg_pool2d(score_map.unsqueeze(0).unsqueeze(0), (kh, kw), stride=1).squeeze(0).squeeze(0)
    idx = int(pooled.argmax().item())
    i, j = divmod(idx, pooled.shape[1])
    return float(pooled.flatten()[idx].item()), (j + kw / 2) / grid, (i + kh / 2) / grid


def dense_peak(
    puzzle_chw: torch.Tensor, tokens: torch.Tensor, weights: torch.Tensor, kh: int, kw: int, grid: int
) -> tuple[float, float, float]:
    """Cross-correlate a weighted piece token grid against the puzzle grid over scales.

    Masked NCC in feature space: at each puzzle window, the score is the
    foreground-weighted mean cosine similarity between the (resized) piece
    tokens and the puzzle tokens.

    Args:
        puzzle_chw: Puzzle patch tokens as (1, C, grid, grid), unit-norm per location.
        tokens: Piece patch tokens as (h, w, C), unit-norm per location.
        weights: Piece foreground weights as (h, w).
        kh: Nominal cell height in puzzle patches.
        kw: Nominal cell width in puzzle patches.
        grid: Side length of the puzzle patch grid.

    Returns:
        (score, nx, ny): best window's score and its center normalized to [0, 1].
    """
    piece_chw = tokens.permute(2, 0, 1).unsqueeze(0)
    weight_map = weights.unsqueeze(0).unsqueeze(0)
    best = (-float("inf"), 0.5, 0.5)
    for scale in DENSE_SCALES:
        sh = min(max(int(round(kh * scale)), 2), grid)
        sw = min(max(int(round(kw * scale)), 2), grid)
        kernel = F.interpolate(piece_chw, size=(sh, sw), mode="bilinear", align_corners=False)
        kernel = F.normalize(kernel, dim=1)
        w_k = F.interpolate(weight_map, size=(sh, sw), mode="bilinear", align_corners=False)
        total = float(w_k.sum().item())
        if total < 1e-6:
            continue
        response = F.conv2d(puzzle_chw, kernel * w_k) / total
        response_2d = response.squeeze(0).squeeze(0)
        idx = int(response_2d.argmax().item())
        i, j = divmod(idx, response_2d.shape[1])
        score = float(response_2d.flatten()[idx].item())
        if score > best[0]:
            best = (score, (j + sw / 2) / grid, (i + sh / 2) / grid)
    return best


def probe_piece(
    encoder: DinoEncoder,
    crop: np.ndarray,
    puzzle_tokens: torch.Tensor,
    rows: int,
    cols: int,
) -> dict[str, list[tuple[float, float, float]]]:
    """Score the 4 clockwise rotations of one base piece crop against its puzzle.

    Args:
        encoder: Frozen encoder.
        crop: Segmented square piece crop (RGB uint8, black background).
        puzzle_tokens: Puzzle patch tokens as (grid, grid, C), unit-norm.
        rows: Grid rows for this puzzle.
        cols: Grid cols for this puzzle.

    Returns:
        For each method, a list over net clockwise rotation r=0..3 of
        (score, nx, ny).
    """
    grid = puzzle_tokens.shape[0]
    kh = min(max(int(round(grid / rows)), 2), grid)
    kw = min(max(int(round(grid / cols)), 2), grid)
    puzzle_chw = puzzle_tokens.permute(2, 0, 1).unsqueeze(0)

    rotated = [cv2.resize(rotate_cw(crop, r), (PIECE_SIDE, PIECE_SIDE), interpolation=cv2.INTER_AREA) for r in range(4)]
    tokens4 = encoder.encode(rotated)
    weights4 = [foreground_weights(img, encoder.device) for img in rotated]

    out: dict[str, list[tuple[float, float, float]]] = {"dino_mean": [], "dino_dense": []}
    for r in range(4):
        tokens, weights = tokens4[r], weights4[r]
        w_sum = weights.sum()
        if float(w_sum.item()) < 1e-6:
            out["dino_mean"].append((-float("inf"), 0.5, 0.5))
            out["dino_dense"].append((-float("inf"), 0.5, 0.5))
            continue
        desc = F.normalize((tokens * weights.unsqueeze(-1)).sum(dim=(0, 1)) / w_sum, dim=0)
        sim_map = torch.einsum("hwc,c->hw", puzzle_tokens, desc)
        out["dino_mean"].append(window_peak(sim_map, kh, kw, grid))
        out["dino_dense"].append(dense_peak(puzzle_chw, tokens, weights, kh, kw, grid))
    return out


def main() -> None:
    """Run the zero-shot probe and write results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Segmented crop cache (default: sibling of v1)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--limit-puzzles", type=int, default=0, help="Probe only the first N puzzles (0 = all)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "zero_shot_results.json")
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
    print(f"Loading {args.model_name} on {device}...")
    encoder = DinoEncoder(args.model_name, device)

    samples: list[dict[str, Any]] = []
    start = time.time()
    for p_idx, puzzle_id in enumerate(puzzle_ids):
        with Image.open(args.dataset_root / puzzle_id / "overview.jpg") as img:
            overview = np.array(img.convert("RGB"))
        x1, y1, x2, y2 = crop_overview(overview)
        overview = cv2.resize(overview[y1:y2, x1:x2], (PUZZLE_SIDE, PUZZLE_SIDE), interpolation=cv2.INTER_AREA)
        puzzle_tokens = encoder.encode([overview])[0]

        for rec in records_by_puzzle[puzzle_id]:
            with Image.open(Path(cache_dir) / cache_name(rec)) as img:
                crop = np.array(img.convert("RGB"))
            rows, cols = rec["rows"], rec["cols"]
            t0 = time.perf_counter()
            scored = probe_piece(encoder, crop, puzzle_tokens, rows, cols)
            elapsed = time.perf_counter() - t0

            # Best net clockwise rotation per method (independent of the
            # applied rotation because np.rot90 is lossless; see module doc).
            best_r = {m: int(np.argmax([s[0] for s in scored[m]])) for m in METHOD_NAMES}
            for applied_idx in range(4):
                preds = {}
                for m in METHOD_NAMES:
                    score, nx, ny = scored[m][best_r[m]]
                    pred_rot = (applied_idx - best_r[m]) % 4
                    preds[m] = (cell_index(nx, ny, rows, cols), pred_rot) if np.isfinite(score) else None
                samples.append(
                    {
                        "puzzle_id": puzzle_id,
                        "piece_file": rec["piece_file"],
                        "background": rec["background"],
                        "rows": rows,
                        "cols": cols,
                        "applied_idx": applied_idx,
                        "true_cell": rec["true_cell"],
                        "true_rotation_idx": (rec["base_rotation_idx"] + applied_idx) % 4,
                        "preds": preds,
                        "times": {m: elapsed / 4 for m in METHOD_NAMES},
                    }
                )
        print(f"  [{p_idx + 1}/{len(puzzle_ids)}] {puzzle_id} {time.time() - start:.0f}s", flush=True)

    is_4x4 = {pid for pid in puzzle_ids if records_by_puzzle[pid][0]["rows"] == records_by_puzzle[pid][0]["cols"] == 4}
    methods_results: dict[str, Any] = {}
    for method in METHOD_NAMES:
        methods_results[method] = {
            "overall": compute_metrics(samples, method),
            "grid_4x4_subset": compute_metrics([s for s in samples if s["puzzle_id"] in is_4x4], method),
            "per_background": {
                bg: compute_metrics([s for s in samples if s["background"] == bg], method) for bg in BACKGROUNDS
            },
            "per_puzzle": {
                pid: compute_metrics([s for s in samples if s["puzzle_id"] == pid], method) for pid in puzzle_ids
            },
            "rotation_confusion_true_x_pred": rotation_confusion(samples, method),
            "runtime_ms_per_sample": sum(s["times"][method] for s in samples) / len(samples) * 1000,
        }
        overall = methods_results[method]["overall"]
        print(
            f"{method:>12s} cell={overall['cell_accuracy']:.1%} rot={overall['rotation_accuracy']:.1%} "
            f"both={overall['both_accuracy']:.1%} ({methods_results[method]['runtime_ms_per_sample']:.0f} ms/sample)"
        )

    output = {
        "probe": "exp27 stage 0: zero-shot frozen DINOv2 features, no training",
        "model": args.model_name,
        "n_samples": len(samples),
        "protocol": {
            "piece_prep": "exp25 cache (bbox crop -> rembg -> largest component on black -> pad square)",
            "piece_side": PIECE_SIDE,
            "puzzle_side": PUZZLE_SIDE,
            "dense_scales": list(DENSE_SCALES),
            "rotation": "encode 4 rotated piece images; best net rotation r*; pred for applied a = (a - r*) % 4",
            "cell_binning": "peak window center normalized to cropped overview, row-major rows x cols grid",
        },
        "exp25_reference_both_accuracy": EXP25_REFERENCE_BOTH,
        "methods": methods_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
