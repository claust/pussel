#!/usr/bin/env python3
"""Classical (non-learned) baselines on the exp20 4x4 realistic-pieces benchmark.

Runs OpenCV normalized cross-correlation template matching (masked
TM_CCOEFF_NORMED over the 4 candidate rotations) and SIFT/ORB keypoint
matching (Lowe ratio test + partial-affine RANSAC) against the exact test
protocol used for the exp20 CNN re-evaluation: every piece of every test
puzzle at all 4 applied rotations, labels composed as
(baked_rotation_from_filename + applied) % 360.

Images are used at their native resolution: puzzle JPEGs are 256x256 and
piece PNGs are variable-size tight crops (~110-130 px) on black backgrounds.
Unlike the CNN input pipeline, pieces are NOT squashed to 128x128 squares.

Usage:
    python evaluate.py \
        --dataset-root /path/to/datasets/realistic_4x4_20k_test \
        --puzzle-root /path/to/datasets/puzzles
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "exp20_realistic_pieces"))

from dataset import ROTATION_ANGLES, get_cell_index, parse_piece_filename  # noqa: E402

METHOD_NAMES = ("ncc", "sift", "orb")

# Pixels with max(R, G, B) above this are considered piece content (the
# background introduced by the generator is pure black).
MASK_THRESHOLD = 8

LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 3.0
MIN_GOOD_MATCHES = 4
MIN_INLIERS = 3

# ORB needs aggressive settings on these small, low-texture pieces: default
# FAST/edge thresholds detect almost no keypoints (~8% of samples matchable).
ORB_PARAMS = {"nfeatures": 3000, "fastThreshold": 0, "edgeThreshold": 10, "nlevels": 12}

# Per-process cache of KeypointMatcher instances (workers reuse detectors).
_MATCHER_CACHE: dict[str, "KeypointMatcher"] = {}


def load_observed_piece(piece_path: Path, applied_rotation_idx: int) -> np.ndarray:
    """Load a piece PNG and apply the test-time rotation exactly as the CNN evaluation did.

    Mirrors RealisticPieceTestDataset: PIL clockwise rotation with
    expand=False and bilinear resampling (the piece PNG already has its
    generation-time rotation baked in).

    Args:
        piece_path: Path to the piece PNG.
        applied_rotation_idx: Index into ROTATION_ANGLES for the applied rotation.

    Returns:
        RGB uint8 array of the observed (rotated) piece.
    """
    piece_img = Image.open(piece_path).convert("RGB")
    angle = ROTATION_ANGLES[applied_rotation_idx]
    if angle != 0:
        piece_img = piece_img.rotate(-angle, expand=False, resample=Image.Resampling.BILINEAR)
    return np.array(piece_img)


def piece_mask(gray: np.ndarray) -> np.ndarray:
    """Build a template mask that excludes the black background.

    Args:
        gray: Grayscale piece image.

    Returns:
        uint8 mask (255 = piece content, 0 = background).
    """
    return (gray > MASK_THRESHOLD).astype(np.uint8) * 255


def predict_ncc(observed_rgb: np.ndarray, puzzle_bgr: np.ndarray) -> tuple[int, int] | None:
    """Predict cell and rotation via masked normalized cross-correlation.

    Slides the piece over the puzzle at each of the 4 candidate un-rotations
    (lossless np.rot90) and takes the best-scoring location + rotation.

    Args:
        observed_rgb: Observed piece (RGB uint8, rotation to be recovered).
        puzzle_bgr: Full puzzle image (BGR uint8).

    Returns:
        Tuple of (pred_cell, pred_rotation_idx), or None if no candidate fit.
    """
    best_score = -np.inf
    best: tuple[int, int] | None = None
    puz_h, puz_w = puzzle_bgr.shape[:2]

    for candidate_idx in range(4):
        # np.rot90(k) rotates counter-clockwise: un-rotates a piece that was
        # rotated clockwise by candidate_idx * 90 degrees.
        template = np.ascontiguousarray(np.rot90(observed_rgb, k=candidate_idx))
        template_bgr = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
        tpl_h, tpl_w = template_bgr.shape[:2]
        if tpl_h > puz_h or tpl_w > puz_w:
            continue

        mask = piece_mask(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY))
        response = cv2.matchTemplate(puzzle_bgr, template_bgr, cv2.TM_CCOEFF_NORMED, mask=mask)
        response = np.nan_to_num(response, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)

        if max_val > best_score:
            best_score = max_val
            center_x = (max_loc[0] + tpl_w / 2) / puz_w
            center_y = (max_loc[1] + tpl_h / 2) / puz_h
            best = (get_cell_index(center_x, center_y), candidate_idx)

    return best


class KeypointMatcher:
    """SIFT or ORB keypoint matching with ratio test + partial-affine RANSAC.

    Position is derived from the inlier centroid in puzzle coordinates and
    rotation from the recovered similarity transform snapped to the nearest
    90 degrees.
    """

    def __init__(self, kind: str):
        """Initialize detector and matcher.

        Args:
            kind: Either "sift" or "orb".
        """
        if kind == "sift":
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif kind == "orb":
            self.detector = cv2.ORB_create(**ORB_PARAMS)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError(f"Unknown keypoint matcher kind: {kind}")

    def detect_puzzle(self, puzzle_gray: np.ndarray) -> tuple[Any, Any]:
        """Detect keypoints/descriptors on the full puzzle (once per puzzle).

        Args:
            puzzle_gray: Grayscale puzzle image.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        return self.detector.detectAndCompute(puzzle_gray, None)

    def predict(
        self,
        observed_rgb: np.ndarray,
        puzzle_kp: Any,
        puzzle_des: Any,
        puzzle_shape: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Predict cell and rotation for one observed piece.

        Args:
            observed_rgb: Observed piece (RGB uint8).
            puzzle_kp: Puzzle keypoints from detect_puzzle.
            puzzle_des: Puzzle descriptors from detect_puzzle.
            puzzle_shape: (height, width) of the puzzle image.

        Returns:
            Tuple of (pred_cell, pred_rotation_idx), or None if matching failed.
        """
        gray = cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2GRAY)
        piece_kp, piece_des = self.detector.detectAndCompute(gray, piece_mask(gray))
        if piece_des is None or puzzle_des is None or len(piece_kp) < MIN_GOOD_MATCHES:
            return None

        pairs = self.matcher.knnMatch(piece_des, puzzle_des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_GOOD_MATCHES:
            return None

        src = np.float32([piece_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([puzzle_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
        )
        if transform is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
            return None

        # The transform un-rotates the observed piece onto the puzzle, so its
        # rotation angle is minus the piece's total rotation.
        theta = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
        pred_rotation_idx = int(round(-theta / 90)) % 4

        inlier_flags = inliers.ravel().astype(bool)
        centroid = dst.reshape(-1, 2)[inlier_flags].mean(axis=0)
        puz_h, puz_w = puzzle_shape
        pred_cell = get_cell_index(float(centroid[0]) / puz_w, float(centroid[1]) / puz_h)
        return pred_cell, pred_rotation_idx


def _get_matcher(kind: str) -> KeypointMatcher:
    """Get (or create) the per-process matcher for the given kind.

    Args:
        kind: Either "sift" or "orb".

    Returns:
        Cached KeypointMatcher instance.
    """
    if kind not in _MATCHER_CACHE:
        _MATCHER_CACHE[kind] = KeypointMatcher(kind)
    return _MATCHER_CACHE[kind]


def enumerate_pieces(puzzle_dir: Path, puzzle_id: str) -> list[tuple[Path, float, float, int]]:
    """Enumerate unique piece files for a puzzle, mirroring RealisticPieceTestDataset.

    Args:
        puzzle_dir: Directory containing the puzzle's piece PNGs.
        puzzle_id: Puzzle identifier.

    Returns:
        List of (piece_path, cx, cy, base_rotation), first file per position.
    """
    positions: dict[tuple[float, float], tuple[Path, int]] = {}
    for piece_path in puzzle_dir.glob(f"{puzzle_id}_x*_y*_rot*.png"):
        parsed = parse_piece_filename(piece_path.name)
        if parsed:
            _, cx, cy, base_rotation = parsed
            if (cx, cy) not in positions:
                positions[(cx, cy)] = (piece_path, base_rotation)
    return [(path, cx, cy, rot) for (cx, cy), (path, rot) in sorted(positions.items())]


def evaluate_puzzle(job: tuple[str, str, str, tuple[str, ...]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate all samples of one puzzle with all requested methods (worker entry point).

    Args:
        job: Tuple of (puzzle_id, dataset_root, puzzle_root, methods).

    Returns:
        Tuple of (sample records, per-method puzzle preparation seconds).
    """
    puzzle_id, dataset_root, puzzle_root, methods = job
    cv2.setRNGSeed(0)  # make RANSAC deterministic regardless of worker/puzzle assignment
    puzzle_rgb = np.array(Image.open(Path(puzzle_root) / f"{puzzle_id}.jpg").convert("RGB"))
    puzzle_bgr = cv2.cvtColor(puzzle_rgb, cv2.COLOR_RGB2BGR)
    puzzle_gray = cv2.cvtColor(puzzle_rgb, cv2.COLOR_RGB2GRAY)
    puzzle_shape = puzzle_gray.shape[:2]

    # One-time per-puzzle keypoint extraction (amortized over 64 samples).
    prep_seconds: dict[str, float] = {}
    puzzle_features: dict[str, tuple[Any, Any]] = {}
    for method in methods:
        if method in ("sift", "orb"):
            start = time.perf_counter()
            puzzle_features[method] = _get_matcher(method).detect_puzzle(puzzle_gray)
            prep_seconds[method] = time.perf_counter() - start

    records: list[dict[str, Any]] = []
    for piece_path, cx, cy, base_rotation in enumerate_pieces(Path(dataset_root) / puzzle_id, puzzle_id):
        true_cell = get_cell_index(cx, cy)
        for applied_idx in range(4):
            true_rotation_idx = ((base_rotation + ROTATION_ANGLES[applied_idx]) % 360) // 90
            observed = load_observed_piece(piece_path, applied_idx)

            preds: dict[str, tuple[int, int] | None] = {}
            times: dict[str, float] = {}
            for method in methods:
                start = time.perf_counter()
                if method == "ncc":
                    preds[method] = predict_ncc(observed, puzzle_bgr)
                else:
                    kp, des = puzzle_features[method]
                    preds[method] = _get_matcher(method).predict(observed, kp, des, puzzle_shape)
                times[method] = time.perf_counter() - start

            records.append(
                {
                    "puzzle_id": puzzle_id,
                    "true_cell": true_cell,
                    "true_rotation_idx": true_rotation_idx,
                    "preds": preds,
                    "times": times,
                }
            )
    return records, prep_seconds


def compute_metrics(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Compute accuracy metrics for one method over a set of sample records.

    Samples where the method produced no prediction count as wrong in the
    headline metrics; *_covered variants condition on a prediction existing.

    Args:
        records: Sample records from evaluate_puzzle.
        method: Method name.

    Returns:
        Metrics dictionary.
    """
    n = len(records)
    covered = [r for r in records if r["preds"][method] is not None]
    cell_ok = sum(r["preds"][method][0] == r["true_cell"] for r in covered)
    rot_ok = sum(r["preds"][method][1] == r["true_rotation_idx"] for r in covered)
    both_ok = sum(
        r["preds"][method][0] == r["true_cell"] and r["preds"][method][1] == r["true_rotation_idx"] for r in covered
    )
    n_cov = len(covered)
    return {
        "n_samples": n,
        "coverage": n_cov / n if n else 0.0,
        "cell_accuracy": cell_ok / n if n else 0.0,
        "rotation_accuracy": rot_ok / n if n else 0.0,
        "both_accuracy": both_ok / n if n else 0.0,
        "cell_accuracy_covered": cell_ok / n_cov if n_cov else 0.0,
        "rotation_accuracy_covered": rot_ok / n_cov if n_cov else 0.0,
        "both_accuracy_covered": both_ok / n_cov if n_cov else 0.0,
    }


def run_evaluation(
    puzzle_ids: list[str], dataset_root: Path, puzzle_root: Path, methods: tuple[str, ...], workers: int
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate all puzzles in parallel.

    Args:
        puzzle_ids: Puzzle IDs to evaluate.
        dataset_root: Root directory of the piece dataset.
        puzzle_root: Root directory of the puzzle JPEGs.
        methods: Method names to run.
        workers: Number of worker processes.

    Returns:
        Tuple of (all sample records, summed per-method puzzle prep seconds).
    """
    jobs = [(pid, str(dataset_root), str(puzzle_root), methods) for pid in puzzle_ids]
    all_records: list[dict[str, Any]] = []
    prep_totals: dict[str, float] = dict.fromkeys(methods, 0.0)
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (records, prep_seconds) in enumerate(pool.map(evaluate_puzzle, jobs, chunksize=4)):
            all_records.extend(records)
            for method, seconds in prep_seconds.items():
                prep_totals[method] += seconds
            if (i + 1) % 100 == 0:
                print(f"  [{i + 1}/{len(jobs)} puzzles] {time.time() - start:.0f}s", flush=True)
    return all_records, prep_totals


def add_hybrid(records: list[dict[str, Any]]) -> None:
    """Add a derived sift_else_ncc prediction to each record in place.

    The hybrid is still fully classical: use the SIFT prediction when SIFT
    produced one (it is highly accurate when it matches), otherwise fall back
    to NCC. Its per-sample time is the SIFT time plus the NCC time on the
    samples where the fallback is actually needed.

    Args:
        records: Sample records that already contain "sift" and "ncc" entries.
    """
    for record in records:
        sift_pred = record["preds"]["sift"]
        record["preds"]["sift_else_ncc"] = sift_pred if sift_pred is not None else record["preds"]["ncc"]
        record["times"]["sift_else_ncc"] = record["times"]["sift"] + (
            record["times"]["ncc"] if sift_pred is None else 0.0
        )


def summarize(
    all_records: list[dict[str, Any]],
    prep_totals: dict[str, float],
    methods: tuple[str, ...],
    subsample_ids: list[str],
    n_puzzles: int,
) -> dict[str, Any]:
    """Aggregate metrics over the full set and the matched subsample.

    Args:
        all_records: Sample records from all puzzles.
        prep_totals: Summed per-method puzzle preparation seconds.
        methods: Method names that were run.
        subsample_ids: Puzzle IDs in the seed-42 subsample.
        n_puzzles: Total number of puzzles evaluated.

    Returns:
        Per-method results dictionary.
    """
    subsample_set = set(subsample_ids)
    subsample_records = [r for r in all_records if r["puzzle_id"] in subsample_set]
    results: dict[str, Any] = {}
    for method in methods:
        match_seconds = sum(r["times"][method] for r in all_records)
        results[method] = {
            "full": compute_metrics(all_records, method),
            "subsample": compute_metrics(subsample_records, method),
            "runtime_ms_per_sample": match_seconds / len(all_records) * 1000 if all_records else 0.0,
            "puzzle_prep_ms_per_puzzle": prep_totals.get(method, 0.0) / n_puzzles * 1000 if n_puzzles else 0.0,
        }
    return results


def main() -> None:
    """Run the classical baselines and write results.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--puzzle-root", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="ncc,sift,orb")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-puzzles", type=int, default=0, help="Evaluate only the first N puzzles (0 = all)")
    parser.add_argument("--subsample-size", type=int, default=200)
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "results.json")
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    for method in methods:
        if method not in METHOD_NAMES:
            raise SystemExit(f"Unknown method: {method} (choose from {METHOD_NAMES})")

    puzzle_ids = sorted(d.name for d in args.dataset_root.iterdir() if d.is_dir() and d.name.startswith("puzzle_"))
    if args.limit_puzzles:
        puzzle_ids = puzzle_ids[: args.limit_puzzles]
    rng = random.Random(args.subsample_seed)
    subsample_ids = sorted(rng.sample(puzzle_ids, min(args.subsample_size, len(puzzle_ids))))
    print(f"Puzzles: {len(puzzle_ids)} | subsample: {len(subsample_ids)} (seed {args.subsample_seed})")
    print(f"Methods: {', '.join(methods)} | workers: {args.workers}")

    all_records, prep_totals = run_evaluation(puzzle_ids, args.dataset_root, args.puzzle_root, methods, args.workers)
    report_methods = methods
    if "sift" in methods and "ncc" in methods:
        add_hybrid(all_records)
        report_methods = methods + ("sift_else_ncc",)
    results = summarize(all_records, prep_totals, report_methods, subsample_ids, len(puzzle_ids))

    print(f"\nSamples: {len(all_records)} ({len(puzzle_ids)} puzzles x 16 pieces x 4 rotations)")
    for method in report_methods:
        full = results[method]["full"]
        print(
            f"{method.upper():5s} cell={full['cell_accuracy']:.1%} rot={full['rotation_accuracy']:.1%} "
            f"both={full['both_accuracy']:.1%} coverage={full['coverage']:.1%} "
            f"({results[method]['runtime_ms_per_sample']:.1f} ms/sample)"
        )

    output = {
        "dataset_root": args.dataset_root.name,
        "puzzle_root": args.puzzle_root.name,
        "n_puzzles": len(puzzle_ids),
        "n_samples": len(all_records),
        "subsample": {"seed": args.subsample_seed, "n_puzzles": len(subsample_ids)},
        "protocol": {
            "applied_rotations_per_piece": 4,
            "label_composition": "(baked_rotation_from_filename + applied) % 360",
            "piece_resolution": "native crop (~110-130 px), not resized",
            "puzzle_resolution": "native 256x256",
            "ncc": {"opencv_method": "TM_CCOEFF_NORMED", "masked": True, "mask_threshold": MASK_THRESHOLD},
            "keypoint": {
                "lowe_ratio": LOWE_RATIO,
                "ransac_reproj_threshold": RANSAC_REPROJ_THRESHOLD,
                "min_good_matches": MIN_GOOD_MATCHES,
                "min_inliers": MIN_INLIERS,
                "orb_params": ORB_PARAMS,
            },
        },
        "cnn_reference": {
            "checkpoint": "exp20 checkpoint_best.pt (reeval_fixed_labels.json)",
            "cell_accuracy": 0.729,
            "rotation_accuracy": 0.946,
            "both_accuracy": 0.722,
        },
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
