#!/usr/bin/env python3
"""Exp25: first evaluation on the north-star real-photo benchmark.

Evaluates the existing methods on `north_star v1` (14 physical kids' puzzles,
236 pieces photographed on 4 backgrounds = 944 piece photos):

- Masked multi-scale NCC template matching (adapted from exp23; real photos
  add scale uncertainty, so NCC searches over piece scales).
- SIFT keypoint matching (ratio test + partial-affine RANSAC, from exp23).
- SIFT->NCC hybrid (SIFT when it matches, NCC fallback) - exp23's winner.
- The exp20 CNN (FastBackboneModel), plain single pass.
- The exp20 CNN with test-time 4-rotation search (critical-review item #7):
  feed all 4 un-rotations, pick the one the model scores most upright.

Protocol:
- Piece preparation mirrors the deployed preview path (exp24/backend): crop
  the photo to the piece bbox from metadata.csv (this removes the orientation
  arrow), segment with rembg, composite the largest component on black, pad
  square. Failures fall back to the raw bbox crop padded square.
- The overview photo is auto-cropped to the puzzle/poster region (the photos
  have background margins that would corrupt normalized cell binning); a
  review sheet of all crops is written next to the results.
- Each of the 944 upright piece crops is evaluated at all 4 applied rotations
  (clockwise, matching the exp20 convention) -> 3,776 samples per method.
  True rotation index = applied index (pieces are photographed upright).
- Cell prediction: continuous position normalized to the cropped overview,
  binned into the puzzle's own rows x cols grid (row-major cell index).

Usage:
    uv run python evaluate.py \
        --dataset-root /path/to/datasets/north_star/v1 \
        --checkpoint /path/to/exp20_realistic_pieces/outputs/checkpoint_best.pt
"""

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "exp24_piece_classifier"))

from data_prep import pad_to_square, rgba_to_classifier_input  # noqa: E402  (exp24)


def _load_exp20_model_class() -> type:
    """Load FastBackboneModel from exp20 by file path (avoids the exp24 model.py name clash).

    Returns:
        The FastBackboneModel class.
    """
    path = Path(__file__).parent.parent / "exp20_realistic_pieces" / "model.py"
    spec = importlib.util.spec_from_file_location("exp20_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FastBackboneModel


METHOD_NAMES = ("ncc", "sift", "sift_else_ncc", "cnn", "cnn_rotsearch")
BACKGROUNDS = ("red_carpet", "gray_fabric", "cardboard", "wood")

# Pixels whose grayscale value is above this count as piece content (the
# segmented crops have pure-black backgrounds).
MASK_THRESHOLD = 8

# NCC: the piece template is resized so its content spans cell_size * scale.
# Real pieces with tabs span ~1.1-1.5 cells; the range covers loose crops too.
NCC_SCALES = (0.9, 1.1, 1.3, 1.5)
NCC_OVERVIEW_SIDE = 256  # match the synthetic benchmark's puzzle resolution

# SIFT (exp23 settings; FLANN instead of brute force for the larger images)
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_GOOD_MATCHES = 4
MIN_INLIERS = 3
SIFT_OVERVIEW_SIDE = 768
SIFT_PIECE_SIDE = 384

CNN_PIECE_SIZE = 128
CNN_PUZZLE_SIZE = 256


def rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate an image array clockwise by k * 90 degrees (lossless).

    Args:
        arr: HxWxC image array.
        k: Number of clockwise 90-degree steps.

    Returns:
        The rotated array (contiguous).
    """
    return np.ascontiguousarray(np.rot90(arr, k=-k))


def cell_index(nx: float, ny: float, rows: int, cols: int) -> int:
    """Map a normalized position to a row-major cell index for a rows x cols grid.

    Args:
        nx: Normalized x in [0, 1] (relative to the cropped overview).
        ny: Normalized y in [0, 1].
        rows: Grid rows.
        cols: Grid columns.

    Returns:
        Row-major cell index in [0, rows*cols).
    """
    col = min(max(int(nx * cols), 0), cols - 1)
    row = min(max(int(ny * rows), 0), rows - 1)
    return row * cols + col


def load_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    """Load north_star metadata.csv into per-photo records.

    Args:
        dataset_root: Root of the north_star v1 dataset.

    Returns:
        One record per piece photo with grid labels and the piece bbox.
    """
    records = []
    with open(dataset_root / "metadata.csv") as f:
        for row in csv.DictReader(f):
            records.append(
                {
                    "puzzle_id": row["puzzle_id"],
                    "piece_file": row["piece_file"],
                    "rows": int(row["rows"]),
                    "cols": int(row["cols"]),
                    "true_cell": int(row["row"]) * int(row["cols"]) + int(row["col"]),
                    "background": row["background"],
                    "bbox": (int(row["piece_x1"]), int(row["piece_y1"]), int(row["piece_x2"]), int(row["piece_y2"])),
                    "base_rotation_idx": int(row["rotation"]) // 90,
                }
            )
    return records


def crop_overview(rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Find the puzzle/poster region in an overview photo.

    Models the background as the two dominant border colours (k-means in LAB
    with down-weighted lightness, like the ingest segmenter), thresholds the
    distance map with Otsu, and takes the bounding rect of the largest
    component. Falls back to the full image when the result is implausible.

    Args:
        rgb: Overview photo as an RGB uint8 array.

    Returns:
        Crop box (x1, y1, x2, y2), exclusive on the right/bottom.
    """
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 0.5  # down-weight lightness: shadows stay background
    bw, bh = max(w // 20, 1), max(h // 20, 1)
    border = np.concatenate(
        [
            lab[:bh].reshape(-1, 3),
            lab[-bh:].reshape(-1, 3),
            lab[:, :bw].reshape(-1, 3),
            lab[:, -bw:].reshape(-1, 3),
        ]
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(border, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    dist = np.min(
        np.stack([np.linalg.norm(lab - c.reshape(1, 1, 3), axis=2) for c in centers]),
        axis=0,
    )
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((max(h // 64, 3),) * 2, np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Union of all sizable components: posters with background-like colours
    # fragment under Otsu, so the largest component alone under-crops.
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 0.02 * w * h]
    if boxes:
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        if 0.30 <= (x2 - x1) * (y2 - y1) / (w * h) <= 0.98:
            return x1, y1, x2, y2
    return 0, 0, w, h


def build_piece_cache(records: list[dict[str, Any]], dataset_root: Path, cache_dir: Path) -> dict[str, bool]:
    """Segment every piece photo through the deployed preview pipeline and cache the crops.

    bbox crop (removes the orientation arrow) -> rembg -> largest opaque
    component composited on black -> pad square. Idempotent: existing cache
    files are kept.

    Args:
        records: Metadata records from load_metadata.
        dataset_root: Root of the north_star v1 dataset.
        cache_dir: Directory for the segmented PNG crops.

    Returns:
        Mapping of piece_file -> True when rembg failed and the raw bbox crop
        was used instead.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = cache_dir / "fallbacks.json"
    if fallback_path.exists() and all((cache_dir / cache_name(r)).exists() for r in records):
        with open(fallback_path) as f:
            return json.load(f)

    from rembg import new_session, remove  # local import: heavy, pulls onnxruntime

    session = new_session("u2net")
    fallbacks: dict[str, bool] = {}
    start = time.time()
    for i, rec in enumerate(records):
        out_path = cache_dir / cache_name(rec)
        if out_path.exists():
            fallbacks[rec["piece_file"]] = False  # assume prior success; fallbacks.json is authoritative when complete
            continue
        img = Image.open(dataset_root / rec["piece_file"]).convert("RGB")
        x1, y1, x2, y2 = rec["bbox"]
        crop = img.crop((x1, y1, x2 + 1, y2 + 1))
        rgba = remove(crop, session=session)
        segmented = rgba_to_classifier_input(rgba) if isinstance(rgba, Image.Image) else None
        fallbacks[rec["piece_file"]] = segmented is None
        (segmented if segmented is not None else pad_to_square(crop)).save(out_path)
        if (i + 1) % 100 == 0:
            print(f"  [cache {i + 1}/{len(records)}] {time.time() - start:.0f}s", flush=True)
    with open(fallback_path, "w") as f:
        json.dump(fallbacks, f)
    return fallbacks


def cache_name(rec: dict[str, Any]) -> str:
    """Flat cache filename for one piece photo record.

    Args:
        rec: Metadata record.

    Returns:
        PNG filename encoding puzzle and piece photo.
    """
    return rec["piece_file"].replace("/", "__").replace(".jpg", ".png")


def predict_ncc(
    observed_rgb: np.ndarray, puzzle_bgr: np.ndarray, rows: int, cols: int
) -> tuple[tuple[int, int] | None, float]:
    """Masked NCC over 4 candidate un-rotations x piece scales.

    Args:
        observed_rgb: Observed (rotated) piece crop, RGB uint8, black background.
        puzzle_bgr: Cropped overview resized for NCC, BGR uint8.
        rows: Grid rows.
        cols: Grid columns.

    Returns:
        ((pred_cell, pred_rotation_idx) or None, best score).
    """
    puz_h, puz_w = puzzle_bgr.shape[:2]
    nominal = max(puz_w / cols, puz_h / rows)
    best_score = -np.inf
    best: tuple[int, int] | None = None
    for candidate_idx in range(4):
        unrot = np.rot90(observed_rgb, k=candidate_idx)
        for scale in NCC_SCALES:
            side = int(round(nominal * scale))
            if side < 8 or side > min(puz_h, puz_w):
                continue
            template = cv2.cvtColor(cv2.resize(unrot, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
            mask = (cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) > MASK_THRESHOLD).astype(np.uint8) * 255
            response = cv2.matchTemplate(puzzle_bgr, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            response = np.nan_to_num(response, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
            _, max_val, _, max_loc = cv2.minMaxLoc(response)
            if max_val > best_score:
                best_score = max_val
                nx = (max_loc[0] + side / 2) / puz_w
                ny = (max_loc[1] + side / 2) / puz_h
                best = (cell_index(nx, ny, rows, cols), candidate_idx)
    return best, float(best_score)


class SiftMatcher:
    """SIFT + FLANN + partial-affine RANSAC matcher (exp23 recipe, real-photo sizes)."""

    def __init__(self) -> None:
        """Create the detector and FLANN matcher."""
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})

    def detect_puzzle(self, puzzle_gray: np.ndarray) -> tuple[Any, Any]:
        """Detect keypoints/descriptors on the cropped overview (once per puzzle).

        Args:
            puzzle_gray: Grayscale overview image.

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
        rows: int,
        cols: int,
    ) -> tuple[int, int] | None:
        """Predict (cell, rotation) for one observed piece crop.

        Args:
            observed_rgb: Observed piece crop, RGB uint8, black background.
            puzzle_kp: Overview keypoints.
            puzzle_des: Overview descriptors.
            puzzle_shape: (height, width) of the overview used for detection.
            rows: Grid rows.
            cols: Grid columns.

        Returns:
            (pred_cell, pred_rotation_idx), or None when matching fails.
        """
        gray = cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2GRAY)
        mask = (gray > MASK_THRESHOLD).astype(np.uint8) * 255
        piece_kp, piece_des = self.detector.detectAndCompute(gray, mask)
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
        theta = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
        pred_rotation_idx = int(round(-theta / 90)) % 4
        inlier_flags = inliers.ravel().astype(bool)
        centroid = dst.reshape(-1, 2)[inlier_flags].mean(axis=0)
        puz_h, puz_w = puzzle_shape
        pred_cell = cell_index(float(centroid[0]) / puz_w, float(centroid[1]) / puz_h, rows, cols)
        return pred_cell, pred_rotation_idx


_SIFT: SiftMatcher | None = None


def _resize_max_side(rgb: np.ndarray, side: int) -> np.ndarray:
    """Resize an image so its longer side equals `side` (aspect preserved).

    Args:
        rgb: Image array.
        side: Target size of the longer side.

    Returns:
        Resized array (unchanged when already smaller).
    """
    h, w = rgb.shape[:2]
    scale = side / max(h, w)
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def evaluate_puzzle_classical(
    job: tuple[str, list[dict[str, Any]], str, str, tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """Worker: run NCC and SIFT on all samples of one puzzle.

    Args:
        job: (puzzle_id, records, dataset_root, cache_dir, overview_crop_box).

    Returns:
        Per-sample result dicts (4 applied rotations per piece photo).
    """
    global _SIFT
    puzzle_id, records, dataset_root, cache_dir, crop_box = job
    cv2.setRNGSeed(0)
    if _SIFT is None:
        _SIFT = SiftMatcher()

    overview = np.array(Image.open(Path(dataset_root) / puzzle_id / "overview.jpg").convert("RGB"))
    x1, y1, x2, y2 = crop_box
    overview = overview[y1:y2, x1:x2]
    ncc_bgr = cv2.cvtColor(_resize_max_side(overview, NCC_OVERVIEW_SIDE), cv2.COLOR_RGB2BGR)
    sift_rgb = _resize_max_side(overview, SIFT_OVERVIEW_SIDE)
    sift_gray = cv2.cvtColor(sift_rgb, cv2.COLOR_RGB2GRAY)
    puzzle_kp, puzzle_des = _SIFT.detect_puzzle(sift_gray)

    out = []
    for rec in records:
        crop = np.array(Image.open(Path(cache_dir) / cache_name(rec)).convert("RGB"))
        sift_crop = _resize_max_side(crop, SIFT_PIECE_SIDE)
        rows, cols = rec["rows"], rec["cols"]
        for applied_idx in range(4):
            true_rot = (rec["base_rotation_idx"] + applied_idx) % 4
            observed = rotate_cw(crop, applied_idx)
            observed_sift = rotate_cw(sift_crop, applied_idx)

            t0 = time.perf_counter()
            ncc_pred, ncc_score = predict_ncc(observed, ncc_bgr, rows, cols)
            t1 = time.perf_counter()
            sift_pred = _SIFT.predict(observed_sift, puzzle_kp, puzzle_des, sift_gray.shape[:2], rows, cols)
            t2 = time.perf_counter()

            out.append(
                {
                    "puzzle_id": puzzle_id,
                    "piece_file": rec["piece_file"],
                    "background": rec["background"],
                    "rows": rows,
                    "cols": cols,
                    "applied_idx": applied_idx,
                    "true_cell": rec["true_cell"],
                    "true_rotation_idx": true_rot,
                    "preds": {"ncc": ncc_pred, "sift": sift_pred},
                    "scores": {"ncc": ncc_score},
                    "times": {"ncc": t1 - t0, "sift": t2 - t1},
                }
            )
    return out


def evaluate_cnn(
    records_by_puzzle: dict[str, list[dict[str, Any]]],
    samples: list[dict[str, Any]],
    dataset_root: Path,
    cache_dir: Path,
    crop_boxes: dict[str, tuple[int, int, int, int]],
    checkpoint: Path,
    batch_size: int,
) -> None:
    """Run the exp20 CNN (plain + test-time rotation search) and add preds in place.

    For every sample, all 4 un-rotations of the observed piece are forwarded.
    The plain prediction reads candidate 0 (the observed piece as-is); the
    rotation-search prediction picks the candidate with the highest
    P(rotation=0) and takes cell and rotation from it.

    Args:
        records_by_puzzle: Metadata records grouped by puzzle.
        samples: Per-sample dicts from the classical stage (updated in place).
        dataset_root: Dataset root (for overview photos).
        cache_dir: Segmented piece crop cache.
        crop_boxes: Overview crop box per puzzle.
        checkpoint: Path to the exp20 checkpoint (raw state_dict).
        batch_size: Samples per forward batch (each sample = 4 candidate inputs).
    """
    import torch
    from torchvision import transforms

    fast_backbone_model = _load_exp20_model_class()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = fast_backbone_model(backbone_name="shufflenet_v2_x0_5", pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    print(f"CNN device: {device}")

    to_tensor = transforms.ToTensor()
    puzzle_tensors: dict[str, "torch.Tensor"] = {}
    for puzzle_id in records_by_puzzle:
        overview = Image.open(dataset_root / puzzle_id / "overview.jpg").convert("RGB")
        x1, y1, x2, y2 = crop_boxes[puzzle_id]
        overview = overview.crop((x1, y1, x2, y2)).resize((CNN_PUZZLE_SIZE, CNN_PUZZLE_SIZE), Image.Resampling.BILINEAR)
        puzzle_tensors[puzzle_id] = to_tensor(overview)

    crop_cache: dict[str, np.ndarray] = {}
    start = time.time()
    with torch.no_grad():
        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start : batch_start + batch_size]
            piece_inputs = []
            puzzle_inputs = []
            for s in batch:
                if s["piece_file"] not in crop_cache:
                    rec = next(r for r in records_by_puzzle[s["puzzle_id"]] if r["piece_file"] == s["piece_file"])
                    crop_cache[s["piece_file"]] = np.array(
                        Image.open(cache_dir / cache_name(rec)).convert("RGB").resize((CNN_PIECE_SIZE, CNN_PIECE_SIZE))
                    )
                observed = rotate_cw(crop_cache[s["piece_file"]], s["applied_idx"])
                for candidate in range(4):
                    piece_inputs.append(to_tensor(Image.fromarray(np.rot90(observed, k=candidate).copy())))
                    puzzle_inputs.append(puzzle_tensors[s["puzzle_id"]])
            pieces = torch.stack(piece_inputs).to(device)
            puzzles = torch.stack(puzzle_inputs).to(device)
            positions, rotation_logits, _ = model(pieces, puzzles)
            positions = positions.cpu().float().numpy().reshape(len(batch), 4, 2)
            probs = torch.softmax(rotation_logits, dim=1).cpu().float().numpy().reshape(len(batch), 4, 4)

            for s, pos4, prob4 in zip(batch, positions, probs):
                rows, cols = s["rows"], s["cols"]
                plain_cell = cell_index(float(pos4[0, 0]), float(pos4[0, 1]), rows, cols)
                s["preds"]["cnn"] = (plain_cell, int(prob4[0].argmax()))
                best_k = int(prob4[:, 0].argmax())
                search_cell = cell_index(float(pos4[best_k, 0]), float(pos4[best_k, 1]), rows, cols)
                s["preds"]["cnn_rotsearch"] = (search_cell, best_k)
                s["times"]["cnn"] = 0.0  # filled from batch totals below
                s["times"]["cnn_rotsearch"] = 0.0
            if (batch_start // batch_size + 1) % 10 == 0:
                print(f"  [cnn {batch_start + len(batch)}/{len(samples)}] {time.time() - start:.0f}s", flush=True)
    total = time.time() - start
    per_sample = total / len(samples)
    for s in samples:
        s["times"]["cnn"] = per_sample / 4  # plain pass is 1 of the 4 candidate forwards
        s["times"]["cnn_rotsearch"] = per_sample


def add_hybrid(samples: list[dict[str, Any]]) -> None:
    """Add the SIFT->NCC fallback prediction in place (exp23's best method).

    Args:
        samples: Per-sample dicts with "sift" and "ncc" predictions.
    """
    for s in samples:
        sift_pred = s["preds"]["sift"]
        s["preds"]["sift_else_ncc"] = sift_pred if sift_pred is not None else s["preds"]["ncc"]
        s["times"]["sift_else_ncc"] = s["times"]["sift"] + (s["times"]["ncc"] if sift_pred is None else 0.0)


def compute_metrics(samples: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Accuracy metrics for one method over a set of samples.

    Missing predictions count as wrong in headline metrics; *_covered
    variants condition on a prediction existing.

    Args:
        samples: Per-sample dicts.
        method: Method name.

    Returns:
        Metrics dict.
    """
    n = len(samples)
    covered = [s for s in samples if s["preds"].get(method) is not None]
    cell_ok = sum(s["preds"][method][0] == s["true_cell"] for s in covered)
    rot_ok = sum(s["preds"][method][1] == s["true_rotation_idx"] for s in covered)
    both_ok = sum(
        s["preds"][method][0] == s["true_cell"] and s["preds"][method][1] == s["true_rotation_idx"] for s in covered
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


def rotation_confusion(samples: list[dict[str, Any]], method: str) -> list[list[int]]:
    """4x4 rotation confusion matrix (rows = true, cols = predicted).

    Args:
        samples: Per-sample dicts.
        method: Method name.

    Returns:
        Confusion counts; uncovered samples are excluded.
    """
    confusion = [[0] * 4 for _ in range(4)]
    for s in samples:
        pred = s["preds"].get(method)
        if pred is not None:
            confusion[s["true_rotation_idx"]][pred[1]] += 1
    return confusion


def write_overview_sheet(
    crop_boxes: dict[str, tuple[int, int, int, int]], dataset_root: Path, path: Path, tile: int = 192
) -> None:
    """Write a review sheet of all overview crops (original with crop box + cropped).

    Args:
        crop_boxes: Crop box per puzzle.
        dataset_root: Dataset root.
        path: Output JPEG path.
        tile: Tile height in pixels.
    """
    rows = []
    for puzzle_id, (x1, y1, x2, y2) in sorted(crop_boxes.items()):
        rgb = np.array(Image.open(dataset_root / puzzle_id / "overview.jpg").convert("RGB"))
        boxed = rgb.copy()
        cv2.rectangle(boxed, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 6)
        pair = []
        for img in (boxed, rgb[y1:y2, x1:x2]):
            h, w = img.shape[:2]
            scale = tile / h
            pair.append(cv2.resize(img, (int(w * scale), tile)))
        rows.append((np.concatenate(pair, axis=1), puzzle_id))
    width = max(r.shape[1] for r, _ in rows) + 220
    sheet = np.full((len(rows) * tile, width, 3), 30, np.uint8)
    for i, (row_img, label) in enumerate(rows):
        sheet[i * tile : (i + 1) * tile, 220 : 220 + row_img.shape[1]] = row_img
        cv2.putText(sheet, label, (4, i * tile + tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(path, quality=85)


def main() -> None:
    """Run the exp25 north-star evaluation and write results.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Segmented crop cache (default: sibling of v1)")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--limit-puzzles", type=int, default=0, help="Evaluate only the first N puzzles (0 = all)")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "results.json")
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

    print("Building segmented piece cache (rembg, deployed preview pipeline)...")
    fallbacks = build_piece_cache(records, args.dataset_root, cache_dir)
    n_fallback = sum(fallbacks.get(r["piece_file"], False) for r in records)
    print(f"  cache: {len(records)} crops, {n_fallback} rembg fallbacks")

    print("Cropping overviews to the puzzle region...")
    crop_boxes: dict[str, tuple[int, int, int, int]] = {}
    for puzzle_id in puzzle_ids:
        rgb = np.array(Image.open(args.dataset_root / puzzle_id / "overview.jpg").convert("RGB"))
        crop_boxes[puzzle_id] = crop_overview(rgb)
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = crop_boxes[puzzle_id]
        print(f"  {puzzle_id}: ({x1},{y1})-({x2},{y2}) of {w}x{h} ({(x2 - x1) * (y2 - y1) / (w * h):.0%})")
    write_overview_sheet(crop_boxes, args.dataset_root, args.output.parent / "overview_crops.jpg")

    print("Classical methods (NCC multi-scale, SIFT)...")
    jobs = [
        (pid, records_by_puzzle[pid], str(args.dataset_root), str(cache_dir), crop_boxes[pid]) for pid in puzzle_ids
    ]
    samples: list[dict[str, Any]] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, puzzle_samples in enumerate(pool.map(evaluate_puzzle_classical, jobs)):
            samples.extend(puzzle_samples)
            print(f"  [{i + 1}/{len(jobs)} puzzles] {time.time() - start:.0f}s", flush=True)
    add_hybrid(samples)

    print("CNN (exp20 checkpoint, plain + rotation search)...")
    evaluate_cnn(records_by_puzzle, samples, args.dataset_root, cache_dir, crop_boxes, args.checkpoint, args.batch_size)

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
            f"{method:>14s} cell={overall['cell_accuracy']:.1%} rot={overall['rotation_accuracy']:.1%} "
            f"both={overall['both_accuracy']:.1%} coverage={overall['coverage']:.1%} "
            f"({methods_results[method]['runtime_ms_per_sample']:.0f} ms/sample)"
        )

    output = {
        "dataset": "north_star v1 (14 puzzles, 236 pieces, 944 piece photos)",
        "n_samples": len(samples),
        "n_rembg_fallbacks": n_fallback,
        "protocol": {
            "piece_prep": "bbox crop -> rembg -> largest component on black -> pad square (exp24 preview pipeline)",
            "applied_rotations_per_piece_photo": 4,
            "true_rotation": "applied index (pieces photographed upright)",
            "overview": "auto-cropped to puzzle region (see overview_crops.jpg), shared by all methods",
            "cell_binning": "continuous position normalized to cropped overview, row-major rows x cols grid",
            "ncc": {"overview_side": NCC_OVERVIEW_SIDE, "scales": list(NCC_SCALES), "mask_threshold": MASK_THRESHOLD},
            "sift": {
                "overview_side": SIFT_OVERVIEW_SIDE,
                "piece_side": SIFT_PIECE_SIDE,
                "lowe_ratio": LOWE_RATIO,
                "ransac_reproj_threshold": RANSAC_REPROJ_THRESHOLD,
            },
            "cnn": {
                "checkpoint": args.checkpoint.name,
                "piece_size": CNN_PIECE_SIZE,
                "puzzle_size": CNN_PUZZLE_SIZE,
                "rotsearch": "argmax over candidates of P(rotation=0)",
            },
        },
        "synthetic_reference": {
            "note": "same methods on the synthetic exp20 benchmark (exp23 results.json / exp20 re-eval)",
            "sift_else_ncc": {"cell_accuracy": 0.829, "rotation_accuracy": 0.903, "both_accuracy": 0.822},
            "ncc": {"cell_accuracy": 0.775, "rotation_accuracy": 0.872, "both_accuracy": 0.769},
            "cnn": {"cell_accuracy": 0.729, "rotation_accuracy": 0.946, "both_accuracy": 0.722},
        },
        "overview_crop_boxes": {pid: list(box) for pid, box in crop_boxes.items()},
        "methods": methods_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
