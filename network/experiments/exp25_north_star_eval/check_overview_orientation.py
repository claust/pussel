#!/usr/bin/env python3
"""QA: determine each north-star overview photo's orientation from the pieces.

The overview JPEGs produced by ingest keep their EXIF orientation tag as
metadata, so their raw pixel orientation varies per puzzle, and iPhone EXIF is
unreliable for top-down shots. The piece crops, however, are verified upright
(arrow + contact-sheet review). This script SIFT-matches every piece crop
against the overview at all 4 raw-pixel rotations and, for each rotation,
counts pieces whose recovered in-plane rotation snaps to 0 degrees. The
correct overview orientation is the rotation where upright pieces match
upright — a strong, label-independent signal.

Usage:
    uv run python check_overview_orientation.py \
        --dataset-root /path/to/datasets/north_star/v1 \
        --cache-dir /path/to/datasets/north_star/v1_eval_cache
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from evaluate import (
    MIN_GOOD_MATCHES,
    MIN_INLIERS,
    SIFT_OVERVIEW_SIDE,
    SIFT_PIECE_SIDE,
    SiftMatcher,
    _resize_max_side,
    cache_name,
    load_metadata,
    rotate_cw,
)
from PIL import Image

# EXIF orientation tag -> clockwise rotation (degrees) a viewer would apply.
EXIF_TO_CW_DEG = {1: 0, 3: 180, 6: 90, 8: 270}


def snapped_rotation_votes(
    matcher: SiftMatcher,
    piece_gray: np.ndarray,
    piece_mask: np.ndarray,
    puzzle_kp: object,
    puzzle_des: object,
) -> int | None:
    """Recover the piece's in-plane rotation vs one overview orientation.

    Args:
        matcher: SIFT matcher.
        piece_gray: Grayscale upright piece crop.
        piece_mask: Content mask for the crop.
        puzzle_kp: Overview keypoints.
        puzzle_des: Overview descriptors.

    Returns:
        Snapped rotation index (0-3), or None when matching fails.
    """
    piece_kp, piece_des = matcher.detector.detectAndCompute(piece_gray, piece_mask)
    if piece_des is None or puzzle_des is None or len(piece_kp) < MIN_GOOD_MATCHES:
        return None
    pairs = matcher.matcher.knnMatch(piece_des, puzzle_des, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        return None
    src = np.float32([piece_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([puzzle_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    transform, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if transform is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return None
    theta = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
    return int(round(-theta / 90)) % 4


def main() -> None:
    """Vote on the upright raw-pixel rotation of every overview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / "overview_orientation.json")
    args = parser.parse_args()

    records = load_metadata(args.dataset_root)
    by_puzzle: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_puzzle[rec["puzzle_id"]].append(rec)

    matcher = SiftMatcher()
    results: dict[str, dict] = {}
    for puzzle_id in sorted(by_puzzle):
        raw = np.array(Image.open(args.dataset_root / puzzle_id / "overview.jpg").convert("RGB"))
        exif_tag = Image.open(args.dataset_root / puzzle_id / "overview.jpg").getexif().get(274, 1)

        votes = dict.fromkeys(range(4), 0)
        matched = 0
        for k in range(4):
            view = _resize_max_side(rotate_cw(raw, k), SIFT_OVERVIEW_SIDE)
            gray = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
            puzzle_kp, puzzle_des = matcher.detect_puzzle(gray)
            for rec in by_puzzle[puzzle_id]:
                crop_path = args.cache_dir / cache_name(rec)
                if not crop_path.exists():
                    continue
                crop = _resize_max_side(np.array(Image.open(crop_path).convert("RGB")), SIFT_PIECE_SIDE)
                piece_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                mask = (piece_gray > 8).astype(np.uint8) * 255
                rot = snapped_rotation_votes(matcher, piece_gray, mask, puzzle_kp, puzzle_des)
                if rot is not None:
                    matched += 1
                    if rot == 0:
                        votes[k] += 1
        best_k = max(votes, key=lambda k: votes[k])
        exif_cw = EXIF_TO_CW_DEG.get(int(exif_tag), 0)
        results[puzzle_id] = {
            "votes_upright_by_cw_rotation": votes,
            "best_cw_rotation_deg": best_k * 90,
            "exif_tag": int(exif_tag),
            "exif_implied_cw_deg": exif_cw,
            "agrees_with_exif": best_k * 90 == exif_cw,
            "n_match_attempts": matched,
        }
        print(
            f"{puzzle_id}: votes {votes} -> rotate {best_k * 90} deg CW "
            f"(EXIF implies {exif_cw}, {'OK' if best_k * 90 == exif_cw else 'MISMATCH'})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
