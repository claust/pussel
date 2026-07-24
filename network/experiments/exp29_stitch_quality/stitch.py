#!/usr/bin/env python3
"""Offline, faithful-in-spirit reimplementation of the iOS app's glare-free min-composite stitch.

For each present corner shot: clamp blown highlights to a flat gray before feature
detection (mirroring the app's highlight-capping, so glare doesn't produce spurious
SIFT keypoints), fit a SIFT + RANSAC homography onto the reference frame, warp the
ORIGINAL (uncapped) corner image with that homography, fill any region the corner
shot doesn't cover with white (so it can never win a darkest-pixel-wins comparison),
and min-composite it onto a running result seeded with the reference. This is a
second, independent composite to score with `score_stitch.py` against the app's own
`composite.jpg` -- e.g. to check whether a registration failure the app hit is a
capture-pipeline bug or an inherently hard scene.

This is NOT a byte-identical reimplementation of the Swift pipeline -- it exists to
let us iterate on the stitching approach offline in Python.

Usage:
    cd network
    uv run python experiments/exp29_stitch_quality/stitch.py /path/to/dump --out /tmp/restitched.jpg
    uv run python experiments/exp29_stitch_quality/stitch.py /path/to/dump --out /tmp/restitched.jpg --skip-unverified
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from common import CaptureDump, load_dump, match_sift_ransac

# Gray level (0-255) above which pixels are flattened before feature detection, mirroring
# the app's pre-SIFT highlight-capping at ~0.55 normalized gray.
HIGHLIGHT_CAP_GRAY = round(0.55 * 255)

# Mean central grayscale absdiff (0-255) above which --skip-unverified drops a frame.
UNVERIFIED_ABSDIFF_THRESHOLD = 18.0
# Fraction of width/height kept for the central crop the --skip-unverified gate checks.
CENTRAL_CROP_FRAC = 0.5


def capped_gray_for_features(bgr: np.ndarray, cap: int = HIGHLIGHT_CAP_GRAY) -> np.ndarray:
    """Grayscale copy with blown highlights clamped flat, for feature detection only.

    Glare blows a region of the sensor image to near-white, which otherwise produces a
    cluster of spurious, poorly localized SIFT keypoints along its rim. Clamping removes
    the contrast that creates them. The ORIGINAL color image (not this one) is still what
    gets warped and composited.

    Args:
        bgr: Source color image, BGR.
        cap: Gray level (0-255) above which pixels are flattened.

    Returns:
        Single-channel grayscale image with all values <= `cap`.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.where(gray > cap, np.uint8(cap), gray)


def register_corner(corner_bgr: np.ndarray, reference_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], int, int]:
    """Estimate a SIFT + RANSAC homography mapping a corner shot onto the reference frame.

    Args:
        corner_bgr: Corner-offset shot, BGR, same working size as `reference_bgr`.
        reference_bgr: Centered reference shot, BGR.

    Returns:
        Tuple of (homography mapping corner pixels -> reference pixels, or None if
        registration failed, number of ratio-test matches, number of RANSAC inliers).
    """
    result = match_sift_ransac(capped_gray_for_features(corner_bgr), capped_gray_for_features(reference_bgr))
    return result.homography, result.n_ratio_matches, result.n_inliers


def warp_with_white_fill(corner_bgr: np.ndarray, homography: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Warp a corner shot into the reference frame, filling uncovered area with white.

    White fill is what makes the subsequent min-composite safe: white (255) never wins a
    darkest-pixel-wins comparison, so pixels the corner shot didn't actually observe can't
    corrupt the composite.

    Args:
        corner_bgr: Corner-offset shot to warp, BGR.
        homography: 3x3 homography mapping `corner_bgr` pixel coordinates to the reference frame.
        out_size: (width, height) of the reference frame.

    Returns:
        Warped BGR image of size `out_size`, white outside the corner shot's footprint.
    """
    warped = cv2.warpPerspective(corner_bgr, homography, out_size, borderValue=(255, 255, 255))
    coverage = cv2.warpPerspective(
        np.full(corner_bgr.shape[:2], 255, dtype=np.uint8),
        homography,
        out_size,
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    warped[coverage < 128] = 255
    return warped


def frame_is_verified(warped_bgr: np.ndarray, reference_bgr: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Check whether a warped corner frame agrees with the reference in a central crop.

    A grossly wrong homography (e.g. RANSAC latching onto a repeated pattern) still
    produces *a* warp; this sanity gate catches it by comparing mean absolute grayscale
    difference in the image's central region, which both the reference and every corner
    shot should cover regardless of registration quality.

    Args:
        warped_bgr: Corner frame warped into the reference frame (white-filled).
        reference_bgr: Reference frame, same size.
        threshold: Mean central absdiff (0-255 gray scale) above which the frame is rejected.

    Returns:
        Tuple of (verified, mean central absdiff).
    """
    height, width = reference_bgr.shape[:2]
    margin_y = int(height * (1 - CENTRAL_CROP_FRAC) / 2)
    margin_x = int(width * (1 - CENTRAL_CROP_FRAC) / 2)
    warped_crop = cv2.cvtColor(
        warped_bgr[margin_y : height - margin_y, margin_x : width - margin_x], cv2.COLOR_BGR2GRAY
    )
    reference_crop = cv2.cvtColor(
        reference_bgr[margin_y : height - margin_y, margin_x : width - margin_x], cv2.COLOR_BGR2GRAY
    )
    mean_absdiff = float(cv2.absdiff(warped_crop, reference_crop).mean())
    return mean_absdiff <= threshold, mean_absdiff


@dataclass(frozen=True)
class FrameReport:
    """Per-corner-frame registration/compositing outcome.

    Attributes:
        corner: 1-based corner index (1-4).
        status: One of "verified", "unverified", "skipped_unverified", "registration_failed".
        matches: Number of ratio-test SIFT matches found.
        inliers: Number of RANSAC inlier matches.
        mean_central_absdiff: Mean central grayscale absdiff after warping, or None when
            registration failed before a warp could be attempted.
    """

    corner: int
    status: str
    matches: int
    inliers: int
    mean_central_absdiff: Optional[float]


def stitch(
    dump: CaptureDump,
    skip_unverified: bool = False,
    unverified_threshold: float = UNVERIFIED_ABSDIFF_THRESHOLD,
) -> Tuple[np.ndarray, List[FrameReport]]:
    """Rebuild the glare-free composite offline from a loaded capture dump.

    Args:
        dump: A `CaptureDump` loaded by `common.load_dump`.
        skip_unverified: When True, drop any corner frame whose warp fails the central-absdiff
            sanity gate instead of compositing it.
        unverified_threshold: Mean central absdiff threshold for the gate.

    Returns:
        Tuple of (the rebuilt composite BGR image, one `FrameReport` per present corner shot).
    """
    out_size = (dump.reference.shape[1], dump.reference.shape[0])
    composite = dump.reference.copy()
    frame_reports: List[FrameReport] = []

    for index in sorted(dump.corners):
        corner = dump.corners[index]
        homography, n_matches, n_inliers = register_corner(corner, dump.reference)
        if homography is None:
            frame_reports.append(FrameReport(index, "registration_failed", n_matches, n_inliers, None))
            continue

        warped = warp_with_white_fill(corner, homography, out_size)
        verified, mean_absdiff = frame_is_verified(warped, dump.reference, unverified_threshold)

        if skip_unverified and not verified:
            frame_reports.append(FrameReport(index, "skipped_unverified", n_matches, n_inliers, mean_absdiff))
            continue

        composite = np.minimum(composite, warped)
        status = "verified" if verified else "unverified"
        frame_reports.append(FrameReport(index, status, n_matches, n_inliers, mean_absdiff))

    return composite, frame_reports


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Offline reimplementation of the glare-free min-composite stitch.")
    parser.add_argument("dump_dir", type=Path, help="Directory with reference.jpg, composite.jpg, corner_N.jpg, ...")
    parser.add_argument("--out", type=Path, required=True, help="Output composite image path (e.g. restitched.jpg)")
    parser.add_argument(
        "--skip-unverified",
        action="store_true",
        help="Drop a corner frame from the composite when its central-region absdiff after warping exceeds the gate",
    )
    parser.add_argument("--unverified-threshold", type=float, default=UNVERIFIED_ABSDIFF_THRESHOLD)
    args = parser.parse_args()

    dump = load_dump(args.dump_dir)
    composite, frame_reports = stitch(dump, args.skip_unverified, args.unverified_threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), composite)

    print(f"Wrote {args.out} ({composite.shape[1]}x{composite.shape[0]})")
    for report in frame_reports:
        extra = f" absdiff={report.mean_central_absdiff:.2f}" if report.mean_central_absdiff is not None else ""
        print(f"  corner_{report.corner}: {report.status} (matches={report.matches} inliers={report.inliers}{extra})")


if __name__ == "__main__":
    main()
