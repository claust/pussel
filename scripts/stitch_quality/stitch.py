#!/usr/bin/env python3
"""Offline, faithful-in-spirit reimplementation of the iOS app's glare-free min-composite stitch.

`--mode app` (default): for each present corner shot, clamp blown highlights to a flat gray
before feature detection (mirroring the app's highlight-capping, so glare doesn't produce
spurious SIFT keypoints), fit a SIFT + RANSAC homography onto the reference frame, warp the
ORIGINAL (uncapped) corner image with that homography, fill any region the corner shot doesn't
cover with white (so it can never win a darkest-pixel-wins comparison), and min-composite it
onto a running result seeded with the reference. This is a second, independent composite to
score with `score_stitch.py` against the app's own `composite.jpg` -- e.g. to check whether a
registration failure the app hit is a capture-pipeline bug or an inherently hard scene.

`--mode masked`: an experimental alternative pipeline motivated by the exp29 stitch-quality
benchmark's findings -- a global min-composite erases fine bright detail (stars) under 1-3px
residual misalignment, matte glare is a desaturating gray sheen that never approaches white,
and background micro-parallax the homography can't fit (e.g. carpet) smears under compositing.
Same SIFT+RANSAC registration, refined with `cv2.findTransformECC` for sub-pixel accuracy, then
per-frame photometric gain compensation (so an exposure-mismatched frame's global brightness
can't win the darkest-pixel-wins comparison on brightness alone), a gain-corrected min-composite,
and finally a feathered glare mask (darkening AND a reference-brightness floor, to keep dark
moving-shadow regions out) that blends the min-composite in only where it actually helps,
keeping pristine reference pixels everywhere else -- restricting compositing to real glare
regions instead of the whole frame.

Neither mode is a byte-identical port of the Swift pipeline -- both exist to let us iterate on
the stitching approach offline in Python.

Usage (from the repo root):
    uv run python scripts/stitch_quality/stitch.py /path/to/dump --out /tmp/restitched.jpg
    uv run python scripts/stitch_quality/stitch.py /path/to/dump --out /tmp/restitched.jpg --skip-unverified
    uv run python scripts/stitch_quality/stitch.py /path/to/dump --out /tmp/masked.jpg --mode masked
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from common import CaptureDump, load_dump, match_sift_ransac, resize_to_long_side

# Gray level (0-255) above which pixels are flattened before feature detection, mirroring
# the app's pre-SIFT highlight-capping at ~0.55 normalized gray.
HIGHLIGHT_CAP_GRAY = round(0.55 * 255)

# Mean central grayscale absdiff (0-255) above which --skip-unverified drops a frame (--mode
# app), or a frame is always dropped (--mode masked).
UNVERIFIED_ABSDIFF_THRESHOLD = 18.0
# Fraction of width/height kept for the central crop the verification gate checks.
CENTRAL_CROP_FRAC = 0.5

# --- --mode masked tunables ---

# Long-side size (pixels) grayscale images are downscaled to before ECC refinement -- ECC on
# the full working-size image is slow and more easily trapped in a local optimum by fine
# texture; a coarser scale converges faster onto the right sub-pixel neighborhood.
ECC_DOWNSCALE_LONG_SIDE = 1024
# Gaussian blur size ECC applies internally to both images before each iteration.
ECC_GAUSS_FILT_SIZE = 5
ECC_MAX_ITERATIONS = 100
ECC_EPSILON = 1e-6

# Gray range (0-255) a pixel must fall in, on both the reference and a warped corner frame, to
# contribute to that frame's photometric gain estimate -- excludes near-black (noisy ratio,
# possible shadow) and near-white (glare/saturation, exactly what compositing is trying to fix)
# pixels from the estimate.
GAIN_GRAY_LOW = 30
GAIN_GRAY_HIGH = 220

# Gaussian blur sigma applied to the darkening map before thresholding it into a glare mask, to
# suppress resample/JPEG noise rather than reacting to it pixel-for-pixel.
MASK_DARKENING_BLUR_SIGMA = 3.0
# Per-pixel darkening (0-255 gray levels) above which a pixel is considered candidate glare, as
# measured on `compute_darkening_robust`'s vote-of-covered-frames darkening estimate (see that
# function's docstring for why a vote rather than min or median). Empirically tuned on the real
# dumps: a sweep over {10, 15, 20, 30} on the glossy starry-box capture found carpet-region
# alpha>0.5 coverage of 83.8% / 80.1% / 68.6% / 31.3% (median-era signal) alongside local-ghosting
# p95 of 32.2 / 27.2 / 8.5 / 0.5px -- carpet leakage directly feeds spurious phase-correlation
# readings on fabric texture -- while bright-detail retention-excl-healed cleared the 0.9 target
# at every value tried (0.975/1.065/1.074/1.084). 30 was the best of the four on both leakage
# axes by a wide margin for negligible retention cost. See README.md's "Mask signal tuning
# history" for the full story (min -> median -> vote) and the residual-leakage caveat.
MASK_DARKENING_THRESHOLD = 30.0
# Reference gray level below which a pixel is excluded from the glare mask regardless of
# darkening -- keeps moving hand shadows (dark in the reference already) out: glare heals FROM
# a bright, washed-out reference pixel, a shadow doesn't.
MASK_BRIGHTNESS_FLOOR = 110.0
# Elliptical structuring-element radius (pixels) the raw glare mask is dilated by, so the
# blended region comfortably covers a glare patch's soft, gradually-darkening rim, not just its
# thresholded core.
MASK_DILATE_RADIUS_PX = 15
# Gaussian blur sigma used to feather the dilated mask into a smooth [0, 1] alpha, avoiding a
# hard seam at the blend boundary.
MASK_FEATHER_SIGMA = 8.0


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


def refine_homography_ecc(
    corner_bgr: np.ndarray, reference_bgr: np.ndarray, homography: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Sub-pixel-refine a SIFT+RANSAC homography with ECC (MOTION_HOMOGRAPHY).

    Runs `cv2.findTransformECC` on downscaled grayscale copies (working-size ECC is slower and
    more easily trapped in a local optimum by fine texture), initialized from `homography`
    rescaled into the downscaled coordinate frame. Falls back to `homography` unchanged on any
    ECC failure (non-convergence, a singular Hessian, a degenerate initial guess) -- ECC
    refinement is a bonus on top of the SIFT+RANSAC estimate, not a hard requirement, and a
    failed refinement must not make registration worse than the SIFT-only estimate.

    `cv2.findTransformECC(templateImage, inputImage, warpMatrix, ...)` fits `warpMatrix` such
    that `templateImage(x, y)` corresponds to `inputImage(warpMatrix @ [x, y, 1])` -- i.e. the
    returned matrix maps TEMPLATE coordinates onto INPUT coordinates. With
    `templateImage=reference, inputImage=corner`, that is the reference -> corner direction,
    the inverse of `homography`'s corner -> reference convention -- so the initial guess is
    `homography`'s inverse, and the result is inverted back before returning.

    Args:
        corner_bgr: Corner-offset shot, BGR, working size.
        reference_bgr: Reference shot, BGR, same size as `corner_bgr`.
        homography: 3x3 SIFT+RANSAC homography mapping corner pixels -> reference pixels,
            working size.

    Returns:
        Tuple of (refined homography, corner -> reference, working-size scale -- or
        `homography` unchanged on failure --, whether ECC refinement actually succeeded).
    """
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    corner_gray = cv2.cvtColor(corner_bgr, cv2.COLOR_BGR2GRAY)
    reference_ds, ds_scale = resize_to_long_side(reference_gray, ECC_DOWNSCALE_LONG_SIDE)
    corner_ds, _ = resize_to_long_side(corner_gray, ECC_DOWNSCALE_LONG_SIDE)

    scale = np.diag([ds_scale, ds_scale, 1.0])
    try:
        inv_scale = np.linalg.inv(scale)
        # ECC's convention maps template (reference) -> input (corner), the inverse of
        # `homography`; rescale that inverse into the downscaled coordinate frame as the
        # initial guess.
        warp_init = (scale @ np.linalg.inv(homography) @ inv_scale).astype(np.float32)
    except np.linalg.LinAlgError:
        return homography, False

    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        ECC_MAX_ITERATIONS,
        ECC_EPSILON,
    )
    try:
        # cv2-stubs' `findTransformECC` overloads don't accept `inputMask=None` (the C++ API
        # does -- it's the documented way to say "no mask") and don't otherwise match this
        # call shape; this is a stub gap, not a real type error.
        _cc, warp_ds = cv2.findTransformECC(  # type: ignore[call-overload]
            reference_ds,
            corner_ds,
            warp_init,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,  # type: ignore[arg-type]
            ECC_GAUSS_FILT_SIZE,
        )
        refined_ds = np.linalg.inv(warp_ds)  # back to corner -> reference, downscaled coords
        refined = inv_scale @ refined_ds @ scale  # back to working-size coords
    except (cv2.error, np.linalg.LinAlgError):
        return homography, False

    if not np.all(np.isfinite(refined)):
        return homography, False
    return refined, True


def warp_with_coverage(
    corner_bgr: np.ndarray, homography: np.ndarray, out_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp a corner shot into the reference frame, also returning its coverage mask.

    White fill is what makes a subsequent min-composite of `warped` alone safe: white (255)
    never wins a darkest-pixel-wins comparison, so pixels the corner shot didn't actually
    observe can't corrupt the composite. `coverage` lets callers that need to treat covered and
    uncovered pixels differently (photometric gain estimation, a coverage-aware composite) do
    so explicitly instead of relying on that safety property alone.

    Args:
        corner_bgr: Corner-offset shot to warp, BGR.
        homography: 3x3 homography mapping `corner_bgr` pixel coordinates to the reference frame.
        out_size: (width, height) of the reference frame.

    Returns:
        Tuple of (warped BGR image of size `out_size`, white outside the corner shot's
        footprint; uint8 coverage mask, same size, 255 inside the footprint and 0 outside).
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
    return warped, coverage


def warp_with_white_fill(corner_bgr: np.ndarray, homography: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Warp a corner shot into the reference frame, filling uncovered area with white.

    Args:
        corner_bgr: Corner-offset shot to warp, BGR.
        homography: 3x3 homography mapping `corner_bgr` pixel coordinates to the reference frame.
        out_size: (width, height) of the reference frame.

    Returns:
        Warped BGR image of size `out_size`, white outside the corner shot's footprint.
    """
    warped, _coverage = warp_with_coverage(corner_bgr, homography, out_size)
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
        warped_bgr[margin_y : height - margin_y, margin_x : width - margin_x],
        cv2.COLOR_BGR2GRAY,
    )
    reference_crop = cv2.cvtColor(
        reference_bgr[margin_y : height - margin_y, margin_x : width - margin_x],
        cv2.COLOR_BGR2GRAY,
    )
    mean_absdiff = float(cv2.absdiff(warped_crop, reference_crop).mean())
    return mean_absdiff <= threshold, mean_absdiff


def compute_gain_factor(warped_bgr: np.ndarray, reference_bgr: np.ndarray, coverage: np.ndarray) -> float:
    """Median reference/warped gray ratio over covered, mid-tone pixels -- one frame's exposure gain.

    Corrects an exposure/white-balance mismatch between a corner shot and the reference before
    it enters the min-composite. Left uncorrected, the darkest-pixel-wins selection can be won
    by whichever frame happens to be globally darker rather than by which pixel actually has
    less glare -- the source of the patchy, non-glare darkening seen on real captures with an
    auto-exposure drift between shots.

    Args:
        warped_bgr: Corner shot warped into the reference frame (white-filled outside its
            footprint), BGR.
        reference_bgr: Reference shot, BGR, same size as `warped_bgr`.
        coverage: uint8/bool mask, same size, nonzero where the warp actually covers (i.e. not
            white-filled).

    Returns:
        The median gain factor to multiply `warped_bgr` by; 1.0 (no correction) if no pixel
        qualifies.
    """
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid = (
        (coverage > 0)
        & (warped_gray >= GAIN_GRAY_LOW)
        & (warped_gray <= GAIN_GRAY_HIGH)
        & (reference_gray >= GAIN_GRAY_LOW)
        & (reference_gray <= GAIN_GRAY_HIGH)
    )
    if not np.any(valid):
        return 1.0
    ratio = reference_gray[valid] / np.maximum(warped_gray[valid], 1.0)
    return float(np.median(ratio))


def build_gain_corrected_min_composite(
    reference_bgr: np.ndarray, corrected_frames: List[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Min-composite gain-corrected, verified frames; fall back to the reference where none cover.

    Unlike the `--mode app` composite (seeded with the reference, so the reference always
    participates in the min), this leaves a covered pixel's value entirely up to the covering
    frame(s) -- the reference only supplies pixels no verified frame reaches at all. The
    reference's own contribution to the final output happens later, via the glare-mask alpha
    blend, not here.

    Args:
        reference_bgr: Reference shot, BGR, working size.
        corrected_frames: One (gain-corrected warped BGR float32, coverage mask) pair per
            verified corner frame.

    Returns:
        Tuple of (the min-composite BGR uint8, boolean mask of pixels at least one frame covered).
    """
    height, width = reference_bgr.shape[:2]
    min_accum = np.full((height, width, 3), 255.0, dtype=np.float32)
    coverage_any = np.zeros((height, width), dtype=bool)
    for corrected_bgr, coverage in corrected_frames:
        covered = coverage > 0
        candidate = np.minimum(min_accum, corrected_bgr)
        min_accum = np.where(covered[..., None], candidate, min_accum)
        coverage_any |= covered
    result = np.where(coverage_any[..., None], min_accum, reference_bgr.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8), coverage_any


def compute_darkening_robust(
    reference_bgr: np.ndarray, corrected_frames: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """Per-pixel darkening estimate for the glare MASK, robust to min-of-N order-statistic noise.

    `compute_glare_alpha` originally drove the glare mask directly off
    `max(0, gray(reference) - gray(min_composite))` -- the same signal `common.compute_darkening_map`
    uses for scoring. That is an honest benefit signal for SCORING (the min-composite really did
    pick that darker pixel), but as the mask's OWN candidate-selection signal it conflates real
    glare with a pure order-statistics artifact: the min of several independent noisy samples of a
    high-variance texture (e.g. carpet) reads systematically lower than any individual sample,
    everywhere, with nothing to do with glare. On a real capture with a gray carpet background
    (median gray ~135, above `MASK_BRIGHTNESS_FLOOR`), that noise floor alone pushed the min-based
    mask's alpha above 0.5 over 93% of the carpet (see README.md's "Mask signal tuning history").

    The first fix (a median across covered frames) still left real carpet leakage: Vision/ECC
    registration error is largest near the frame edges (the verification gate only checks the
    CENTRAL 50%), so at a border carpet pixel it's common for exactly 2 of 3 (or 2 of 4) covered
    frames to agree on the same shifted-texture-driven darker reading -- a coincidence, not
    glare, but a median treats 2-of-3 agreement as "most frames" and lets it through. The current
    rule replaces the median with an explicit VOTE, tuned to require broader agreement before the
    mask fires at all:

    - 2 or 3 covered frames: the MAX (brightest) covered gray -- i.e. an ALL-of-N vote. Every
      single covered frame must read darker than the reference for the pixel to register as
      darkened at all; one frame reading bright (still glared, or just texture noise) is enough
      to veto it. For n=2 this matches the median-era rule (a 2-sample median is just their
      average, which understates darkening at a genuinely glared pixel where only one frame sees
      through the glare, so max was already used); for n=3 this TIGHTENS the previous median
      (effectively a 2-of-3 vote: the median is decided by whichever 2 of the 3 samples are
      closer together) into an all-of-3 vote.
    - 4 covered frames: the SECOND-brightest covered gray -- a 3-of-4 vote. Four is common enough
      (every corner frame aligned) that requiring literal unanimity would make the mask too
      trigger-happy to reject on a single coincidental bright reading, so one frame is allowed to
      disagree (e.g. a corner shot that happens to share glare position with the reference).
      A plain median-of-4 was, in effect, closer to a 2-of-4 vote (the median of 4 sits
      between the 2nd and 3rd values, and `np.nanmedian` averages them) -- the new second-highest
      rule requires a full extra frame's agreement.

    Real glare survives this tightening because it moves between shots (each corner frame
    glares, if at all, at a different location -- see the capture technique this tool models):
    at a genuinely glared reference pixel, ALL (or all-but-one, at n=4) covered frames typically
    show the true darker surface, easily clearing even the strictest vote. A carpet pixel's
    noisy texture value, by contrast, only rarely has every covered frame land on the darker side
    of the same texture edge by coincidence -- which is exactly the discrimination this vote is
    tuned for.

    Pixels covered by fewer than 2 gain-corrected verified frames get 0 (no meaningful robust
    estimate at all -- the mask stays off there, matching this pipeline's overall bias toward
    leaving pixels alone when unsure).

    Args:
        reference_bgr: Reference shot, BGR, working size.
        corrected_frames: One (gain-corrected warped BGR float32, coverage mask) pair per
            verified corner frame (see `build_gain_corrected_min_composite`).

    Returns:
        float32 darkening map (same H, W as `reference_bgr`), 0.0 where fewer than 2 frames
        cover a pixel or the vote-selected covered gray is >= the reference gray there.
    """
    height, width = reference_bgr.shape[:2]
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if not corrected_frames:
        return np.zeros((height, width), dtype=np.float32)

    grays = np.stack(
        [cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2GRAY) for corrected_bgr, _coverage in corrected_frames], axis=0
    )
    covered = np.stack([coverage > 0 for _corrected_bgr, coverage in corrected_frames], axis=0)
    counts = covered.sum(axis=0)
    masked_grays = np.where(covered, grays, np.nan)

    # Sort covered gray values to read off the brightest and second-brightest covered readings
    # per pixel -- the two candidates the vote above picks between. Sorted via negation rather
    # than `np.sort(...)[::-1]`: numpy's sort always pushes NaN (uncovered frames) to the END of
    # an ascending sort, so reversing an ascending sort would instead put them FIRST, corrupting
    # `sorted_desc[0]`/`[1]` at any pixel fewer than every frame covers. Sorting the negation
    # ascending keeps NaN at the end while still ordering the real values from largest to
    # smallest once negated back. Pixels no corrected frame covers are all-NaN along axis 0 by
    # construction (`counts == 0` below discards them either way); numpy's own "All-NaN slice
    # encountered" RuntimeWarning for that expected, already-handled case is noise, not a signal
    # of a real problem here.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        sorted_desc = -np.sort(-masked_grays, axis=0)
    max_gray = sorted_desc[0]
    second_max_gray = sorted_desc[1] if sorted_desc.shape[0] >= 2 else sorted_desc[0]
    # n=2 or 3: ALL-of-N vote (the brightest covered reading -- one bright frame vetoes).
    # n=4: 3-of-4 vote (the second-brightest -- one dissenting frame is tolerated).
    robust_gray = np.nan_to_num(np.where(counts == 4, second_max_gray, max_gray), nan=0.0)

    darkening = np.maximum(0.0, reference_gray - robust_gray)
    return np.where(counts >= 2, darkening, 0.0).astype(np.float32)


def compute_glare_alpha(
    reference_bgr: np.ndarray, darkening_robust: np.ndarray, darkening_threshold: float = MASK_DARKENING_THRESHOLD
) -> np.ndarray:
    """Feathered [0, 1] blend weight favoring the min-composite only in likely-glare regions.

    A pixel only joins the glare mask when `darkening_robust` (see `compute_darkening_robust`),
    blurred to suppress resample/JPEG noise, clears `darkening_threshold` AND the reference pixel
    itself is bright enough (`MASK_BRIGHTNESS_FLOOR`) -- the brightness floor is what keeps a
    moving hand shadow (dark in the reference already, and darker still in whichever frame's
    shadow happened to fall elsewhere) out of the mask: glare heals FROM a bright, washed-out
    reference pixel, a shadow doesn't start bright to begin with.

    Args:
        reference_bgr: Reference shot, BGR, working size.
        darkening_robust: Per-pixel darkening estimate from `compute_darkening_robust`, same size.
        darkening_threshold: Per-pixel (blurred) darkening above which a pixel is candidate glare
            (exposed as a parameter, rather than always reading the module constant, so a
            threshold sweep can reuse one registration/compositing pass across several
            thresholds -- see `blend_with_glare_mask`).

    Returns:
        float32 alpha map (same H, W as the inputs), in [0, 1]; 0 = keep the reference pixel
        unchanged, 1 = use the min-composite pixel.
    """
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    darkening = cv2.GaussianBlur(darkening_robust, ksize=(0, 0), sigmaX=MASK_DARKENING_BLUR_SIGMA)

    raw_mask = (darkening > darkening_threshold) & (reference_gray > MASK_BRIGHTNESS_FLOOR)
    raw_mask_u8 = raw_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_DILATE_RADIUS_PX + 1,) * 2)
    dilated = cv2.dilate(raw_mask_u8, kernel)

    alpha = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, ksize=(0, 0), sigmaX=MASK_FEATHER_SIGMA)
    return np.clip(alpha, 0.0, 1.0)


@dataclass(frozen=True)
class FrameReport:
    """Per-corner-frame registration/compositing outcome.

    Attributes:
        corner: 1-based corner index (1-4).
        status: One of "verified", "unverified", "skipped_unverified", "registration_failed".
            `--mode masked` never produces "unverified" -- its verification gate is always on,
            so a frame that fails it is always "skipped_unverified".
        matches: Number of ratio-test SIFT matches found.
        inliers: Number of RANSAC inlier matches.
        mean_central_absdiff: Mean central grayscale absdiff after warping, or None when
            registration failed before a warp could be attempted.
        ecc_refined: `--mode masked` only: whether ECC sub-pixel refinement succeeded (True) or
            fell back to the SIFT+RANSAC homography unchanged (False). None for `--mode app`.
        gain: `--mode masked` only: the frame's estimated photometric gain factor (see
            `compute_gain_factor`). None for `--mode app`, or when the frame wasn't verified.
    """

    corner: int
    status: str
    matches: int
    inliers: int
    mean_central_absdiff: Optional[float]
    ecc_refined: Optional[bool] = None
    gain: Optional[float] = None


def stitch(
    dump: CaptureDump,
    skip_unverified: bool = False,
    unverified_threshold: float = UNVERIFIED_ABSDIFF_THRESHOLD,
) -> Tuple[np.ndarray, List[FrameReport]]:
    """Rebuild the glare-free composite offline from a loaded capture dump (`--mode app`).

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


@dataclass(frozen=True)
class MaskedStitchIntermediate:
    """Registration/compositing results for `--mode masked`, before the mask threshold is applied.

    Splitting this out of `stitch_masked` lets a threshold sweep (see `blend_with_glare_mask`)
    reuse one SIFT+RANSAC/ECC/gain-compensation pass -- by far the slowest part of the pipeline --
    across several `MASK_DARKENING_THRESHOLD` candidates, instead of re-registering per threshold.

    Attributes:
        reference_bgr: The dump's reference shot, BGR, working size.
        min_composite_bgr: Gain-corrected min-composite of the verified frames (see
            `build_gain_corrected_min_composite`) -- what the final blend uses inside the mask.
        darkening_robust: Per-pixel darkening estimate for mask candidacy (see
            `compute_darkening_robust`) -- independent of any threshold.
        frame_reports: One `FrameReport` per present corner shot.
    """

    reference_bgr: np.ndarray
    min_composite_bgr: np.ndarray
    darkening_robust: np.ndarray
    frame_reports: List[FrameReport]


def register_and_composite_masked(
    dump: CaptureDump,
    unverified_threshold: float = UNVERIFIED_ABSDIFF_THRESHOLD,
) -> MaskedStitchIntermediate:
    """Run registration, gain compensation, and the min-composite for `--mode masked`.

    Per verified corner frame: SIFT+RANSAC registration (as `--mode app`), `refine_homography_ecc`
    sub-pixel refinement, warp with coverage, the central-absdiff verification gate (always on --
    unlike `--mode app`, there is no `--skip-unverified` toggle here, since a badly-registered
    frame poisoning a masked-blend region is just as bad as poisoning a plain min-composite), and
    `compute_gain_factor` photometric correction. The gain-corrected, verified frames are then
    min-composited (`build_gain_corrected_min_composite`) and reduced to a robust darkening
    estimate for the mask (`compute_darkening_robust`) -- everything `blend_with_glare_mask` needs
    to try one or more `MASK_DARKENING_THRESHOLD` values without repeating this (slow) work.

    Args:
        dump: A `CaptureDump` loaded by `common.load_dump`.
        unverified_threshold: Mean central absdiff threshold for the verification gate.

    Returns:
        The `MaskedStitchIntermediate`.
    """
    out_size = (dump.reference.shape[1], dump.reference.shape[0])
    corrected_frames: List[Tuple[np.ndarray, np.ndarray]] = []
    frame_reports: List[FrameReport] = []

    for index in sorted(dump.corners):
        corner = dump.corners[index]
        sift_homography, n_matches, n_inliers = register_corner(corner, dump.reference)
        if sift_homography is None:
            frame_reports.append(FrameReport(index, "registration_failed", n_matches, n_inliers, None))
            continue

        refined_homography, ecc_refined = refine_homography_ecc(corner, dump.reference, sift_homography)
        warped, coverage = warp_with_coverage(corner, refined_homography, out_size)
        verified, mean_absdiff = frame_is_verified(warped, dump.reference, unverified_threshold)

        if not verified:
            frame_reports.append(
                FrameReport(
                    index,
                    "skipped_unverified",
                    n_matches,
                    n_inliers,
                    mean_absdiff,
                    ecc_refined,
                )
            )
            continue

        gain = compute_gain_factor(warped, dump.reference, coverage)
        corrected = np.clip(warped.astype(np.float32) * gain, 0.0, 255.0)
        corrected_frames.append((corrected, coverage))
        frame_reports.append(FrameReport(index, "verified", n_matches, n_inliers, mean_absdiff, ecc_refined, gain))

    min_composite, _coverage_any = build_gain_corrected_min_composite(dump.reference, corrected_frames)
    darkening_robust = compute_darkening_robust(dump.reference, corrected_frames)
    return MaskedStitchIntermediate(dump.reference, min_composite, darkening_robust, frame_reports)


def blend_with_glare_mask(
    intermediate: MaskedStitchIntermediate, darkening_threshold: float = MASK_DARKENING_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """Blend the min-composite over the reference within a feathered glare mask.

    Args:
        intermediate: Registration/compositing results from `register_and_composite_masked`.
        darkening_threshold: Per-pixel darkening threshold for mask candidacy, see
            `compute_glare_alpha`.

    Returns:
        Tuple of (the blended composite BGR image, the alpha mask used for the blend as a uint8
        grayscale image, 0-255, for use as a diagnostic PNG).
    """
    alpha = compute_glare_alpha(intermediate.reference_bgr, intermediate.darkening_robust, darkening_threshold)
    blended = (
        intermediate.reference_bgr.astype(np.float32) * (1.0 - alpha[..., None])
        + intermediate.min_composite_bgr.astype(np.float32) * alpha[..., None]
    )
    output = np.clip(blended, 0, 255).astype(np.uint8)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return output, alpha_u8


def stitch_masked(
    dump: CaptureDump,
    unverified_threshold: float = UNVERIFIED_ABSDIFF_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, List[FrameReport]]:
    """Rebuild the composite with ECC-refined registration, gain compensation, and a glare mask.

    See `register_and_composite_masked` and `blend_with_glare_mask` for the two halves of this
    pipeline -- split out so a threshold sweep can reuse the (slow) registration/compositing pass
    across several `MASK_DARKENING_THRESHOLD` candidates. `stitch_masked` itself just chains them
    with the module-default threshold, restricting the min-composite's effect (and its failure
    modes: fine-detail erasure under residual misalignment, background micro-parallax smearing) to
    regions that actually show glare, instead of the whole frame.

    Args:
        dump: A `CaptureDump` loaded by `common.load_dump`.
        unverified_threshold: Mean central absdiff threshold for the verification gate.

    Returns:
        Tuple of (the blended composite BGR image, the alpha mask used for the blend as a uint8
        grayscale image (0-255, for use as a diagnostic PNG), one `FrameReport` per present
        corner shot).
    """
    intermediate = register_and_composite_masked(dump, unverified_threshold)
    output, alpha_u8 = blend_with_glare_mask(intermediate)
    return output, alpha_u8, intermediate.frame_reports


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Offline reimplementation of the glare-free min-composite stitch.")
    parser.add_argument(
        "dump_dir",
        type=Path,
        help="Directory with reference.jpg, composite.jpg, corner_N.jpg, ...",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output composite image path (e.g. restitched.jpg)",
    )
    parser.add_argument(
        "--mode",
        choices=["app", "masked"],
        default="app",
        help="'app' (default): highlight-cap -> SIFT+RANSAC -> white-fill min-composite, mirroring the shipped "
        "app. 'masked': ECC-refined registration + photometric gain compensation + a feathered glare mask that "
        "blends the min-composite in only over likely-glare regions (see module docstring).",
    )
    parser.add_argument(
        "--skip-unverified",
        action="store_true",
        help="--mode app only: drop a corner frame from the composite when its central-region absdiff after "
        "warping exceeds the gate. --mode masked always gates (no toggle).",
    )
    parser.add_argument("--unverified-threshold", type=float, default=UNVERIFIED_ABSDIFF_THRESHOLD)
    args = parser.parse_args()
    if args.mode == "masked" and args.skip_unverified:
        # Reject rather than silently ignore: masked mode always gates, so
        # the flag would suggest a behavior change that never happens.
        parser.error("--skip-unverified only applies to --mode app; --mode masked always gates unverified frames")

    dump = load_dump(args.dump_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mask_path: Optional[Path] = None
    mask_coverage: Optional[float] = None
    if args.mode == "masked":
        composite, alpha_mask, frame_reports = stitch_masked(dump, args.unverified_threshold)
        mask_path = args.out.with_name(f"{args.out.stem}_mask.png")
        cv2.imwrite(str(mask_path), alpha_mask)
        # alpha_mask is alpha*255 quantized to uint8; >127 corresponds to alpha > 0.5.
        mask_coverage = float(np.count_nonzero(alpha_mask > 127)) / alpha_mask.size
    else:
        composite, frame_reports = stitch(dump, args.skip_unverified, args.unverified_threshold)

    cv2.imwrite(str(args.out), composite)

    print(f"Wrote {args.out} ({composite.shape[1]}x{composite.shape[0]})")
    if mask_path is not None:
        print(f"Wrote {mask_path} (glare-mask diagnostic)")
    if mask_coverage is not None:
        print(f"Mask coverage (alpha > 0.5): {100 * mask_coverage:.2f}% of pixels")
    for report in frame_reports:
        extra = f" absdiff={report.mean_central_absdiff:.2f}" if report.mean_central_absdiff is not None else ""
        if report.ecc_refined is not None:
            extra += f" ecc_refined={report.ecc_refined}"
        if report.gain is not None:
            extra += f" gain={report.gain:.3f}"
        print(f"  corner_{report.corner}: {report.status} (matches={report.matches} inliers={report.inliers}{extra})")


if __name__ == "__main__":
    main()
