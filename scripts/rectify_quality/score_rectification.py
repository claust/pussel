"""Score how rectangular — and how faithful — the puzzle overview crop comes out.

Runs the shipped detection + warp (`app.services.puzzle_detector`) over a
DEBUG-build capture dump and reports, per image, where the detected quad
landed, what aspect ratio the physical rectangle actually has, what aspect the
crop ended up with, and how far the crop's own straight structure still sits
off-axis. See `README.md` in this directory for what each number means, and
`docs/puzzle-image-rectification.html` for the pipeline being measured.

Usage (from the repo root):

    uv run python scripts/rectify_quality/score_rectification.py <dump_dir>
    uv run python scripts/rectify_quality/score_rectification.py <dump_dir> --all-frames
    uv run python scripts/rectify_quality/score_rectification.py --image photo.jpg
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

# Scoped, not permanent: `test_rectify_quality.py` imports this module, and a
# `backend/` left on sys.path for the rest of the session would change how
# whatever pytest collects next resolves `app.*`. Mirrors the same pattern in
# that test file.
_BACKEND = str(Path(__file__).resolve().parents[2] / "backend")
sys.path.insert(0, _BACKEND)
try:
    from app.services.puzzle_detector import get_puzzle_detector  # noqa: E402
    from rectify import (  # noqa: E402  (same-directory module, run as a script)
        REPORTING_LONG_SIDE,
        corner_errors,
        detect_with_path,
        order_corners,
        plane_tilt_degrees,
        quad_iou,
        rectangle_aspect,
        residual_skew_degrees,
        warp_to_aspect,
    )
finally:
    sys.path.remove(_BACKEND)

Point = Tuple[float, float]

REFERENCE_FILENAME = "reference.jpg"
COMPOSITE_FILENAME = "composite.jpg"
CORNER_FILENAME_TEMPLATE = "corner_{index}.jpg"
TRUTH_FILENAME = "truth.json"
OUTPUT_DIRNAME = "rectify_scores"

# Detection confidence below which the app tells the user the detection is uncertain
# (ConfirmTrimView.lowConfidence / the frontend's LOW_CONFIDENCE_THRESHOLD).
LOW_CONFIDENCE_THRESHOLD = 0.4

DETECTED_COLOR = (255, 140, 40)
TRUTH_COLOR = (60, 200, 110)


def load_rgb(path: Path) -> "np.ndarray":
    """Load an image as EXIF-corrected RGB, matching what the backend receives.

    Args:
        path: Image file to load.

    Returns:
        RGB uint8 array.
    """
    with Image.open(path) as handle:
        return np.asarray(ImageOps.exif_transpose(handle).convert("RGB"))


def load_truth(dump_dir: Optional[Path]) -> Dict[str, List[Point]]:
    """Load hand-labelled quads written by `label_quad.py`, if any.

    Args:
        dump_dir: The dump directory, or None when scoring a loose image.

    Returns:
        Mapping of filename to four normalized (x, y) corners; empty when no
        labels exist.
    """
    if dump_dir is None:
        return {}
    path = dump_dir / TRUTH_FILENAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        name: [(float(x), float(y)) for x, y in entry["corners"]] for name, entry in payload.get("images", {}).items()
    }


def load_capture_geometry(dump_dir: Optional[Path]) -> Dict[str, Dict[str, object]]:
    """Load the per-shot ARKit camera geometry a DEBUG build dumps, if any.

    Written by `GlareFreeDump` since the plane-rectification work: matrices are
    row-major, and the array is reference-first then one entry per corner shot,
    with nulls where a shot carried no geometry. A dump without it predates that
    change or came from the Simulator path, and its composite was *not*
    plane-rectified — which is the first thing to know before reading any of the
    numbers below.

    Args:
        dump_dir: The dump directory, or None when scoring a loose image.

    Returns:
        Mapping of image filename to that shot's geometry; empty when the dump
        carries none.
    """
    if dump_dir is None:
        return {}
    path = dump_dir / "metadata.json"
    if not path.exists():
        return {}
    geometries = json.loads(path.read_text()).get("geometries")
    if not geometries:
        return {}
    names = [REFERENCE_FILENAME] + [CORNER_FILENAME_TEMPLATE.format(index=index) for index in range(1, 5)]
    return {name: entry for name, entry in zip(names, geometries) if entry}


def focal_px_for(geometry: Optional[Dict[str, object]], width: int) -> Optional[float]:
    """The camera's focal length in this image's own pixels, from the dump.

    The dumped intrinsics are a rotation of the sensor's, so the two focal
    entries have swapped places but kept their magnitudes; with square pixels
    (every iPhone rear camera) their mean is the focal length. It is rescaled
    here because the scored file may not be at the resolution the intrinsics
    describe.

    Returns None without geometry, which leaves `rectangle_aspect` to solve for
    the focal length itself — the pre-geometry behavior, and still the only
    option for the composite, whose rectified pixels have no focal length.

    Args:
        geometry: That shot's dumped geometry, or None.
        width: The scored image's pixel width.

    Returns:
        Focal length in pixels, or None.
    """
    if not geometry:
        return None
    intrinsics = geometry.get("intrinsics")
    dumped_width = geometry.get("imageWidth")
    if not intrinsics or len(intrinsics) != 9 or not dumped_width:
        return None
    matrix = np.asarray(intrinsics, dtype=float).reshape(3, 3)
    # Row-major, and the rotation may have swapped the axes, so read each focal
    # as the larger magnitude in its row's first two columns.
    focal = float(np.mean([np.abs(matrix[0, :2]).max(), np.abs(matrix[1, :2]).max()]))
    if focal <= 0:
        return None
    return focal * width / float(dumped_width)


def score_image(
    path: Path,
    truth_corners: Optional[Sequence[Point]],
    output_dir: Optional[Path],
    known_focal_px: Optional[float] = None,
    geometry: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Detect, warp and measure one image.

    Args:
        path: The image to score.
        truth_corners: Hand-labelled normalized corners, or None.
        output_dir: Directory for diagnostic images, or None to skip them.
        known_focal_px: The camera's actual focal length in pixels, when
            known. Without it the aspect metrics solve for one from the quad,
            and fall back to an assumed field of view when the quad is
            degenerate (a tilt about a single axis).
        geometry: This shot's dumped camera geometry, used to derive the focal
            length when `known_focal_px` was not supplied. Read here rather
            than by the caller so the image is only decoded once.

    Returns:
        The metrics dict for this image (also the JSON record written out).
    """
    rgb = load_rgb(path)
    height, width = rgb.shape[:2]
    if known_focal_px is None:
        known_focal_px = focal_px_for(geometry, width=width)
    detector = get_puzzle_detector()
    corners, confidence, detection_path, candidates = detect_with_path(detector, rgb)

    detected_pixels = order_corners([(x * width, y * height) for x, y in corners])
    crop = warp_to_aspect(rgb, detected_pixels, aspect=None)
    crop_aspect = crop.shape[1] / crop.shape[0]

    record: Dict[str, object] = {
        "image": path.name,
        "size": [width, height],
        "detection": {
            "path": detection_path,
            "confidence": round(confidence, 4),
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "corners": [[round(x, 5), round(y, 5)] for x, y in corners],
            "area_ratio": round(_area_ratio(corners), 4),
            "components": (
                {key: round(value, 4) for key, value in candidates[0].components.items()} if candidates else {}
            ),
            # The runners-up say how close the call was: a clear winner and a
            # photo-finish between two very different quads are different risks.
            "runners_up": [
                {"source": candidate.source, "confidence": round(candidate.confidence, 4)}
                for candidate in candidates[1:4]
            ],
        },
        "crop": {"size": [crop.shape[1], crop.shape[0]], "aspect": round(crop_aspect, 4)},
    }

    # Aspect fidelity is measured from the hand-labelled quad when there is one: it
    # isolates the warp's own aspect error from the detector's corner error, which the
    # `truth` block reports separately. Without labels the detected quad is all we have,
    # and the number then folds both errors together — flagged by `aspect.source`.
    aspect_quad = list(truth_corners) if truth_corners is not None else list(corners)
    aspect_pixels = order_corners([(x * width, y * height) for x, y in aspect_quad])
    aspect_result = rectangle_aspect(aspect_pixels, (width, height), known_focal_px=known_focal_px)
    true_aspect = aspect_result["aspect"]
    aspect_block: Dict[str, object] = {
        "source": "truth" if truth_corners is not None else "detected",
        "focal_px": round(float(aspect_result["focal_px"]), 1),
        "focal_source": aspect_result["focal_source"],
        "true_aspect": round(float(true_aspect), 4) if true_aspect else None,
        "crop_aspect": round(crop_aspect, 4),
    }
    if true_aspect:
        aspect_block["aspect_error_pct"] = round(abs(crop_aspect / float(true_aspect) - 1.0) * 100.0, 2)
        aspect_block["tilt_deg"] = round(
            plane_tilt_degrees(aspect_pixels, float(true_aspect), float(aspect_result["focal_px"]), (width, height)),
            2,
        )
    record["aspect"] = aspect_block

    skew = residual_skew_degrees(crop)
    record["skew"] = {
        "residual_skew_deg": round(float(skew["skew_deg"]), 3) if skew["skew_deg"] is not None else None,
        "n_lines": skew["n_lines"],
    }

    if truth_corners is not None:
        errors = corner_errors(corners, truth_corners)
        record["truth"] = {
            "corners": [[round(x, 5), round(y, 5)] for x, y in truth_corners],
            "iou": round(quad_iou(corners, truth_corners), 4),
            "corner_rms_px": round(errors["rms_px"], 1),
            "corner_max_px": round(errors["max_px"], 1),
            "reported_at_long_side": REPORTING_LONG_SIDE,
        }
        # What the crop would look like had the quad been right: the ceiling this
        # capture's pixels allow, so a detector fix and a warp fix can be told apart.
        truth_pixels = order_corners([(x * width, y * height) for x, y in truth_corners])
        ideal = warp_to_aspect(rgb, truth_pixels, aspect=float(true_aspect) if true_aspect else None)
        ideal_skew = residual_skew_degrees(ideal)
        record["ideal"] = {
            "crop_aspect": round(ideal.shape[1] / ideal.shape[0], 4),
            "residual_skew_deg": (
                round(float(ideal_skew["skew_deg"]), 3) if ideal_skew["skew_deg"] is not None else None
            ),
        }
        if output_dir is not None:
            cv2.imwrite(str(output_dir / f"{path.stem}_ideal.jpg"), cv2.cvtColor(ideal, cv2.COLOR_RGB2BGR))

    if output_dir is not None:
        cv2.imwrite(str(output_dir / f"{path.stem}_crop.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        overlay = _overlay(rgb, corners, truth_corners)
        cv2.imwrite(str(output_dir / f"{path.stem}_quads.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return record


def _area_ratio(corners: Sequence[Point]) -> float:
    """Fraction of the frame the quad covers.

    Args:
        corners: Four normalized (x, y) points.

    Returns:
        Area of the quad in unit coordinates.
    """
    points = np.array(order_corners(corners), dtype=np.float32)
    return float(cv2.contourArea(points.reshape(-1, 1, 2)))


def _overlay(rgb: "np.ndarray", detected: Sequence[Point], truth: Optional[Sequence[Point]]) -> "np.ndarray":
    """Draw the detected (orange) and hand-labelled (green) quads on the photo.

    Args:
        rgb: The source image.
        detected: Detected normalized corners.
        truth: Hand-labelled normalized corners, or None.

    Returns:
        A downscaled annotated copy.
    """
    height, width = rgb.shape[:2]
    scale = min(1.0, 1400 / max(width, height))
    canvas = cv2.resize(rgb, (round(width * scale), round(height * scale))) if scale < 1.0 else rgb.copy()
    canvas_h, canvas_w = canvas.shape[:2]
    for quad, color in ((detected, DETECTED_COLOR), (truth, TRUTH_COLOR)):
        if quad is None:
            continue
        points = np.array([(x * canvas_w, y * canvas_h) for x, y in order_corners(quad)], dtype=np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=3)
        for index, point in enumerate(points):
            cv2.circle(canvas, tuple(point), 8, color, -1)
            cv2.putText(
                canvas, "TL TR BR BL".split()[index], tuple(point + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
    return canvas


def collect_images(dump_dir: Path, all_frames: bool) -> List[Path]:
    """Pick which images in a dump directory to score.

    Args:
        dump_dir: The dump directory.
        all_frames: Whether to include the reference and corner shots as well
            as the composite.

    Returns:
        Existing image paths, composite first.
    """
    names = [COMPOSITE_FILENAME]
    if all_frames:
        names.append(REFERENCE_FILENAME)
        names.extend(CORNER_FILENAME_TEMPLATE.format(index=index) for index in range(1, 5))
    return [dump_dir / name for name in names if (dump_dir / name).exists()]


def print_table(records: Sequence[Dict[str, object]]) -> None:
    """Print the one-line-per-image summary.

    Args:
        records: The per-image metric dicts.
    """
    header = (
        f"{'image':<16}{'path':<9}{'conf':>6}{'IoU':>7}{'corner_rms':>12}"
        f"{'true_ar':>9}{'crop_ar':>9}{'ar_err%':>9}{'tilt°':>7}{'skew°':>7}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        detection = record["detection"]
        aspect = record["aspect"]
        truth = record.get("truth")
        print(
            f"{str(record['image']):<16}"
            f"{str(detection['path']):<9}"
            f"{_cell(detection['confidence'], 2, 6)}"
            f"{_cell(truth['iou'] if truth else None, 2, 7)}"
            f"{_cell(truth['corner_rms_px'] if truth else None, 1, 12)}"
            f"{_cell(aspect['true_aspect'], 3, 9)}"
            f"{_cell(aspect['crop_aspect'], 3, 9)}"
            f"{_cell(aspect.get('aspect_error_pct'), 2, 9)}"
            f"{_cell(aspect.get('tilt_deg'), 1, 7)}"
            f"{_cell(record['skew']['residual_skew_deg'], 2, 7)}"
        )


def _cell(value: Optional[float], decimals: int, width: int) -> str:
    """Format one table cell, showing an em dash where a metric is unavailable.

    Args:
        value: The number, or None when the metric could not be computed.
        decimals: Decimal places.
        width: Column width.

    Returns:
        The right-aligned cell text.
    """
    if value is None:
        return f"{'—':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def main() -> int:
    """Entry point.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_dir", nargs="?", type=Path, help="Capture dump directory")
    parser.add_argument("--image", type=Path, help="Score a single loose image instead of a dump directory")
    parser.add_argument("--all-frames", action="store_true", help="Also score reference.jpg and corner_1..4.jpg")
    parser.add_argument("--out", type=Path, help=f"Output directory (default: <dump_dir>/{OUTPUT_DIRNAME})")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip writing annotated images")
    parser.add_argument(
        "--focal-px",
        type=float,
        help="The camera's focal length in pixels, if known — otherwise it is solved for from the quad",
    )
    arguments = parser.parse_args()

    if arguments.image is not None:
        images = [arguments.image]
        dump_dir = None
        default_out = arguments.image.parent / OUTPUT_DIRNAME
    elif arguments.dump_dir is not None:
        dump_dir = arguments.dump_dir
        images = collect_images(dump_dir, arguments.all_frames)
        default_out = dump_dir / OUTPUT_DIRNAME
        if not images:
            print(f"No images found in {dump_dir}", file=sys.stderr)
            return 1
    else:
        parser.error("give a dump directory or --image")
        return 2

    output_dir: Optional[Path] = None
    if not arguments.no_diagnostics:
        output_dir = arguments.out or default_out
        output_dir.mkdir(parents=True, exist_ok=True)

    truth = load_truth(dump_dir)
    if dump_dir is not None and not truth:
        print(f"No {TRUTH_FILENAME} — run label_quad.py to enable IoU and true-aspect metrics.\n", file=sys.stderr)

    geometry = load_capture_geometry(dump_dir)
    if dump_dir is not None:
        print(
            (
                "Capture geometry: present — this burst's composite was plane-rectified on device."
                if geometry
                else "Capture geometry: absent — composite NOT plane-rectified (pre-geometry dump, "
                "Simulator path, or the surface was never found)."
            ),
            file=sys.stderr,
        )

    records = [
        score_image(path, truth.get(path.name), output_dir, arguments.focal_px, geometry.get(path.name))
        for path in images
    ]
    print_table(records)
    if output_dir is not None:
        metrics_path = output_dir / "rectify_metrics.json"
        metrics_path.write_text(json.dumps({"images": records}, indent=2) + "\n")
        print(f"\nWrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
