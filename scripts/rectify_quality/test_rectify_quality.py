"""Synthetic self-tests for the rectification metrics.

These validate the *measurements*, not the detector: a metric nobody has
checked against a known answer is worse than no metric, because a wrong number
still looks like progress. Each test renders a rectangle of known aspect ratio
through a known camera and checks the metric recovers it.

Run with:

    uv run pytest scripts/rectify_quality/test_rectify_quality.py
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest

# Scoped, not permanent: a generic module name like `rectify` must not stay
# importable (and shadowable) for whatever the test runner collects next.
# Mirrors scripts/stitch_quality/test_stitch_quality.py.
sys.path.insert(0, str(Path(__file__).parent))
try:
    from rectify import (  # noqa: E402
        corner_errors,
        order_corners,
        plane_tilt_degrees,
        quad_iou,
        rectangle_aspect,
        residual_skew_degrees,
        warp_to_aspect,
    )
    from score_rectification import focal_px_for, load_capture_geometry  # noqa: E402
finally:
    sys.path.remove(str(Path(__file__).parent))

Point = Tuple[float, float]

IMAGE_SIZE = (1600, 1200)
FOCAL_PX = 1500.0


def project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> List[Point]:
    """Project 3-D plane points into the image with a pinhole camera.

    Args:
        points: (N, 3) world points.
        rotation: 3x3 rotation matrix (world to camera).
        translation: (3,) translation (world to camera).

    Returns:
        The projected (x, y) pixel points.
    """
    width, height = IMAGE_SIZE
    calibration = np.array([[FOCAL_PX, 0, width / 2], [0, FOCAL_PX, height / 2], [0, 0, 1]])
    camera = (rotation @ points.T).T + translation
    projected = (calibration @ camera.T).T
    return [(float(x / z), float(y / z)) for x, y, z in projected]


def rotation_about_x(degrees: float) -> np.ndarray:
    """Rotation matrix about the x axis.

    Args:
        degrees: Angle in degrees.

    Returns:
        A 3x3 rotation matrix.
    """
    angle = math.radians(degrees)
    return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])


def rectangle_corners(aspect: float, tilt_deg: float, yaw_deg: float = 0.0) -> List[Point]:
    """Image corners of a unit-height rectangle of the given aspect, seen at a tilt.

    Args:
        aspect: The rectangle's width/height.
        tilt_deg: Rotation about the horizontal axis (0 is square-on).
        yaw_deg: Additional rotation about the vertical axis.

    Returns:
        Four (x, y) pixel corners ordered TL, TR, BR, BL.
    """
    half_w, half_h = aspect / 2.0, 0.5
    plane = np.array([(-half_w, -half_h, 0.0), (half_w, -half_h, 0.0), (half_w, half_h, 0.0), (-half_w, half_h, 0.0)])
    yaw = math.radians(yaw_deg)
    rotation_y = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    rotation = rotation_about_x(tilt_deg) @ rotation_y
    return project(plane, rotation, np.array([0.0, 0.0, 2.2]))


@pytest.mark.parametrize("aspect", [1.0, 1.4, 0.72])
@pytest.mark.parametrize("tilt", [12.0, 25.0, 38.0])
def test_rectangle_aspect_recovers_known_ratio(aspect: float, tilt: float) -> None:
    """The true aspect ratio comes back from the image quad within 2%."""
    corners = rectangle_corners(aspect, tilt_deg=tilt, yaw_deg=8.0)
    result = rectangle_aspect(corners, IMAGE_SIZE)
    assert result["aspect"] is not None
    assert result["aspect"] == pytest.approx(aspect, rel=0.02)


def test_rectangle_aspect_recovers_focal_length() -> None:
    """The focal length solved for from the quad matches the camera that took it."""
    corners = rectangle_corners(1.4, tilt_deg=30.0, yaw_deg=10.0)
    result = rectangle_aspect(corners, IMAGE_SIZE)
    assert result["focal_source"] == "estimated"
    assert float(result["focal_px"]) == pytest.approx(FOCAL_PX, rel=0.05)


def test_square_on_view_is_parallel_and_needs_no_focal() -> None:
    """A square-on shot yields a parallelogram; the aspect still comes out right."""
    corners = rectangle_corners(1.4, tilt_deg=0.0)
    result = rectangle_aspect(corners, IMAGE_SIZE)
    assert result["focal_source"] == "parallel"
    assert float(result["aspect"]) == pytest.approx(1.4, rel=0.01)


def test_single_vanishing_point_falls_back_to_an_assumed_focal() -> None:
    """A tilt about exactly one axis cannot solve for the focal length.

    With zero yaw the horizontal edges stay parallel in the image, so there is
    only one finite vanishing point and the orthogonality constraint has
    nothing to pin `f` with. The estimate then rides on the assumed field of
    view, and the aspect error that follows (~7% here, for an assumed focal
    23% off the true one) is the strongest argument for dumping the camera's
    real intrinsics alongside each capture rather than inferring them.
    """
    corners = rectangle_corners(1.0, tilt_deg=35.0, yaw_deg=0.0)
    result = rectangle_aspect(corners, IMAGE_SIZE)
    assert result["focal_source"] == "assumed"
    assert float(result["aspect"]) != pytest.approx(1.0, rel=0.02)

    # Handing it the true focal length repairs it completely.
    informed = rectangle_aspect(corners, IMAGE_SIZE, known_focal_px=FOCAL_PX)
    assert informed["focal_source"] == "known"
    assert float(informed["aspect"]) == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("tilt", [0.0, 15.0, 35.0])
def test_plane_tilt_matches_the_rendering_tilt(tilt: float) -> None:
    """The recovered plane tilt matches the angle the scene was rendered at."""
    corners = rectangle_corners(1.4, tilt_deg=tilt)
    result = rectangle_aspect(corners, IMAGE_SIZE)
    focal = float(result["focal_px"]) if result["focal_source"] == "estimated" else FOCAL_PX
    measured = plane_tilt_degrees(corners, float(result["aspect"]), focal, IMAGE_SIZE)
    assert measured == pytest.approx(tilt, abs=2.0)


def test_backend_warp_sizing_is_wrong_under_tilt() -> None:
    """The shipped max-edge-length sizing mis-shapes a tilted rectangle.

    This is the defect the `aspect_error_pct` metric exists to quantify, so it
    is worth pinning: a square photographed at 35 degrees warps to a crop that
    is materially non-square, while the true-aspect warp stays square.
    """
    corners = rectangle_corners(1.0, tilt_deg=35.0, yaw_deg=6.0)
    scene = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    backend_crop = warp_to_aspect(scene, corners, aspect=None)
    backend_aspect = backend_crop.shape[1] / backend_crop.shape[0]
    assert abs(backend_aspect - 1.0) > 0.1

    true_aspect = float(rectangle_aspect(corners, IMAGE_SIZE)["aspect"])
    corrected = warp_to_aspect(scene, corners, aspect=true_aspect)
    assert corrected.shape[1] / corrected.shape[0] == pytest.approx(1.0, rel=0.02)


def _striped_rectangle(angle_deg: float) -> np.ndarray:
    """Render a bordered rectangle of horizontal/vertical lines rotated by an angle.

    Args:
        angle_deg: Rotation applied to the whole pattern.

    Returns:
        An RGB uint8 image.
    """
    image = np.full((600, 800, 3), 30, dtype=np.uint8)
    for offset in range(60, 540, 60):
        cv2.line(image, (60, offset), (740, offset), (230, 230, 230), 3)
    for offset in range(60, 740, 60):
        cv2.line(image, (offset, 60), (offset, 540), (230, 230, 230), 3)
    matrix = cv2.getRotationMatrix2D((400, 300), angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (800, 600), borderValue=(30, 30, 30))


def test_residual_skew_is_zero_on_axis_aligned_structure() -> None:
    """An axis-aligned grid reads as unskewed."""
    result = residual_skew_degrees(_striped_rectangle(0.0))
    assert result["skew_deg"] is not None
    assert float(result["skew_deg"]) < 1.0


@pytest.mark.parametrize("angle", [4.0, 9.0])
def test_residual_skew_tracks_a_rotated_pattern(angle: float) -> None:
    """A pattern rotated by a known angle reads back as that angle."""
    result = residual_skew_degrees(_striped_rectangle(angle))
    assert result["skew_deg"] is not None
    assert float(result["skew_deg"]) == pytest.approx(angle, abs=1.5)


def test_quad_iou_is_one_for_identical_quads_and_zero_for_disjoint() -> None:
    """Overlap scoring behaves at both extremes."""
    quad = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    assert quad_iou(quad, quad) == pytest.approx(1.0, abs=0.01)
    far = [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]
    assert quad_iou(quad, far) == pytest.approx(0.0, abs=0.01)


def test_corner_errors_are_order_independent() -> None:
    """Corner errors compare TL to TL regardless of the input ordering."""
    truth = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    shuffled = [truth[2], truth[0], truth[3], truth[1]]
    errors = corner_errors(shuffled, truth)
    assert errors["rms_px"] == pytest.approx(0.0, abs=1e-6)


def test_order_corners_matches_the_backend_convention() -> None:
    """Ordering agrees with `puzzle_detector._order_corners`."""
    points = [(9.0, 1.0), (1.0, 8.0), (1.0, 1.0), (9.0, 8.0)]
    assert order_corners(points) == [(1.0, 1.0), (9.0, 1.0), (9.0, 8.0), (1.0, 8.0)]


# --- Dumped capture geometry ------------------------------------------------


def _write_metadata(directory: Path, payload: dict) -> None:
    """Write a dump's metadata.json.

    Args:
        directory: The dump directory.
        payload: The JSON body.
    """
    (directory / "metadata.json").write_text(json.dumps(payload))


def test_capture_geometry_maps_onto_the_dumps_filenames(tmp_path: Path) -> None:
    """The reference-first array lines up with reference.jpg and corner_N.jpg."""
    _write_metadata(tmp_path, {"geometries": [{"planeHeight": -0.5}, None, {"planeHeight": -0.5}]})
    geometry = load_capture_geometry(tmp_path)
    # Position matters: the null must drop corner_1, not shift corner_2's
    # geometry onto it.
    assert set(geometry) == {"reference.jpg", "corner_2.jpg"}


def test_capture_geometry_is_empty_for_a_pre_geometry_dump(tmp_path: Path) -> None:
    """A dump written before the geometry work reports nothing, not an error."""
    _write_metadata(tmp_path, {"alignedFrameCount": 4})
    assert load_capture_geometry(tmp_path) == {}
    assert load_capture_geometry(None) == {}


def test_focal_reads_through_the_portrait_rotation() -> None:
    """The dumped intrinsics give back the focal length despite the rotation.

    The still is rotated 90 degrees into portrait, which moves each focal entry
    off the diagonal — reading the matrix naively would return 0.
    """
    rotated = [0.0, -1500.0, 3024.0, 1500.0, 0.0, 2016.0, 0.0, 0.0, 1.0]
    geometry = {"intrinsics": rotated, "imageWidth": 3024.0}
    assert focal_px_for(geometry, width=3024) == pytest.approx(1500.0)


def test_focal_rescales_to_the_scored_images_resolution() -> None:
    """A dump scored at half resolution halves the focal length with it."""
    geometry = {"intrinsics": [0.0, -1500.0, 3024.0, 1500.0, 0.0, 2016.0, 0.0, 0.0, 1.0], "imageWidth": 3024.0}
    assert focal_px_for(geometry, width=1512) == pytest.approx(750.0)


def test_focal_is_none_without_geometry() -> None:
    """No geometry means the aspect solver keeps estimating the focal itself."""
    assert focal_px_for(None, width=2048) is None
    assert focal_px_for({"imageWidth": 3024.0}, width=2048) is None
