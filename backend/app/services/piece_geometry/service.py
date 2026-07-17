"""Piece-geometry pipeline orchestration: photo -> structured piece record.

Ported from ``network/experiments/exp28_piece_geometry`` end to end (contour
extraction, corner detection, edge splitting, fingerprinting) — keep
algorithm changes in sync with `contour.py`, `corners.py`, `edges.py`, and
`fingerprint.py` (each ported from the matching exp28 source file).

Pipeline: photo bytes -> rembg background removal (reusing
`app.services.background_remover.get_background_remover`, same as
`app.services.piece_detector`) -> hardened alpha mask -> largest-component
bbox -> CROP to bbox + 15% margin -> contour -> quality gate -> corners
(polydp + curvature cross-check) -> edges (split + classify + canonicalize)
-> fingerprint (shape + spatial color).

The crop step is essential for calibration parity with exp28: the exp28
pipeline (quality gates, corner windows, spatial color grid) was tuned on
piece-bbox crops with a 15% margin (from dataset metadata), not on full
camera frames. Running on the full frame both (a) counts stray objects far
from the piece as extra components, and (b) computes area_ratio against the
whole frame instead of a piece-sized crop. All pipeline stages therefore
run WITHIN the crop; the returned record's contour, corners, and raw edge
polylines are mapped back to full-image coordinates via the crop offset
(the fingerprint's spatial grid is bbox-relative and stays in the crop
frame).

A record is ``lockable`` when the contour passed the quality gate, the two
corner detectors agreed, and edges/fingerprint were successfully computed.
Quality flags only assert what was actually measured: ``corner_disagreement``
is None (not True) when corner detection never ran (e.g. the contour failed
the quality gate).
"""

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.background_remover import BackgroundRemover, get_background_remover
from app.services.piece_geometry.contour import (
    QualityMetrics,
    alpha_to_mask,
    contour_quality,
    crop_with_margin,
    largest_component_bbox,
    mask_to_contour,
)
from app.services.piece_geometry.corners import InsufficientCornersError, detect_corners_with_cross_check
from app.services.piece_geometry.edges import Edge, split_edges
from app.services.piece_geometry.fingerprint import PieceFingerprint, build_fingerprint

# rembg model used for piece-geometry segmentation, matching
# app/services/piece_detector.py's default.
REMBG_MODEL = "u2net"

# Crop margin around the piece's bbox, matching exp28's dataset-crop margin.
CROP_MARGIN_FRAC = 0.15


@dataclass(frozen=True)
class PieceGeometryRecord:
    """The full result of running the piece-geometry pipeline on one photo.

    Attributes:
        contour: Full contour (Nx2, FULL-IMAGE pixel coordinates), or None
            when no contour could be extracted at all.
        corners: 4x2 corner points (polydp, full-image coordinates), or None
            when corner detection never ran or failed.
        corner_confidences: Per-corner confidences aligned with `corners`, or None.
        corner_disagreement: True when the curvature cross-check disagreed
            with polydp (or the cross-check itself couldn't run); False when
            the detectors agreed; None when corner detection never ran at
            all (contour missing, quality gate failed, or splitting failed)
            — the flag only asserts what was actually measured.
        edges: The piece's 4 classified, canonicalized edges (contour
            order; raw polylines in full-image coordinates), or None when
            splitting failed or the quality gate was not met.
        fingerprint: The piece's shape + spatial-color fingerprint, or None
            when `edges` is None. The spatial color grid is bbox-relative
            (crop frame), which is invariant to the crop offset.
        quality: Contour quality metrics, computed WITHIN the piece crop
            (always populated; `is_clean=False` with zeroed metrics when no
            contour was found at all).
        lockable: True only when the piece is clean, corner detectors agree,
            and edges/fingerprint were computed successfully — i.e. good
            enough for the scan-lock UX to auto-accept without asking for
            more frames.
    """

    contour: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    corner_confidences: Optional[List[float]]
    corner_disagreement: Optional[bool]
    edges: Optional[List[Edge]]
    fingerprint: Optional[PieceFingerprint]
    quality: QualityMetrics
    lockable: bool


def _empty_quality_record() -> PieceGeometryRecord:
    """Build the record returned when no contour could be extracted at all.

    Returns:
        A `PieceGeometryRecord` with every downstream field empty,
        `corner_disagreement=None` (corners never ran), and `lockable=False`.
    """
    quality = QualityMetrics(n_large_components=0, border_touching=False, area_ratio=0.0, solidity=0.0, is_clean=False)
    return PieceGeometryRecord(
        contour=None,
        corners=None,
        corner_confidences=None,
        corner_disagreement=None,
        edges=None,
        fingerprint=None,
        quality=quality,
        lockable=False,
    )


def _failed_stage_record(contour_full: np.ndarray, quality: QualityMetrics) -> PieceGeometryRecord:
    """Build the record returned when the pipeline stopped after contour extraction.

    Args:
        contour_full: The extracted contour, already mapped to full-image coordinates.
        quality: The crop-frame quality metrics.

    Returns:
        A `PieceGeometryRecord` with corners/edges/fingerprint empty and
        `corner_disagreement=None` (corners never produced a measurement).
    """
    return PieceGeometryRecord(
        contour=contour_full,
        corners=None,
        corner_confidences=None,
        corner_disagreement=None,
        edges=None,
        fingerprint=None,
        quality=quality,
        lockable=False,
    )


class PieceGeometryService:
    """Runs the full photo -> piece-geometry-record pipeline."""

    def __init__(self, background_remover: Optional[BackgroundRemover] = None) -> None:
        """Initialize the service.

        Args:
            background_remover: Background remover to use; defaults to the
                shared singleton (matching `app.services.piece_detector`).
                Injectable so tests can supply a mock, mirroring
                `PieceDetector`'s pattern.
        """
        self._background_remover = background_remover or get_background_remover(REMBG_MODEL)

    def process(self, image_bytes: bytes) -> PieceGeometryRecord:
        """Run the piece-geometry pipeline on a single piece photo.

        The photo is first reduced to a crop around the largest opaque
        component's bbox (+15% margin), reproducing exp28's calibrated frame
        of reference; every stage runs within that crop, and the returned
        contour/corners/edge polylines are mapped back to full-image
        coordinates.

        Args:
            image_bytes: Raw photo bytes (JPEG/PNG).

        Returns:
            The resulting `PieceGeometryRecord`. Downstream fields (corners,
            edges, fingerprint) are None when an earlier stage fails or the
            quality gate isn't met; `quality` and `lockable` are always
            populated.
        """
        rgba = self._background_remover.remove_background(image_bytes).convert("RGBA")
        rgba_arr = np.array(rgba)
        full_mask = alpha_to_mask(rgba_arr[..., 3])

        bbox = largest_component_bbox(full_mask)
        if bbox is None:
            return _empty_quality_record()

        # Crop mask and color image to the piece bbox + margin: the whole
        # calibrated pipeline runs in this crop frame (a stray object outside
        # the crop must not fail the gate, and area_ratio is crop-relative).
        mask_crop, (offset_x, offset_y) = crop_with_margin(full_mask, bbox, CROP_MARGIN_FRAC)
        image_bgr_full = cv2.cvtColor(rgba_arr[..., :3], cv2.COLOR_RGB2BGR)
        image_crop, _ = crop_with_margin(image_bgr_full, bbox, CROP_MARGIN_FRAC)
        offset = np.array([offset_x, offset_y], dtype=np.float64)

        contour = mask_to_contour(mask_crop)
        if contour is None:
            return _empty_quality_record()

        quality = contour_quality(contour, mask_crop, (mask_crop.shape[0], mask_crop.shape[1]))
        if not quality.is_clean:
            return _failed_stage_record(contour + offset, quality)

        split = split_edges(contour)
        if split is None:
            return _failed_stage_record(contour + offset, quality)

        edges, corner_result, corner_disagreement = split
        # Fingerprint is computed in the crop frame (its spatial grid is
        # bbox-relative, so the crop offset cancels out).
        fingerprint = build_fingerprint(edges, image_crop, contour)
        lockable = quality.is_clean and not corner_disagreement

        edges_full = [replace(edge, polyline=edge.polyline + offset) for edge in edges]
        return PieceGeometryRecord(
            contour=contour + offset,
            corners=corner_result.corners + offset,
            corner_confidences=corner_result.confidences,
            corner_disagreement=corner_disagreement,
            edges=edges_full,
            fingerprint=fingerprint,
            quality=quality,
            lockable=lockable,
        )


def quick_quality_from_polygon(
    polygon: List[Tuple[float, float]], canvas_size: int = 512
) -> Tuple[bool, Optional[bool]]:
    """Best-effort (lockable, corner_disagreement) flags from a normalized region polygon.

    Used by the ``/api/v1/piece/preview`` endpoint's ``include_quality``
    option: rasterizes the already-detected, already-simplified preview
    polygon onto a fixed canvas and reruns corner detection on its dense
    re-traced contour. This is a lightweight approximation of the full
    pipeline (no fresh background-removal call), so the preview loop —
    which is polled continuously while the piece camera is open — stays fast.

    Args:
        polygon: Normalized [0, 1] (x, y) points, as returned by
            `app.services.piece_detector.PieceDetector.detect_region`.
        canvas_size: Side length of the square raster canvas the polygon is
            rendered onto before contour re-extraction.

    Returns:
        Tuple of (lockable, corner_disagreement). When the polygon is
        empty/degenerate or corner detection never produces a measurement,
        the result is (False, None) — the disagreement flag only asserts
        what was actually measured.
    """
    if len(polygon) < 3:
        return False, None

    points = np.array([[x * canvas_size, y * canvas_size] for x, y in polygon], dtype=np.int32)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    cv2.fillPoly(mask, [points], (255,))

    # Evaluate in the same crop frame (bbox + margin) as the full pipeline so
    # the crop-calibrated quality gates (notably area_ratio) apply correctly
    # even when the piece covers a small part of the camera frame.
    bbox = largest_component_bbox(mask)
    if bbox is None:
        return False, None
    mask_crop, _ = crop_with_margin(mask, bbox, CROP_MARGIN_FRAC)

    contour = mask_to_contour(mask_crop)
    if contour is None:
        return False, None

    quality = contour_quality(contour, mask_crop, (mask_crop.shape[0], mask_crop.shape[1]))
    try:
        _, corner_disagreement = detect_corners_with_cross_check(contour)
    except InsufficientCornersError:
        return False, None

    return quality.is_clean and not corner_disagreement, corner_disagreement


_piece_geometry_service: Optional[PieceGeometryService] = None


def get_piece_geometry_service() -> PieceGeometryService:
    """Get or create the singleton PieceGeometryService instance.

    Returns:
        The shared PieceGeometryService instance.
    """
    global _piece_geometry_service
    if _piece_geometry_service is None:
        _piece_geometry_service = PieceGeometryService()
    return _piece_geometry_service


def reset_piece_geometry_service() -> None:
    """Reset the singleton PieceGeometryService (test-only helper)."""
    global _piece_geometry_service
    _piece_geometry_service = None
