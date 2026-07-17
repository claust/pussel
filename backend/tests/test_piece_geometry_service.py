"""Tests for app.services.piece_geometry.service."""

from unittest.mock import MagicMock

import numpy as np
from piece_geometry_fixtures import (
    PIECE_A_COLORS,
    PIECE_A_EDGE_TYPES,
    deterministic_config,
    embed_in_frame,
    encode_png,
    make_piece_rgba,
    paint_quadrants,
    rasterize_piece,
    rgba_from_mask_and_image,
)
from PIL import Image

from app.services.piece_geometry.contour import mask_to_contour
from app.services.piece_geometry.service import PieceGeometryService, quick_quality_from_polygon


def _mocked_service(rgba: Image.Image) -> PieceGeometryService:
    """Build a PieceGeometryService whose background remover returns a fixed RGBA image."""
    remover = MagicMock()
    remover.remove_background.return_value = rgba
    return PieceGeometryService(background_remover=remover)


class TestPieceGeometryServiceProcess:
    """Tests for PieceGeometryService.process using an injected mocked background remover."""

    def test_clean_piece_produces_a_full_lockable_record(self) -> None:
        """A clean synthetic piece photo produces corners, edges, and a fingerprint, and is lockable."""
        rgba = make_piece_rgba(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        service = _mocked_service(rgba)

        record = service.process(encode_png(rgba))

        assert record.quality.is_clean is True
        assert record.corner_disagreement is False
        assert record.corners is not None and record.corners.shape == (4, 2)
        assert record.edges is not None and len(record.edges) == 4
        assert record.fingerprint is not None
        assert record.lockable is True

    def test_background_remover_receives_the_raw_bytes(self) -> None:
        """The service hands the raw photo bytes to the background remover, unchanged."""
        rgba = make_piece_rgba(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        remover = MagicMock()
        remover.remove_background.return_value = rgba
        service = PieceGeometryService(background_remover=remover)
        payload = encode_png(rgba)

        service.process(payload)

        remover.remove_background.assert_called_once_with(payload)

    def test_empty_alpha_produces_an_empty_unlockable_record(self) -> None:
        """A fully transparent segmentation result yields no contour and lockable=False."""
        empty_rgba = Image.fromarray(np.zeros((100, 100, 4), dtype=np.uint8), mode="RGBA")
        service = _mocked_service(empty_rgba)

        record = service.process(encode_png(empty_rgba))

        assert record.contour is None
        assert record.corners is None
        assert record.edges is None
        assert record.fingerprint is None
        assert record.quality.is_clean is False
        assert record.lockable is False
        # Corners never ran, so the disagreement flag asserts nothing.
        assert record.corner_disagreement is None

    def test_second_blob_inside_the_crop_fails_the_quality_gate(self) -> None:
        """A comparable second blob NEAR the piece (inside the crop margin) fails is_clean.

        Corner detection never runs, so corner_disagreement stays None.
        """
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[20:140, 20:140] = 255  # the "piece": 120x120
        mask[30:130, 148:156] = 255  # distractor inside the bbox+15% crop window
        image_bgr = paint_quadrants(mask, PIECE_A_COLORS)
        rgba = rgba_from_mask_and_image(mask, image_bgr)
        service = _mocked_service(rgba)

        record = service.process(encode_png(rgba))

        assert record.contour is not None
        assert record.quality.is_clean is False
        assert record.quality.n_large_components == 2
        assert record.corners is None
        assert record.edges is None
        assert record.fingerprint is None
        assert record.lockable is False
        assert record.corner_disagreement is None

    def test_distractor_far_from_the_piece_is_excluded_by_the_crop(self) -> None:
        """A stray object far from the piece must not fail the gate: the crop excludes it.

        This reproduces the north_star integration gap (paper arrow in the
        frame counted as a second component, area_ratio diluted by the full
        frame): the pipeline must run on the piece's bbox crop, exp28's
        calibrated frame of reference.
        """
        config = deterministic_config(PIECE_A_EDGE_TYPES)
        piece_mask, _ = rasterize_piece(config)
        piece_bgr = paint_quadrants(piece_mask, PIECE_A_COLORS)
        frame_mask, frame_bgr = embed_in_frame(piece_mask, piece_bgr, frame_shape=(1200, 1700), offset=(100, 100))
        # Distractor blob far outside the piece's bbox+15% crop window.
        frame_mask[1050:1110, 1450:1510] = 255
        frame_bgr[1050:1110, 1450:1510] = (0, 200, 200)
        rgba = rgba_from_mask_and_image(frame_mask, frame_bgr)
        service = _mocked_service(rgba)

        record = service.process(encode_png(rgba))

        assert record.quality.is_clean is True
        assert record.quality.n_large_components == 1
        assert record.corner_disagreement is False
        assert record.lockable is True
        assert record.fingerprint is not None

    def test_record_coordinates_are_in_full_image_space(self) -> None:
        """Contour, corners, and edge polylines come back in full-image (not crop) coordinates."""
        offset_x, offset_y = 300, 250
        config = deterministic_config(PIECE_A_EDGE_TYPES)
        piece_mask, gt_corners = rasterize_piece(config)
        piece_bgr = paint_quadrants(piece_mask, PIECE_A_COLORS)
        frame_mask, frame_bgr = embed_in_frame(
            piece_mask, piece_bgr, frame_shape=(1400, 1600), offset=(offset_x, offset_y)
        )
        rgba = rgba_from_mask_and_image(frame_mask, frame_bgr)
        service = _mocked_service(rgba)

        record = service.process(encode_png(rgba))

        assert record.corners is not None and record.contour is not None and record.edges is not None
        expected_corners = gt_corners + np.array([offset_x, offset_y])
        diagonal = float(np.linalg.norm(expected_corners[0] - expected_corners[2]))
        # Each detected corner is within 3% of the diagonal of its ground-truth
        # position IN FULL-IMAGE coordinates (order-independent).
        for corner in record.corners:
            nearest = float(np.min(np.linalg.norm(expected_corners - corner, axis=1)))
            assert nearest <= 0.03 * diagonal
        # The contour lives around the paste position, not around the origin.
        assert record.contour[:, 0].min() >= offset_x - 5
        assert record.contour[:, 1].min() >= offset_y - 5
        # Edge polylines are offset consistently with the contour.
        for edge in record.edges:
            assert edge.polyline[:, 0].min() >= offset_x - 5
            assert edge.polyline[:, 1].min() >= offset_y - 5


class TestQuickQualityFromPolygon:
    """Tests for quick_quality_from_polygon (the /piece/preview include_quality helper)."""

    def test_clean_piece_polygon_is_lockable(self) -> None:
        """A polygon closely tracing a clean synthetic piece is reported lockable."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None
        height, width = mask.shape
        polygon = [(float(x) / width, float(y) / height) for x, y in contour[::7]]  # thin but dense enough

        lockable, corner_disagreement = quick_quality_from_polygon(polygon)

        assert corner_disagreement is False
        assert lockable is True

    def test_degenerate_polygon_is_not_lockable(self) -> None:
        """Fewer than 3 points can't form a region: (False, None) — nothing was measured."""
        assert quick_quality_from_polygon([(0.1, 0.1), (0.2, 0.2)]) == (False, None)

    def test_empty_polygon_is_not_lockable(self) -> None:
        """An empty polygon list is handled without error and asserts no measurement."""
        assert quick_quality_from_polygon([]) == (False, None)

    def test_small_in_frame_piece_is_still_lockable(self) -> None:
        """A clean piece covering a small fraction of the frame passes (crop-relative area gate)."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None
        height, width = mask.shape
        # Scale the polygon down to ~1/5 of the frame in each dimension: the
        # piece then covers ~1.6% of the full frame, far below the 5% gate
        # that would apply if quality were (wrongly) computed frame-relative.
        polygon = [(0.4 + 0.2 * float(x) / width, 0.4 + 0.2 * float(y) / height) for x, y in contour[::7]]

        lockable, corner_disagreement = quick_quality_from_polygon(polygon)

        assert corner_disagreement is False
        assert lockable is True
