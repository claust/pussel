"""Tests for app.services.piece_geometry.fingerprint."""

import numpy as np
import pytest
from piece_geometry_fixtures import (
    PIECE_A_COLORS,
    PIECE_A_EDGE_TYPES,
    PIECE_B_COLORS,
    PIECE_B_EDGE_TYPES,
    deterministic_config,
    paint_quadrants,
    rasterize_piece,
)

from app.services.piece_geometry.contour import mask_to_contour
from app.services.piece_geometry.edges import split_edges
from app.services.piece_geometry.fingerprint import (
    build_fingerprint,
    chi_square_distance,
    shape_pair_distance,
    spatial_pair_distance,
)
from app.services.piece_geometry.scoring import FALLBACK_STATS, combined_z


def _build_fingerprint(edge_types: list[str], colors: list[tuple[int, int, int]]):
    """Build a PieceFingerprint for a synthetic piece."""
    config = deterministic_config(edge_types)
    mask, _ = rasterize_piece(config)
    contour = mask_to_contour(mask)
    assert contour is not None
    split = split_edges(contour)
    assert split is not None
    edges, _, _ = split
    image_bgr = paint_quadrants(mask, colors)
    return build_fingerprint(edges, image_bgr, contour)


class TestChiSquareDistance:
    """Tests for chi_square_distance."""

    def test_identical_histograms_have_zero_distance(self) -> None:
        """Comparing a histogram to itself yields distance 0."""
        hist = np.array([0.5, 0.3, 0.2])

        assert chi_square_distance(hist, hist) == pytest.approx(0.0)

    def test_disjoint_histograms_have_max_distance(self) -> None:
        """Completely disjoint L1-normalized histograms yield distance 1."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])

        assert chi_square_distance(p, q) == pytest.approx(1.0)


class TestFingerprintSelfConsistency:
    """Same-piece vs different-piece fingerprint distance and z-score behavior."""

    def test_same_piece_photographed_twice_has_low_shape_distance(self) -> None:
        """Two fingerprints built from the identical piece/photo have ~zero shape distance."""
        fp1 = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp2 = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)

        d_shape, k = shape_pair_distance(fp1.edges_canonical, fp1.edge_types, fp2.edges_canonical, fp2.edge_types)

        assert d_shape == pytest.approx(0.0, abs=1e-9)
        assert k == 0

    def test_different_pieces_have_higher_shape_distance(self) -> None:
        """Two geometrically different pieces have a larger shape distance than a self-comparison."""
        fp_a = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp_a2 = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp_b = _build_fingerprint(PIECE_B_EDGE_TYPES, PIECE_B_COLORS)

        d_self, _ = shape_pair_distance(fp_a.edges_canonical, fp_a.edge_types, fp_a2.edges_canonical, fp_a2.edge_types)
        d_other, _ = shape_pair_distance(fp_a.edges_canonical, fp_a.edge_types, fp_b.edges_canonical, fp_b.edge_types)

        assert d_other > d_self

    def test_combined_z_separates_genuine_from_impostor(self) -> None:
        """z-score for a genuine (self) re-scan is far below the impostor z, using the M7 fallback stats.

        This mirrors the store's own accept/new decision inputs: genuine
        z should land comfortably under `t_accept` and the impostor z should
        land comfortably above `t_new` (see app.config.Settings).
        """
        fp_a = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp_a2 = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp_b = _build_fingerprint(PIECE_B_EDGE_TYPES, PIECE_B_COLORS)

        d_shape_self, k_self = shape_pair_distance(
            fp_a.edges_canonical, fp_a.edge_types, fp_a2.edges_canonical, fp_a2.edge_types
        )
        d_spatial_self = spatial_pair_distance(fp_a, fp_a2, k_self)
        z_self = combined_z(d_shape_self, d_spatial_self, FALLBACK_STATS)

        d_shape_other, k_other = shape_pair_distance(
            fp_a.edges_canonical, fp_a.edge_types, fp_b.edges_canonical, fp_b.edge_types
        )
        d_spatial_other = spatial_pair_distance(fp_a, fp_b, k_other)
        z_other = combined_z(d_shape_other, d_spatial_other, FALLBACK_STATS)

        assert z_self < -4.78  # t_accept default
        assert z_other > -0.80  # t_new default
        assert z_self < z_other


class TestSpatialColorDescriptor:
    """Tests for the 3x3 spatial gray-world a*b* color descriptor."""

    def test_quadrant_painted_piece_has_nonempty_cells(self) -> None:
        """A piece painted with distinct quadrant colors has several non-empty spatial cells."""
        fp = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)

        assert fp.spatial_hists.shape == (9, 64)
        assert fp.spatial_nonempty.sum() >= 4

    def test_spatial_distance_to_self_is_near_zero(self) -> None:
        """The spatial color distance from a piece to an identical re-photograph is ~0."""
        fp1 = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)
        fp2 = _build_fingerprint(PIECE_A_EDGE_TYPES, PIECE_A_COLORS)

        distance = spatial_pair_distance(fp1, fp2, k=0)

        assert distance == pytest.approx(0.0, abs=1e-6)
