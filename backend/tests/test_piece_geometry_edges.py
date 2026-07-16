"""Tests for app.services.piece_geometry.edges."""

from collections import Counter

import numpy as np
import pytest
from piece_geometry_fixtures import deterministic_config, rasterize_piece

from app.services.piece_geometry.contour import mask_to_contour
from app.services.piece_geometry.edges import canonicalize_edge, classify_arc, split_edges


class TestClassifyArc:
    """Unit tests for classify_arc's chord-deviation classification."""

    def test_straight_arc_is_flat(self) -> None:
        """An arc lying exactly on its chord is classified flat."""
        arc = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        centroid = np.array([5.0, -10.0])  # "inside" the piece, above the arc

        edge_type, dominant_dev, _, _, chord_length = classify_arc(arc, centroid)

        assert edge_type == "flat"
        assert dominant_dev == pytest.approx(0.0)
        assert chord_length == pytest.approx(10.0)

    def test_bulge_away_from_centroid_is_tab(self) -> None:
        """An arc that bulges away from the piece centroid is a tab."""
        arc = np.array([[0.0, 0.0], [5.0, -5.0], [10.0, 0.0]])
        centroid = np.array([5.0, 10.0])  # centroid is on the +y side; bulge is -y (away)

        edge_type, dominant_dev, _, _, _ = classify_arc(arc, centroid)

        assert edge_type == "tab"
        assert dominant_dev > 0

    def test_dip_toward_centroid_is_blank(self) -> None:
        """An arc that dips toward the piece centroid is a blank."""
        arc = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
        centroid = np.array([5.0, 10.0])  # centroid is on the +y side; dip is +y (toward)

        edge_type, dominant_dev, _, _, _ = classify_arc(arc, centroid)

        assert edge_type == "blank"
        assert dominant_dev < 0


class TestCanonicalizeEdge:
    """Tests for canonicalize_edge."""

    def test_endpoints_map_to_origin_and_unit_x(self) -> None:
        """Corner A maps to (0,0) and corner B maps to (1,0), regardless of original frame."""
        polyline = np.array([[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]])

        canonical = canonicalize_edge(polyline)

        assert np.allclose(canonical[0], [0.0, 0.0])
        assert np.allclose(canonical[-1], [1.0, 0.0])

    def test_degenerate_zero_length_chord_returns_zeros(self) -> None:
        """A polyline whose endpoints coincide returns an all-zero canonical polyline."""
        polyline = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])

        canonical = canonicalize_edge(polyline)

        assert np.allclose(canonical, 0.0)


class TestSplitEdges:
    """End-to-end edge classification on synthetic puzzle_shapes pieces."""

    @pytest.mark.parametrize(
        "edge_types",
        [
            ["tab", "flat", "blank", "flat"],
            ["tab", "blank", "tab", "blank"],
            ["flat", "flat", "tab", "blank"],
        ],
    )
    def test_detected_type_multiset_matches_configured_signature(self, edge_types: list[str]) -> None:
        """The set of detected edge types (counts) matches the configured signature.

        Corner ordering/winding normalization means the detected edge at
        index 0 does not necessarily correspond to the configured edge at
        index 0, so this compares multisets of types rather than a
        positional mapping (still a precise correctness check: it verifies
        exactly how many tabs/blanks/flats were found).
        """
        config = deterministic_config(edge_types)
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        result = split_edges(contour)

        assert result is not None
        edges, _, disagreement = result
        assert disagreement is False
        assert len(edges) == 4
        assert Counter(e.edge_type for e in edges) == Counter(edge_types)

    def test_edges_carry_100_point_polylines_and_canonical_frames(self) -> None:
        """Each edge has a 100-point raw polyline and a chord-normalized canonical polyline."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        result = split_edges(contour)

        assert result is not None
        edges, _, _ = result
        for edge in edges:
            assert edge.polyline.shape == (100, 2)
            assert edge.canonical_polyline.shape == (100, 2)
            assert np.allclose(edge.canonical_polyline[0], [0.0, 0.0])
            assert np.allclose(edge.canonical_polyline[-1], [1.0, 0.0])

    def test_too_few_contour_points_returns_none(self) -> None:
        """A contour with too few points for corner detection fails to split cleanly."""
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])

        result = split_edges(triangle)

        assert result is None
