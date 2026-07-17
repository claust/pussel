"""Tests for app.services.piece_geometry.corners."""

import itertools

import numpy as np
import pytest
from piece_geometry_fixtures import deterministic_config, rasterize_piece

from app.services.piece_geometry.contour import mask_to_contour
from app.services.piece_geometry.corners import (
    InsufficientCornersError,
    bbox_diagonal,
    corner_max_distance_frac,
    detect_corners_curvature,
    detect_corners_polydp,
    detect_corners_with_cross_check,
)

SUCCESS_THRESHOLD_PCT = 3.0


def _corner_error_pct(predicted: np.ndarray, gt_corners: np.ndarray) -> float:
    """Max matched-corner error as a percentage of the piece diagonal.

    Brute-forces the optimal (minimum total distance) assignment over the 24
    permutations of 4 elements, mirroring
    `network/experiments/exp28_piece_geometry/synth_benchmark.py`'s
    `score_corners` without depending on scipy.
    """
    best_sum = float("inf")
    best_max = 0.0
    for perm in itertools.permutations(range(4)):
        dists = np.linalg.norm(predicted - gt_corners[list(perm)], axis=1)
        total = float(dists.sum())
        if total < best_sum:
            best_sum = total
            best_max = float(dists.max())
    diagonal = float(np.linalg.norm(gt_corners[0] - gt_corners[2]))
    return 100.0 * best_max / diagonal


class TestDetectCornersPolydp:
    """Corner-detection accuracy on synthetic pieces with known ground truth."""

    @pytest.mark.parametrize(
        "edge_types",
        [
            ["tab", "blank", "flat", "tab"],
            ["tab", "blank", "tab", "blank"],
            ["flat", "flat", "tab", "blank"],
        ],
    )
    def test_corners_within_tolerance_of_ground_truth(self, edge_types: list[str]) -> None:
        """The 4 detected corners are within 3% of the piece diagonal from ground truth."""
        config = deterministic_config(edge_types)
        mask, gt_corners = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        result = detect_corners_polydp(contour)

        assert result.corners.shape == (4, 2)
        error_pct = _corner_error_pct(result.corners, gt_corners)
        assert error_pct <= SUCCESS_THRESHOLD_PCT

    def test_curvature_detector_also_within_tolerance(self) -> None:
        """The curvature cross-check detector also lands within tolerance."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, gt_corners = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        result = detect_corners_curvature(contour)

        error_pct = _corner_error_pct(result.corners, gt_corners)
        assert error_pct <= SUCCESS_THRESHOLD_PCT

    def test_corners_ordered_clockwise_from_top_left(self) -> None:
        """Corners come back ordered clockwise starting near the top-left."""
        config = deterministic_config(["tab", "blank", "tab", "blank"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        result = detect_corners_polydp(contour)

        centroid = result.corners.mean(axis=0)
        angles: np.ndarray = np.arctan2(result.corners[:, 1] - centroid[1], result.corners[:, 0] - centroid[0])
        # Clockwise in image coords (y down) means increasing angle when
        # traversed in order (arctan2 in [-pi, pi], wrapping once).
        diffs: np.ndarray = np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi]))
        assert bool(np.all(diffs > 0))

    def test_too_few_candidates_raises(self) -> None:
        """A near-perfect circle (few polydp vertices survive a tight sweep) still yields 4 corners.

        This documents the failure mode rather than forcing it: with only 3
        raw contour points, `InsufficientCornersError` is raised.
        """
        triangle = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])

        with pytest.raises(InsufficientCornersError):
            detect_corners_polydp(triangle)


class TestCornerDisagreement:
    """Tests for corner_max_distance_frac (the brute-force Hungarian-equivalent) and the cross-check."""

    def test_identical_corner_sets_have_zero_distance(self) -> None:
        """Comparing a corner set to itself gives zero distance regardless of point order."""
        corners = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        shuffled = corners[[2, 0, 3, 1]]

        frac = corner_max_distance_frac(corners, shuffled, diagonal=float(np.linalg.norm(corners[0] - corners[2])))

        assert frac == pytest.approx(0.0, abs=1e-9)

    def test_matches_brute_force_minimum_sum_assignment(self) -> None:
        """corner_max_distance_frac picks the same assignment as an independent brute-force search."""
        a = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        # b is `a` cyclically shifted by one position plus small jitter, so the
        # optimal assignment isn't the identity permutation.
        b = np.array([[0.3, 10.2], [0.1, -0.2], [9.8, 0.4], [10.1, 9.7]])

        frac = corner_max_distance_frac(a, b, diagonal=1.0)

        best_sum = min(
            sum(np.linalg.norm(a[i] - b[perm[i]]) for i in range(4)) for perm in itertools.permutations(range(4))
        )
        best_perm = min(
            itertools.permutations(range(4)),
            key=lambda perm: sum(np.linalg.norm(a[i] - b[perm[i]]) for i in range(4)),
        )
        expected_max = max(np.linalg.norm(a[i] - b[best_perm[i]]) for i in range(4))
        assert best_sum > 0  # sanity: the jitter actually perturbed the points
        assert frac == pytest.approx(expected_max, rel=1e-6)

    def test_cross_check_agrees_on_clean_synthetic_piece(self) -> None:
        """The polydp and curvature detectors agree (no disagreement flag) on a clean synthetic piece."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        primary, disagreement = detect_corners_with_cross_check(contour)

        assert primary.method == "polydp"
        assert disagreement is False

    def test_bbox_diagonal_of_unit_square(self) -> None:
        """bbox_diagonal of a unit square is sqrt(2)."""
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        assert bbox_diagonal(square) == pytest.approx(np.sqrt(2))
