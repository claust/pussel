"""Tests for app.services.piece_geometry.contour."""

import cv2
import numpy as np
import pytest
from piece_geometry_fixtures import deterministic_config, rasterize_piece

from app.services.piece_geometry.contour import (
    _gaussian_kernel1d,
    alpha_to_mask,
    contour_quality,
    gaussian_filter1d_wrap,
    mask_to_contour,
    resample_contour,
    resample_polyline,
    smooth_contour,
)


class TestGaussianFilter1dWrap:
    """Equivalence tests for the numpy circular-Gaussian replacement of scipy's gaussian_filter1d."""

    def test_kernel_is_normalized_and_symmetric(self) -> None:
        """The hand-built kernel sums to 1 and is symmetric about its center."""
        kernel = _gaussian_kernel1d(sigma=2.0)

        assert kernel.sum() == pytest.approx(1.0)
        assert np.allclose(kernel, kernel[::-1])

    def test_impulse_response_matches_hand_computed_kernel(self) -> None:
        """Filtering a unit impulse reproduces the kernel weights exactly (hand-computed case).

        scipy.ndimage.gaussian_filter1d(impulse, sigma, mode="wrap") returns
        the kernel itself (circularly shifted to the impulse's position) —
        this is the defining property of a (circular) convolution/correlation
        with a delta function. We verify the numpy replacement against an
        independently hand-computed kernel (not by importing scipy, which the
        backend does not depend on).
        """
        sigma = 1.5
        n = 16
        impulse = np.zeros(n)
        impulse[0] = 1.0

        filtered = gaussian_filter1d_wrap(impulse, sigma=sigma)

        # Hand-computed normalized Gaussian kernel, truncate=4.0 (scipy's default):
        # radius = int(4.0 * 1.5 + 0.5) = 6.
        radius = int(4.0 * sigma + 0.5)
        offsets = np.arange(-radius, radius + 1)
        raw = np.exp(-0.5 * (offsets / sigma) ** 2)
        expected_kernel = raw / raw.sum()

        # The impulse sits at index 0, so filtered[j] = kernel[radius - j] wrapped,
        # i.e. filtered[(-offset) % n] == expected_kernel[radius + offset].
        for offset, weight in zip(offsets, expected_kernel):
            assert filtered[(-offset) % n] == pytest.approx(weight)

    def test_constant_signal_is_unchanged(self) -> None:
        """A constant periodic signal is a fixed point of the filter (kernel sums to 1)."""
        values = np.full(32, 7.0)

        filtered = gaussian_filter1d_wrap(values, sigma=2.0)

        assert np.allclose(filtered, 7.0)

    def test_smooths_a_noisy_periodic_signal(self) -> None:
        """Smoothing reduces sample-to-sample variation on a noisy periodic signal."""
        rng = np.random.default_rng(0)
        n = 200
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        clean = np.sin(t)
        noisy = clean + rng.normal(0, 0.5, size=n)

        filtered = gaussian_filter1d_wrap(noisy, sigma=3.0)

        noisy_roughness = np.sum(np.diff(noisy, append=noisy[:1]) ** 2)
        filtered_roughness = np.sum(np.diff(filtered, append=filtered[:1]) ** 2)
        assert filtered_roughness < noisy_roughness


class TestSmoothContour:
    """Tests for smooth_contour."""

    def test_preserves_length_and_shape(self) -> None:
        """Smoothing a contour keeps its point count and roughly its shape."""
        theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        contour = np.column_stack([50 * np.cos(theta) + 100, 50 * np.sin(theta) + 100])

        smoothed = smooth_contour(contour, sigma=2.0)

        assert smoothed.shape == contour.shape
        # A circle is already smooth: filtering shouldn't move points much.
        assert np.allclose(smoothed, contour, atol=1.5)


class TestMaskToContour:
    """Tests for mask_to_contour."""

    def test_single_blob_returns_contour(self) -> None:
        """A single filled rectangle yields a non-empty contour."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        contour = mask_to_contour(mask)

        assert contour is not None
        assert contour.shape[1] == 2
        assert len(contour) > 0

    def test_empty_mask_returns_none(self) -> None:
        """An all-background mask yields no contour."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        assert mask_to_contour(mask) is None

    def test_picks_largest_of_multiple_blobs(self) -> None:
        """With two separate blobs, the larger one's contour is returned."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # small blob, 20x20
        mask[80:180, 60:190] = 255  # large blob, 100x130

        contour = mask_to_contour(mask)

        assert contour is not None
        x, y, w, h = int(contour[:, 0].min()), int(contour[:, 1].min()), 0, 0
        w = int(contour[:, 0].max()) - x
        h = int(contour[:, 1].max()) - y
        assert (x, y) == (60, 80)
        assert abs(w - 129) <= 1 and abs(h - 99) <= 1


class TestAlphaToMask:
    """Tests for alpha_to_mask."""

    def test_hardens_soft_alpha(self) -> None:
        """Alpha above the threshold becomes 255, at/below becomes 0."""
        alpha = np.array([0, 100, 128, 129, 255], dtype=np.uint8)

        mask = alpha_to_mask(alpha)

        assert list(mask) == [0, 0, 0, 255, 255]


class TestResample:
    """Tests for resample_contour and resample_polyline."""

    def test_resample_contour_preserves_point_count_and_covers_shape(self) -> None:
        """Resampling a square contour to N points keeps the square's bounding box."""
        contour = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])

        resampled = resample_contour(contour, 40)

        assert resampled.shape == (40, 2)
        assert resampled[:, 0].min() >= -1e-6
        assert resampled[:, 0].max() <= 100 + 1e-6

    def test_resample_polyline_keeps_endpoints(self) -> None:
        """An open polyline's resampled endpoints match the input endpoints."""
        points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])

        resampled = resample_polyline(points, 10)

        assert np.allclose(resampled[0], points[0])
        assert np.allclose(resampled[-1], points[-1])
        assert resampled.shape == (10, 2)


class TestContourQuality:
    """Tests for contour_quality using synthetic puzzle_shapes pieces and hand-built masks."""

    def test_clean_piece_passes(self) -> None:
        """A single well-formed synthetic piece contour passes the quality gate."""
        config = deterministic_config(["tab", "blank", "flat", "tab"])
        mask, _ = rasterize_piece(config)
        contour = mask_to_contour(mask)
        assert contour is not None

        quality = contour_quality(contour, mask, mask.shape)

        assert quality.is_clean is True
        assert quality.n_large_components == 1
        assert quality.border_touching is False

    def test_two_blob_mask_fails_is_clean(self) -> None:
        """A mask with two comparably-sized components fails the single-component gate."""
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[20:140, 20:140] = 255  # ~120x120 blob
        mask[160:280, 160:280] = 255  # ~120x120 blob, same size -> both "large"
        contour = mask_to_contour(mask)
        assert contour is not None

        quality = contour_quality(contour, mask, mask.shape)

        assert quality.n_large_components == 2
        assert quality.is_clean is False

    def test_border_touching_fails_gate(self) -> None:
        """A contour touching the crop edge fails the border gate."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:80, 0:80] = 255  # touches the top-left border
        contour = mask_to_contour(mask)
        assert contour is not None

        quality = contour_quality(contour, mask, mask.shape)

        assert quality.border_touching is True
        assert quality.is_clean is False

    def test_low_solidity_fails_gate(self) -> None:
        """A ragged, star-shaped contour with low solidity fails the gate."""
        center = (100, 100)
        outer, inner = 90, 20
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        radii = np.where(np.arange(16) % 2 == 0, outer, inner)
        points = np.column_stack([center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)]).astype(
            np.int32
        )
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.fillPoly(mask, [points], (255,))
        contour = mask_to_contour(mask)
        assert contour is not None

        quality = contour_quality(contour, mask, mask.shape)

        assert quality.solidity < 0.6
        assert quality.is_clean is False
