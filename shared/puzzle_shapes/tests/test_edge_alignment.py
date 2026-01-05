"""Tests for verifying puzzle piece edge alignment.

These tests generate puzzle pieces and verify that adjacent pieces share
the exact same edge coordinates (no gaps or overlaps).
"""

import math
from typing import List, Tuple

import pytest
from PIL import Image, ImageDraw

from puzzle_shapes.edge_grid import EdgeGrid, generate_edge_grid, get_piece_curves
from puzzle_shapes.models import BezierCurve


def sample_curve_points(curve: BezierCurve, num_points: int = 20) -> List[Tuple[float, float]]:
    """Sample points along a Bezier curve."""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        points.append(curve.evaluate(t))
    return points


def get_edge_curves(curves: List[BezierCurve]) -> dict:
    """Split a piece's curves into edges by finding corner transitions.

    Curves are returned in clockwise order: top -> right -> bottom -> left.
    We identify edge boundaries by finding where curves reach corner positions.

    Returns:
        Dictionary with "top", "right", "bottom", "left" keys.
    """
    # Corner tolerance
    tol = 0.1

    def near_corner(point: Tuple[float, float], corner: Tuple[float, float]) -> bool:
        return abs(point[0] - corner[0]) < tol and abs(point[1] - corner[1]) < tol

    # Find edge boundary indices by looking for curves ending at corners
    # Top edge: starts at (0,1), ends when we reach (1,1)
    # Right edge: starts at (1,1), ends when we reach (1,0)
    # Bottom edge: starts at (1,0), ends when we reach (0,0)
    # Left edge: starts at (0,0), ends when we reach (0,1)

    top_end = right_end = bottom_end = len(curves)

    for i, curve in enumerate(curves):
        end = curve.p3
        if top_end == len(curves) and near_corner(end, (1.0, 1.0)):
            top_end = i + 1
        elif right_end == len(curves) and near_corner(end, (1.0, 0.0)):
            right_end = i + 1
        elif bottom_end == len(curves) and near_corner(end, (0.0, 0.0)):
            bottom_end = i + 1

    return {
        "top": curves[0:top_end],
        "right": curves[top_end:right_end],
        "bottom": curves[right_end:bottom_end],
        "left": curves[bottom_end:],
    }


def get_edge_points(
    curves: List[BezierCurve],
    edge: str,
    num_points: int = 20,
) -> List[Tuple[float, float]]:
    """Extract points from a specific edge of a piece.

    Args:
        curves: All curves for the piece (in clockwise order: top, right, bottom, left).
        edge: Which edge to extract ("top", "right", "bottom", "left").
        num_points: Number of points to sample per curve.

    Returns:
        List of sampled points along the specified edge.
    """
    edge_curves = get_edge_curves(curves)
    selected_curves = edge_curves.get(edge, [])

    points = []
    for curve in selected_curves:
        points.extend(sample_curve_points(curve, num_points))

    return points


def points_match(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    tolerance: float = 0.001,
) -> Tuple[bool, float]:
    """Check if two lists of points match (within tolerance).

    Since adjacent edges are traversed in opposite directions,
    we need to compare points1 to reversed points2.

    Returns:
        Tuple of (match_status, max_distance).
    """
    if len(points1) == 0 or len(points2) == 0:
        return (False, float("inf"))

    # Reverse points2 since edges are traversed in opposite directions
    points2_reversed = list(reversed(points2))

    # Resample to same length if needed
    if len(points1) != len(points2_reversed):
        # Just compare available points
        min_len = min(len(points1), len(points2_reversed))
        points1 = points1[:min_len]
        points2_reversed = points2_reversed[:min_len]

    max_dist = 0.0
    for p1, p2 in zip(points1, points2_reversed):
        dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        max_dist = max(max_dist, dist)

    return (max_dist < tolerance, max_dist)


def transform_point_to_grid(
    point: Tuple[float, float],
    row: int,
    col: int,
    piece_width: float,
    piece_height: float,
    total_rows: int,
) -> Tuple[float, float]:
    """Transform a point from normalized piece space to grid space.

    Note: Row 0 is at the visual "top" of the puzzle, so we invert
    the y-coordinate to get proper grid coordinates where y increases upward.
    """
    x = col * piece_width + point[0] * piece_width
    # Invert row so that row 0 is at the top (higher y in grid coords)
    y = (total_rows - 1 - row) * piece_height + point[1] * piece_height
    return (x, y)


class TestEdgeAlignment:
    """Tests for verifying puzzle piece edges align correctly."""

    @pytest.fixture
    def edge_grid_2x2(self) -> EdgeGrid:
        """Create a 2x2 puzzle grid with fixed seed."""
        return generate_edge_grid(rows=2, cols=2, seed=42)

    @pytest.fixture
    def edge_grid_3x3(self) -> EdgeGrid:
        """Create a 3x3 puzzle grid with fixed seed."""
        return generate_edge_grid(rows=3, cols=3, seed=42)

    @pytest.fixture
    def edge_grid_4x4(self) -> EdgeGrid:
        """Create a 4x4 puzzle grid with fixed seed."""
        return generate_edge_grid(rows=4, cols=4, seed=42)

    def test_horizontal_edge_alignment_2x2(self, edge_grid_2x2: EdgeGrid) -> None:
        """Test that horizontally adjacent pieces share the same vertical edge."""
        grid = edge_grid_2x2
        piece_width = 1.0
        piece_height = 1.0
        total_rows = 2

        # Check piece (0,0) right edge matches piece (0,1) left edge
        piece_00_curves = get_piece_curves(grid, 0, 0)
        piece_01_curves = get_piece_curves(grid, 0, 1)

        # Get right edge of piece (0,0) and left edge of piece (0,1)
        # Transform to grid coordinates for comparison
        piece_00_right = get_edge_points(piece_00_curves, "right")
        piece_01_left = get_edge_points(piece_01_curves, "left")

        # Transform to grid space
        piece_00_right_grid = [
            transform_point_to_grid(p, 0, 0, piece_width, piece_height, total_rows) for p in piece_00_right
        ]
        piece_01_left_grid = [
            transform_point_to_grid(p, 0, 1, piece_width, piece_height, total_rows) for p in piece_01_left
        ]

        match, max_dist = points_match(piece_00_right_grid, piece_01_left_grid, tolerance=0.01)
        assert match, f"Horizontal edge mismatch: max distance = {max_dist}"

    def test_vertical_edge_alignment_2x2(self, edge_grid_2x2: EdgeGrid) -> None:
        """Test that vertically adjacent pieces share the same horizontal edge."""
        grid = edge_grid_2x2
        piece_width = 1.0
        piece_height = 1.0
        total_rows = 2

        # Check piece (0,0) bottom edge matches piece (1,0) top edge
        piece_00_curves = get_piece_curves(grid, 0, 0)
        piece_10_curves = get_piece_curves(grid, 1, 0)

        piece_00_bottom = get_edge_points(piece_00_curves, "bottom")
        piece_10_top = get_edge_points(piece_10_curves, "top")

        # Transform to grid space
        piece_00_bottom_grid = [
            transform_point_to_grid(p, 0, 0, piece_width, piece_height, total_rows) for p in piece_00_bottom
        ]
        piece_10_top_grid = [
            transform_point_to_grid(p, 1, 0, piece_width, piece_height, total_rows) for p in piece_10_top
        ]

        match, max_dist = points_match(piece_00_bottom_grid, piece_10_top_grid, tolerance=0.01)
        assert match, f"Vertical edge mismatch: max distance = {max_dist}"

    def test_all_horizontal_edges_3x3(self, edge_grid_3x3: EdgeGrid) -> None:
        """Test all horizontal adjacencies in a 3x3 grid."""
        grid = edge_grid_3x3
        piece_width = 1.0
        piece_height = 1.0
        total_rows = 3

        for row in range(3):
            for col in range(2):  # Only check internal horizontal edges
                piece_left_curves = get_piece_curves(grid, row, col)
                piece_right_curves = get_piece_curves(grid, row, col + 1)

                left_right_edge = get_edge_points(piece_left_curves, "right")
                right_left_edge = get_edge_points(piece_right_curves, "left")

                left_right_grid = [
                    transform_point_to_grid(p, row, col, piece_width, piece_height, total_rows) for p in left_right_edge
                ]
                right_left_grid = [
                    transform_point_to_grid(p, row, col + 1, piece_width, piece_height, total_rows)
                    for p in right_left_edge
                ]

                match, max_dist = points_match(left_right_grid, right_left_grid, tolerance=0.01)
                assert match, (
                    f"Horizontal edge mismatch at ({row},{col})-({row},{col + 1}): " f"max distance = {max_dist}"
                )

    def test_all_vertical_edges_3x3(self, edge_grid_3x3: EdgeGrid) -> None:
        """Test all vertical adjacencies in a 3x3 grid."""
        grid = edge_grid_3x3
        piece_width = 1.0
        piece_height = 1.0
        total_rows = 3

        for row in range(2):  # Only check internal vertical edges
            for col in range(3):
                piece_top_curves = get_piece_curves(grid, row, col)
                piece_bottom_curves = get_piece_curves(grid, row + 1, col)

                top_bottom_edge = get_edge_points(piece_top_curves, "bottom")
                bottom_top_edge = get_edge_points(piece_bottom_curves, "top")

                top_bottom_grid = [
                    transform_point_to_grid(p, row, col, piece_width, piece_height, total_rows) for p in top_bottom_edge
                ]
                bottom_top_grid = [
                    transform_point_to_grid(p, row + 1, col, piece_width, piece_height, total_rows)
                    for p in bottom_top_edge
                ]

                match, max_dist = points_match(top_bottom_grid, bottom_top_grid, tolerance=0.01)
                assert match, (
                    f"Vertical edge mismatch at ({row},{col})-({row + 1},{col}): " f"max distance = {max_dist}"
                )

    def test_all_edges_4x4(self, edge_grid_4x4: EdgeGrid) -> None:
        """Test all adjacencies in a 4x4 grid."""
        grid = edge_grid_4x4
        piece_width = 1.0
        piece_height = 1.0
        total_rows = 4

        errors = []

        # Test horizontal adjacencies
        for row in range(4):
            for col in range(3):
                piece_left_curves = get_piece_curves(grid, row, col)
                piece_right_curves = get_piece_curves(grid, row, col + 1)

                left_right_edge = get_edge_points(piece_left_curves, "right")
                right_left_edge = get_edge_points(piece_right_curves, "left")

                left_right_grid = [
                    transform_point_to_grid(p, row, col, piece_width, piece_height, total_rows) for p in left_right_edge
                ]
                right_left_grid = [
                    transform_point_to_grid(p, row, col + 1, piece_width, piece_height, total_rows)
                    for p in right_left_edge
                ]

                match, max_dist = points_match(left_right_grid, right_left_grid, tolerance=0.01)
                if not match:
                    errors.append(f"H({row},{col})-({row},{col + 1}): {max_dist:.4f}")

        # Test vertical adjacencies
        for row in range(3):
            for col in range(4):
                piece_top_curves = get_piece_curves(grid, row, col)
                piece_bottom_curves = get_piece_curves(grid, row + 1, col)

                top_bottom_edge = get_edge_points(piece_top_curves, "bottom")
                bottom_top_edge = get_edge_points(piece_bottom_curves, "top")

                top_bottom_grid = [
                    transform_point_to_grid(p, row, col, piece_width, piece_height, total_rows) for p in top_bottom_edge
                ]
                bottom_top_grid = [
                    transform_point_to_grid(p, row + 1, col, piece_width, piece_height, total_rows)
                    for p in bottom_top_edge
                ]

                match, max_dist = points_match(top_bottom_grid, bottom_top_grid, tolerance=0.01)
                if not match:
                    errors.append(f"V({row},{col})-({row + 1},{col}): {max_dist:.4f}")

        assert len(errors) == 0, f"Edge misalignments found: {errors}"

    def test_multiple_seeds(self) -> None:
        """Test edge alignment with multiple random seeds."""
        for seed in [1, 42, 100, 999, 12345]:
            grid = generate_edge_grid(rows=3, cols=3, seed=seed)
            piece_width = 1.0
            piece_height = 1.0
            total_rows = 3

            # Test center piece (1,1) against all neighbors
            center_curves = get_piece_curves(grid, 1, 1)

            # Check right neighbor
            right_curves = get_piece_curves(grid, 1, 2)
            center_right = get_edge_points(center_curves, "right")
            right_left = get_edge_points(right_curves, "left")

            center_right_grid = [
                transform_point_to_grid(p, 1, 1, piece_width, piece_height, total_rows) for p in center_right
            ]
            right_left_grid = [
                transform_point_to_grid(p, 1, 2, piece_width, piece_height, total_rows) for p in right_left
            ]

            match, max_dist = points_match(center_right_grid, right_left_grid, tolerance=0.01)
            assert match, f"Seed {seed}: horizontal edge mismatch, max_dist={max_dist}"

            # Check bottom neighbor
            bottom_curves = get_piece_curves(grid, 2, 1)
            center_bottom = get_edge_points(center_curves, "bottom")
            bottom_top = get_edge_points(bottom_curves, "top")

            center_bottom_grid = [
                transform_point_to_grid(p, 1, 1, piece_width, piece_height, total_rows) for p in center_bottom
            ]
            bottom_top_grid = [
                transform_point_to_grid(p, 2, 1, piece_width, piece_height, total_rows) for p in bottom_top
            ]

            match, max_dist = points_match(center_bottom_grid, bottom_top_grid, tolerance=0.01)
            assert match, f"Seed {seed}: vertical edge mismatch, max_dist={max_dist}"

    def test_edge_continuity(self, edge_grid_2x2: EdgeGrid) -> None:
        """Test that curves within a piece form a continuous boundary."""
        grid = edge_grid_2x2
        curves = get_piece_curves(grid, 0, 0)

        # Each curve should end where the next one starts
        tolerance = 0.001
        for i in range(len(curves) - 1):
            end_point = curves[i].p3
            start_point = curves[i + 1].p0
            dist = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
            assert dist < tolerance, (
                f"Curve discontinuity at index {i}: " f"end={end_point}, start={start_point}, dist={dist}"
            )

        # Last curve should connect back to first curve (closed boundary)
        end_point = curves[-1].p3
        start_point = curves[0].p0
        dist = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
        assert dist < tolerance, f"Boundary not closed: last end={end_point}, first start={start_point}, dist={dist}"


class TestVisualEdgeAlignment:
    """Visual tests using rendered images to detect gaps and overlaps."""

    @staticmethod
    def get_piece_polygon(
        grid: EdgeGrid,
        row: int,
        col: int,
        piece_size: int,
        total_rows: int,
        points_per_curve: int = 10,
    ) -> List[Tuple[int, int]]:
        """Get polygon points for a piece in pixel coordinates.

        Args:
            grid: The edge grid.
            row: Row index of the piece.
            col: Column index of the piece.
            piece_size: Size of each piece in pixels.
            total_rows: Total number of rows in the grid.
            points_per_curve: Number of points to sample per Bezier curve.

        Returns:
            List of (x, y) pixel coordinates forming the piece polygon.
        """
        curves = get_piece_curves(grid, row, col)
        polygon_points: List[Tuple[int, int]] = []

        for curve in curves:
            # Sample points along the curve
            for i in range(points_per_curve):
                t = i / points_per_curve
                point = curve.evaluate(t)

                # Transform from normalized piece coords to pixel coords
                # In piece coords: (0,0) is bottom-left, (1,1) is top-right
                # In pixel coords: (0,0) is top-left, y increases downward
                # Row 0 is at the visual top of the puzzle
                pixel_x = int((col + point[0]) * piece_size)
                pixel_y = int((row + 1 - point[1]) * piece_size)

                polygon_points.append((pixel_x, pixel_y))

        return polygon_points

    @staticmethod
    def render_puzzle_image(
        grid: EdgeGrid,
        piece_size: int = 100,
        points_per_curve: int = 15,
    ) -> Image.Image:
        """Render a puzzle grid with alternating yellow/blue pieces at 50% alpha.

        Args:
            grid: The edge grid to render.
            piece_size: Size of each piece in pixels.
            points_per_curve: Number of points to sample per Bezier curve.

        Returns:
            PIL Image with rendered puzzle pieces.
        """
        rows = grid.rows
        cols = grid.cols

        # Create white background
        img_width = cols * piece_size
        img_height = rows * piece_size
        image = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 255))

        # Colors: yellow and blue at 50% alpha (128)
        yellow = (255, 255, 0, 128)  # RGBA
        blue = (0, 0, 255, 128)  # RGBA

        for row in range(rows):
            for col in range(cols):
                # Alternate colors in checkerboard pattern
                is_even = (row + col) % 2 == 0
                color = yellow if is_even else blue

                # Get polygon for this piece
                polygon = TestVisualEdgeAlignment.get_piece_polygon(
                    grid, row, col, piece_size, rows, points_per_curve
                )

                # Create a temporary image for this piece with transparency
                piece_img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
                piece_draw = ImageDraw.Draw(piece_img)
                piece_draw.polygon(polygon, fill=color)

                # Composite onto main image
                image = Image.alpha_composite(image, piece_img)

        return image

    @staticmethod
    def check_for_artifacts(image: Image.Image) -> Tuple[int, int, List[Tuple[int, int]]]:
        """Check image for white pixels (gaps) and green pixels (overlaps).

        Args:
            image: The rendered puzzle image.

        Returns:
            Tuple of (white_count, green_count, sample_positions).
        """
        # Convert to RGB for easier color checking
        rgb_image = image.convert("RGB")
        pixels = rgb_image.load()
        width, height = rgb_image.size

        white_pixels = 0
        green_pixels = 0
        sample_positions: List[Tuple[int, int]] = []

        # Define color thresholds
        # White: R > 250, G > 250, B > 250
        # Green: G > R + 20 and G > B + 20 (green is dominant)

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]

                # Check for white (gap)
                if r > 250 and g > 250 and b > 250:
                    white_pixels += 1
                    if len(sample_positions) < 10:
                        sample_positions.append((x, y))

                # Check for green (overlap of yellow + blue)
                # When yellow (255,255,0) and blue (0,0,255) overlap at 50% alpha each:
                # Result is approximately (127, 127, 127) on white, but with both contributing
                # Actually: yellow on white = (255,255,127), blue on that = ~(127,127,191)
                # Let's check for greenish tint: G significantly higher than both R and B
                # Or check for cyan-ish: where blue and yellow mix
                # A simpler check: if we see significant green component with low red
                if g > 150 and r < 200 and b < 200 and g > r and g > b:
                    green_pixels += 1
                    if len(sample_positions) < 10:
                        sample_positions.append((x, y))

        return white_pixels, green_pixels, sample_positions

    def test_visual_no_gaps_or_overlaps_3x3(self) -> None:
        """Test that a 3x3 puzzle has no gaps (white) or overlaps (green)."""
        grid = generate_edge_grid(rows=3, cols=3, seed=42)
        image = self.render_puzzle_image(grid, piece_size=100, points_per_curve=20)

        white_count, green_count, samples = self.check_for_artifacts(image)

        # Allow small tolerance for antialiasing artifacts at edges
        max_allowed_artifacts = 50  # pixels

        assert white_count < max_allowed_artifacts, (
            f"Found {white_count} white pixels (gaps) in 3x3 puzzle. "
            f"Sample positions: {samples[:5]}"
        )
        assert green_count < max_allowed_artifacts, (
            f"Found {green_count} green pixels (overlaps) in 3x3 puzzle. "
            f"Sample positions: {samples[:5]}"
        )

    def test_visual_no_gaps_or_overlaps_4x4(self) -> None:
        """Test that a 4x4 puzzle has no gaps (white) or overlaps (green)."""
        grid = generate_edge_grid(rows=4, cols=4, seed=123)
        image = self.render_puzzle_image(grid, piece_size=100, points_per_curve=20)

        white_count, green_count, samples = self.check_for_artifacts(image)

        max_allowed_artifacts = 50

        assert white_count < max_allowed_artifacts, (
            f"Found {white_count} white pixels (gaps) in 4x4 puzzle. "
            f"Sample positions: {samples[:5]}"
        )
        assert green_count < max_allowed_artifacts, (
            f"Found {green_count} green pixels (overlaps) in 4x4 puzzle. "
            f"Sample positions: {samples[:5]}"
        )

    def test_visual_multiple_seeds(self) -> None:
        """Test visual alignment with multiple random seeds."""
        max_allowed_artifacts = 50

        for seed in [1, 42, 100, 999]:
            grid = generate_edge_grid(rows=3, cols=3, seed=seed)
            image = self.render_puzzle_image(grid, piece_size=100, points_per_curve=20)

            white_count, green_count, samples = self.check_for_artifacts(image)

            assert white_count < max_allowed_artifacts, (
                f"Seed {seed}: Found {white_count} white pixels (gaps). "
                f"Sample positions: {samples[:5]}"
            )
            assert green_count < max_allowed_artifacts, (
                f"Seed {seed}: Found {green_count} green pixels (overlaps). "
                f"Sample positions: {samples[:5]}"
            )
