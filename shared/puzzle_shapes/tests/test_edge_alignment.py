"""Tests for puzzle piece edge alignment.

These tests verify that adjacent puzzle pieces share edges correctly,
with tabs fitting perfectly into blanks and no gaps between pieces.
"""

import numpy as np
import pytest
from PIL import Image

from puzzle_shapes import CoordinateMapper, create_piece_mask, generate_edge_grid, generate_piece_polygon


def create_colored_piece(
    mask: Image.Image,
    color: tuple[int, int, int],
    alpha: int = 128,
) -> Image.Image:
    """Create a semi-transparent colored piece from a mask.

    Args:
        mask: Grayscale mask image (white=piece, black=background).
        color: RGB color tuple.
        alpha: Alpha value (0-255).

    Returns:
        RGBA image with colored piece.
    """
    mask_array = np.array(mask)
    h, w = mask_array.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    piece_pixels = mask_array > 0
    rgba[piece_pixels, 0] = color[0]
    rgba[piece_pixels, 1] = color[1]
    rgba[piece_pixels, 2] = color[2]
    rgba[piece_pixels, 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def generate_puzzle_overlay(
    rows: int,
    cols: int,
    piece_size: int = 100,
    padding: int = 20,
    seed: int = 42,
) -> Image.Image:
    """Generate a puzzle with alternating yellow/blue semi-transparent pieces.

    Args:
        rows: Number of rows.
        cols: Number of columns.
        piece_size: Size of each piece cell in pixels.
        padding: Padding around pieces for tabs.
        seed: Random seed for reproducibility.

    Returns:
        RGBA image with all pieces composited.
    """
    edge_grid = generate_edge_grid(rows, cols, seed=seed)
    mapper = CoordinateMapper(
        image_width=cols * piece_size,
        image_height=rows * piece_size,
        rows=rows,
        cols=cols,
    )

    canvas_w = cols * piece_size + 2 * padding
    canvas_h = rows * piece_size + 2 * padding
    output = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    yellow = (255, 255, 0)
    blue = (0, 0, 255)

    for r in range(rows):
        for c in range(cols):
            polygon = generate_piece_polygon(edge_grid, mapper, r, c, points_per_curve=50)

            # Calculate bounding box
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox_x1 = max(0, int(min(xs)) - padding)
            bbox_y1 = max(0, int(min(ys)) - padding)
            bbox_x2 = min(cols * piece_size, int(max(xs)) + padding)
            bbox_y2 = min(rows * piece_size, int(max(ys)) + padding)
            bbox_w = bbox_x2 - bbox_x1
            bbox_h = bbox_y2 - bbox_y1

            if bbox_w <= 0 or bbox_h <= 0:
                continue

            local_polygon = [(x - bbox_x1, y - bbox_y1) for x, y in polygon]
            mask = create_piece_mask(local_polygon, bbox_w, bbox_h)

            color = yellow if (r + c) % 2 == 0 else blue
            piece_img = create_colored_piece(mask, color, alpha=128)

            paste_x = bbox_x1 + padding
            paste_y = bbox_y1 + padding
            output.alpha_composite(piece_img, (paste_x, paste_y))

    return output


def analyze_puzzle_colors(img: Image.Image, padding: int = 20) -> dict:
    """Analyze pixel colors in puzzle image.

    Args:
        img: RGBA puzzle image.
        padding: Padding to exclude from analysis.

    Returns:
        Dictionary with color counts.
    """
    data = np.array(img)
    h, w = data.shape[:2]

    # Exclude padding region - analyze only the puzzle area
    inner = data[padding : h - padding, padding : w - padding]
    r, g, b = inner[:, :, 0], inner[:, :, 1], inner[:, :, 2]

    # Define color categories with tolerances
    # White (background/gaps): high R, G, B
    white_mask = (r > 250) & (g > 250) & (b > 250)

    # Green (yellow + blue overlap with equal contribution):
    # When 50% yellow (255,255,0) overlaps 50% blue (0,0,255), we get greenish
    # R ~ 128, G ~ 128, B ~ 128 or variations
    # More specifically: R and B should be similar and G should be high
    green_mask = (g > 100) & (np.abs(r.astype(int) - b.astype(int)) < 50) & ~white_mask

    return {
        "total_pixels": inner.shape[0] * inner.shape[1],
        "white_pixels": int(np.sum(white_mask)),
        "green_pixels": int(np.sum(green_mask)),
        "white_percentage": 100 * np.sum(white_mask) / (inner.shape[0] * inner.shape[1]),
        "green_percentage": 100 * np.sum(green_mask) / (inner.shape[0] * inner.shape[1]),
    }


class TestEdgeAlignment:
    """Tests for edge alignment between adjacent pieces."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_4x4_puzzle_no_gaps(self, seed: int) -> None:
        """Test that a 4x4 puzzle has no white gaps between pieces.

        A white pixel in the puzzle area indicates a gap where adjacent
        pieces don't meet properly.
        """
        img = generate_puzzle_overlay(rows=4, cols=4, piece_size=100, padding=30, seed=seed)
        analysis = analyze_puzzle_colors(img, padding=30)

        # Allow tiny amount of white at corners/edges due to anti-aliasing
        max_white_percentage = 0.1
        assert analysis["white_percentage"] < max_white_percentage, (
            f"Found {analysis['white_pixels']} white pixels ({analysis['white_percentage']:.2f}%) "
            f"indicating gaps between pieces (seed={seed})"
        )

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_4x4_puzzle_no_green_overlap(self, seed: int) -> None:
        """Test that a 4x4 puzzle has no green overlap between pieces.

        A green pixel indicates where yellow and blue pieces overlap,
        meaning pieces are intruding into each other's space.
        """
        img = generate_puzzle_overlay(rows=4, cols=4, piece_size=100, padding=30, seed=seed)
        analysis = analyze_puzzle_colors(img, padding=30)

        # Allow tiny amount of green at boundaries due to anti-aliasing
        max_green_percentage = 0.5
        assert analysis["green_percentage"] < max_green_percentage, (
            f"Found {analysis['green_pixels']} green pixels ({analysis['green_percentage']:.2f}%) "
            f"indicating piece overlap (seed={seed})"
        )

    @pytest.mark.parametrize(
        "rows,cols",
        [(2, 2), (3, 3), (4, 4), (5, 5), (2, 4), (4, 2)],
    )
    def test_various_grid_sizes_no_gaps(self, rows: int, cols: int) -> None:
        """Test that various grid sizes have no gaps."""
        img = generate_puzzle_overlay(rows=rows, cols=cols, piece_size=80, padding=25, seed=42)
        analysis = analyze_puzzle_colors(img, padding=25)

        max_white_percentage = 0.1
        assert analysis["white_percentage"] < max_white_percentage, (
            f"Found gaps in {rows}x{cols} grid: "
            f"{analysis['white_pixels']} white pixels ({analysis['white_percentage']:.2f}%)"
        )

    def test_horizontal_two_pieces_alignment(self) -> None:
        """Test that two horizontally adjacent pieces align perfectly."""
        img = generate_puzzle_overlay(rows=1, cols=2, piece_size=200, padding=50, seed=42)
        analysis = analyze_puzzle_colors(img, padding=50)

        # For just two pieces, should be essentially no white
        assert (
            analysis["white_percentage"] < 0.05
        ), f"Found gaps between two horizontal pieces: {analysis['white_pixels']} white pixels"

    def test_vertical_two_pieces_alignment(self) -> None:
        """Test that two vertically adjacent pieces align perfectly."""
        img = generate_puzzle_overlay(rows=2, cols=1, piece_size=200, padding=50, seed=42)
        analysis = analyze_puzzle_colors(img, padding=50)

        # For just two pieces, should be essentially no white
        assert (
            analysis["white_percentage"] < 0.05
        ), f"Found gaps between two vertical pieces: {analysis['white_pixels']} white pixels"
