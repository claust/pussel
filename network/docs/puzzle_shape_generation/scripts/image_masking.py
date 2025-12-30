"""Image masking and cutting utilities for puzzle piece extraction.

This module provides functions to cut puzzle pieces from source images
using polygon masks generated from Bezier curve boundaries.
"""

from dataclasses import dataclass
from typing import List, Tuple

from edge_grid import EdgeGrid, get_piece_curves
from models import BezierCurve
from PIL import Image, ImageDraw


@dataclass
class CoordinateMapper:
    """Maps between piece coordinates and image pixel coordinates.

    Piece coordinates are normalized (0-1) within each piece.
    Image coordinates are absolute pixel positions.
    """

    image_width: int
    image_height: int
    rows: int
    cols: int

    @property
    def piece_width(self) -> float:
        """Width of each piece in pixels."""
        return self.image_width / self.cols

    @property
    def piece_height(self) -> float:
        """Height of each piece in pixels."""
        return self.image_height / self.rows

    def piece_to_image(
        self,
        row: int,
        col: int,
        norm_x: float,
        norm_y: float,
    ) -> Tuple[float, float]:
        """Convert normalized piece coordinates to image pixel coordinates.

        Args:
            row: Piece row index (0-indexed from top).
            col: Piece column index (0-indexed from left).
            norm_x: X coordinate within piece (0=left, 1=right).
            norm_y: Y coordinate within piece (0=bottom, 1=top in piece space).

        Returns:
            Tuple of (pixel_x, pixel_y) in image coordinates.
            Note: Image Y is inverted (0=top, height=bottom).
        """
        # Piece origin in image space (top-left corner of piece)
        piece_origin_x = col * self.piece_width
        piece_origin_y = row * self.piece_height

        # Convert normalized piece coords to image coords
        # In piece space: y=0 is bottom, y=1 is top
        # In image space: y=0 is top, y=height is bottom
        img_x = piece_origin_x + norm_x * self.piece_width
        img_y = piece_origin_y + (1.0 - norm_y) * self.piece_height

        return (img_x, img_y)


def sample_curve_points(
    curve: BezierCurve,
    num_points: int = 20,
) -> List[Tuple[float, float]]:
    """Sample points from a Bezier curve.

    Args:
        curve: The Bezier curve to sample.
        num_points: Number of points to generate.

    Returns:
        List of (x, y) coordinate tuples.
    """
    points = curve.get_points(num_points)
    return [(float(p[0]), float(p[1])) for p in points]


def generate_piece_polygon(
    edge_grid: EdgeGrid,
    mapper: CoordinateMapper,
    row: int,
    col: int,
    points_per_curve: int = 20,
) -> List[Tuple[float, float]]:
    """Generate the boundary polygon for a piece in image coordinates.

    Args:
        edge_grid: The edge grid containing all puzzle edges.
        mapper: Coordinate mapper for this puzzle.
        row: Piece row index.
        col: Piece column index.
        points_per_curve: Number of points to sample from each Bezier curve.

    Returns:
        List of (x, y) points in image pixel coordinates forming a closed polygon.
    """
    curves = get_piece_curves(edge_grid, row, col)

    polygon: List[Tuple[float, float]] = []

    for curve in curves:
        points = sample_curve_points(curve, points_per_curve)
        for norm_x, norm_y in points[:-1]:  # Skip last to avoid duplication
            img_x, img_y = mapper.piece_to_image(row, col, norm_x, norm_y)
            polygon.append((img_x, img_y))

    # Close the polygon
    if polygon:
        polygon.append(polygon[0])

    return polygon


def create_piece_mask(
    polygon: List[Tuple[float, float]],
    width: int,
    height: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    antialias_scale: int = 4,
) -> Image.Image:
    """Create a mask image for a puzzle piece with anti-aliased edges.

    Uses supersampling for anti-aliasing: renders at higher resolution
    then downsamples.

    Args:
        polygon: List of (x, y) points in image coordinates.
        width: Output mask width in pixels.
        height: Output mask height in pixels.
        offset_x: X offset to subtract from polygon coordinates.
        offset_y: Y offset to subtract from polygon coordinates.
        antialias_scale: Supersampling factor (4 = render at 4x, then downsample).

    Returns:
        Grayscale PIL Image where white=inside, black=outside.
    """
    # Create high-res mask
    hi_res_width = width * antialias_scale
    hi_res_height = height * antialias_scale
    hi_res_mask = Image.new("L", (hi_res_width, hi_res_height), 0)
    draw = ImageDraw.Draw(hi_res_mask)

    # Scale polygon points to high-res, applying offset
    scaled_polygon = [((x - offset_x) * antialias_scale, (y - offset_y) * antialias_scale) for x, y in polygon]

    # Draw filled polygon
    if len(scaled_polygon) >= 3:
        draw.polygon(scaled_polygon, fill=255)

    # Downsample with anti-aliasing
    mask = hi_res_mask.resize((width, height), Image.Resampling.LANCZOS)

    return mask


def calculate_piece_bounds(
    polygon: List[Tuple[float, float]],
    padding: int = 20,
    image_width: int = 0,
    image_height: int = 0,
) -> Tuple[int, int, int, int]:
    """Calculate the bounding box of a piece polygon with padding.

    Args:
        polygon: List of (x, y) points.
        padding: Extra pixels to add around the bounding box.
        image_width: Source image width (for clamping). 0 = no clamping.
        image_height: Source image height (for clamping). 0 = no clamping.

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in integer pixels.
    """
    if not polygon:
        return (0, 0, 0, 0)

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]

    x_min = int(min(xs)) - padding
    y_min = int(min(ys)) - padding
    x_max = int(max(xs)) + padding + 1  # +1 to include the max pixel
    y_max = int(max(ys)) + padding + 1

    # Clamp to image bounds if specified
    if image_width > 0:
        x_min = max(0, x_min)
        x_max = min(image_width, x_max)
    if image_height > 0:
        y_min = max(0, y_min)
        y_max = min(image_height, y_max)

    return (x_min, y_min, x_max, y_max)


def cut_piece(
    source_image: Image.Image,
    polygon: List[Tuple[float, float]],
    padding: int = 20,
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Cut a puzzle piece from the source image.

    Args:
        source_image: The full puzzle image (RGB or RGBA).
        polygon: List of (x, y) points in image coordinates defining the piece boundary.
        padding: Extra pixels around the piece bounding box for tab protrusions.

    Returns:
        Tuple of:
        - RGBA image of the piece with transparent background
        - (x_offset, y_offset) position of the piece's top-left corner in the original image
    """
    # Calculate bounding box
    x_min, y_min, x_max, y_max = calculate_piece_bounds(
        polygon,
        padding=padding,
        image_width=source_image.width,
        image_height=source_image.height,
    )

    crop_width = x_max - x_min
    crop_height = y_max - y_min

    if crop_width <= 0 or crop_height <= 0:
        # Return empty image
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), (x_min, y_min)

    # Crop region from source
    cropped = source_image.crop((x_min, y_min, x_max, y_max))

    # Ensure RGB mode for processing
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")

    # Create anti-aliased mask
    mask = create_piece_mask(
        polygon,
        width=crop_width,
        height=crop_height,
        offset_x=x_min,
        offset_y=y_min,
    )

    # Apply mask to create RGBA output
    result = Image.new("RGBA", (crop_width, crop_height), (0, 0, 0, 0))
    result.paste(cropped, mask=mask)

    return result, (x_min, y_min)


def cut_all_pieces(
    source_image: Image.Image,
    edge_grid: EdgeGrid,
    padding: int = 20,
    points_per_curve: int = 20,
) -> List[List[Tuple[Image.Image, Tuple[int, int]]]]:
    """Cut all puzzle pieces from a source image.

    Args:
        source_image: The full puzzle image.
        edge_grid: The edge grid defining piece boundaries.
        padding: Extra pixels around each piece.
        points_per_curve: Number of points to sample from each curve.

    Returns:
        2D list of (piece_image, (x_offset, y_offset)) tuples indexed by [row][col].
    """
    mapper = CoordinateMapper(
        image_width=source_image.width,
        image_height=source_image.height,
        rows=edge_grid.rows,
        cols=edge_grid.cols,
    )

    pieces: List[List[Tuple[Image.Image, Tuple[int, int]]]] = []

    for row in range(edge_grid.rows):
        row_pieces: List[Tuple[Image.Image, Tuple[int, int]]] = []
        for col in range(edge_grid.cols):
            polygon = generate_piece_polygon(
                edge_grid,
                mapper,
                row,
                col,
                points_per_curve=points_per_curve,
            )
            piece, offset = cut_piece(source_image, polygon, padding=padding)
            row_pieces.append((piece, offset))
        pieces.append(row_pieces)

    return pieces
