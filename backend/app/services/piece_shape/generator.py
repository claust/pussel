"""Generate realistic puzzle pieces from images.

This module provides the PieceShapeGenerator class that extracts jigsaw-shaped
pieces from puzzle images using Bezier curve-based masks.
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from puzzle_shapes import PieceConfig, generate_piece_path


class PieceShapeGenerator:
    """Generate jigsaw-shaped pieces from puzzle images."""

    def __init__(self, piece_size_ratio: float = 0.25):
        """Initialize the generator.

        Args:
            piece_size_ratio: Size of piece relative to puzzle dimensions (0.2-0.35 typical).
                The piece will be approximately this fraction of the smaller dimension.
        """
        self.piece_size_ratio = piece_size_ratio

    def _create_piece_mask(
        self,
        config: PieceConfig,
        output_size: Tuple[int, int],
        piece_center: Tuple[float, float],
        piece_size: float,
    ) -> np.ndarray:
        """Create a binary mask for the piece shape.

        Args:
            config: Piece configuration with edge types and parameters.
            output_size: (width, height) of the mask.
            piece_center: (x, y) center of piece in the output image.
            piece_size: Size of the piece in pixels.

        Returns:
            Binary mask as numpy array (255 for piece, 0 for background).
        """
        width, height = output_size

        # Generate piece path in normalized coordinates (0 to config.size)
        path_x, path_y = generate_piece_path(config, points_per_curve=50)

        # The piece path is in [0, config.size] coordinates with tabs/blanks extending
        # beyond (~35% for tabs). Scale to fit within the output size.
        scale = piece_size / config.size

        # Calculate offset to center the piece
        # The path coordinates are relative to bottom-left of the piece base square
        offset_x = piece_center[0] - (config.size / 2) * scale
        offset_y = piece_center[1] - (config.size / 2) * scale

        # Transform path points to pixel coordinates
        points = []
        for x, y in zip(path_x, path_y):
            px = int(x * scale + offset_x)
            py = int(y * scale + offset_y)
            points.append((px, py))

        # Create mask using PIL polygon fill
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon(points, fill=255)

        return np.array(mask_img)

    def generate_piece(
        self,
        puzzle_image: Image.Image,
        center_x: float,
        center_y: float,
        config: Optional[PieceConfig] = None,
    ) -> Tuple[Image.Image, PieceConfig]:
        """Extract a jigsaw-shaped piece from puzzle image.

        Args:
            puzzle_image: The full puzzle image (PIL Image).
            center_x: Normalized x coordinate of piece center (0-1).
            center_y: Normalized y coordinate of piece center (0-1).
            config: Optional piece configuration (random if None).

        Returns:
            Tuple of (RGBA piece image with transparency, config used).
        """
        if config is None:
            config = PieceConfig.random()

        width, height = puzzle_image.size

        # Calculate piece size in pixels
        piece_size = int(min(width, height) * self.piece_size_ratio)

        # Margin for tabs extending beyond the base square
        margin = int(0.40 * piece_size)
        total_size = piece_size + 2 * margin

        # Calculate extraction bounds centered at click position
        cx = int(center_x * width)
        cy = int(center_y * height)

        x1 = cx - total_size // 2
        y1 = cy - total_size // 2
        x2 = x1 + total_size
        y2 = y1 + total_size

        # Handle edge cases (piece near puzzle border) by clamping and padding
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - width)
        pad_bottom = max(0, y2 - height)

        x1_clamped = max(0, x1)
        y1_clamped = max(0, y1)
        x2_clamped = min(width, x2)
        y2_clamped = min(height, y2)

        # Extract region from puzzle
        region = puzzle_image.crop((x1_clamped, y1_clamped, x2_clamped, y2_clamped))

        # Pad if needed (transparent padding)
        if pad_left or pad_top or pad_right or pad_bottom:
            padded = Image.new("RGBA", (total_size, total_size), (0, 0, 0, 0))
            padded.paste(region.convert("RGBA"), (pad_left, pad_top))
            region = padded
        else:
            region = region.convert("RGBA")

        # Ensure region is exactly total_size x total_size
        if region.size != (total_size, total_size):
            # This shouldn't happen, but handle it gracefully
            padded = Image.new("RGBA", (total_size, total_size), (0, 0, 0, 0))
            padded.paste(region, (0, 0))
            region = padded

        # Create mask centered in the extraction region
        mask = self._create_piece_mask(
            config,
            (total_size, total_size),
            (total_size / 2, total_size / 2),
            piece_size,
        )

        # Apply mask to alpha channel
        region_array = np.array(region)
        region_array[:, :, 3] = mask

        piece_img = Image.fromarray(region_array, mode="RGBA")

        return piece_img, config

    def generate_piece_with_rotation(
        self,
        puzzle_image: Image.Image,
        center_x: float,
        center_y: float,
        rotation: int = 0,
        config: Optional[PieceConfig] = None,
    ) -> Tuple[Image.Image, PieceConfig, int]:
        """Extract a jigsaw-shaped piece and optionally rotate it.

        Args:
            puzzle_image: The full puzzle image (PIL Image).
            center_x: Normalized x coordinate of piece center (0-1).
            center_y: Normalized y coordinate of piece center (0-1).
            rotation: Rotation in degrees (0, 90, 180, 270).
            config: Optional piece configuration (random if None).

        Returns:
            Tuple of (RGBA piece image with transparency, config used, rotation applied).
        """
        piece_img, config = self.generate_piece(puzzle_image, center_x, center_y, config)

        if rotation and rotation in (90, 180, 270):
            piece_img = piece_img.rotate(-rotation, expand=True)

        return piece_img, config, rotation
