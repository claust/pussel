"""Service for cutting puzzle images into jigsaw-shaped pieces."""

import base64
import io
from typing import List, Optional, Tuple

from PIL import Image
from puzzle_shapes import CoordinateMapper, cut_piece, generate_edge_grid, generate_piece_polygon

from app.models.puzzle_model import CutPieceInfo


class PuzzleCutter:
    """Cuts puzzle images into jigsaw-shaped pieces using Bezier curves."""

    def __init__(self, padding: int = 20, points_per_curve: int = 20):
        """Initialize the puzzle cutter.

        Args:
            padding: Padding around each piece in pixels for tab protrusions.
            points_per_curve: Number of points to sample per Bezier curve.
        """
        self.padding = padding
        self.points_per_curve = points_per_curve

    def cut_puzzle(
        self,
        puzzle_image: Image.Image,
        rows: int,
        cols: int,
        seed: Optional[int] = None,
    ) -> Tuple[List[CutPieceInfo], int, int]:
        """Cut a puzzle image into jigsaw-shaped pieces.

        Args:
            puzzle_image: The puzzle image to cut.
            rows: Number of rows in the puzzle grid.
            cols: Number of columns in the puzzle grid.
            seed: Random seed for reproducible edge generation.

        Returns:
            Tuple of (list of piece info, puzzle width, puzzle height).
        """
        # Ensure image is in RGB/RGBA mode
        if puzzle_image.mode not in ("RGB", "RGBA"):
            puzzle_image = puzzle_image.convert("RGB")

        width, height = puzzle_image.size

        # Generate edge grid with consistent Bezier curves
        edge_grid = generate_edge_grid(rows, cols, seed=seed)

        # Create coordinate mapper for pixel <-> grid conversion
        mapper = CoordinateMapper(image_width=width, image_height=height, rows=rows, cols=cols)

        pieces: List[CutPieceInfo] = []

        for r in range(rows):
            for c in range(cols):
                # Generate the piece polygon using Bezier curves
                polygon = generate_piece_polygon(
                    edge_grid,
                    mapper,
                    r,
                    c,
                    points_per_curve=self.points_per_curve,
                )

                # Cut the piece from the puzzle image
                # offset is (x, y) of the top-left corner of the piece bounding box
                piece_img, offset = cut_piece(puzzle_image, polygon, padding=self.padding)

                # Convert piece to base64
                piece_base64 = self._image_to_base64(piece_img)

                # Calculate correct center position based on actual piece bounding box
                # Center is offset + half the piece dimensions, normalized to 0-1
                correct_x = (offset[0] + piece_img.width / 2) / width
                correct_y = (offset[1] + piece_img.height / 2) / height

                pieces.append(
                    CutPieceInfo(
                        id=f"piece_r{r}_c{c}",
                        row=r,
                        col=c,
                        image=piece_base64,
                        correct_x=correct_x,
                        correct_y=correct_y,
                        width=piece_img.width,
                        height=piece_img.height,
                    )
                )

        return pieces, width, height

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64 data URL.

        Args:
            image: The image to convert.

        Returns:
            Base64 encoded data URL string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


# Singleton instance
_puzzle_cutter: Optional[PuzzleCutter] = None


def get_puzzle_cutter() -> PuzzleCutter:
    """Get the singleton PuzzleCutter instance."""
    global _puzzle_cutter
    if _puzzle_cutter is None:
        _puzzle_cutter = PuzzleCutter()
    return _puzzle_cutter
