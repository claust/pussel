import random
from io import BytesIO

from PIL import Image

from app.models.puzzle_model import PieceResponse, Position


class ImageProcessor:
    """Service for processing puzzle pieces and matching them to puzzles."""

    def process_piece(self, piece_data: bytes, puzzle_path: str) -> PieceResponse:
        """Process a puzzle piece and find its position in the puzzle.

        Args:
            piece_data: Raw bytes of the piece image.
            puzzle_path: Path to the puzzle image file.

        Returns:
            PieceResponse: The response containing the piece's position and confidence.
        """
        # Mock implementation - replace with actual image processing logic
        _ = Image.open(BytesIO(piece_data))  # Validate piece image
        puzzle_img = Image.open(puzzle_path)

        # Get dimensions
        puzzle_width, puzzle_height = puzzle_img.size

        # Generate mock position within puzzle bounds
        x = random.randint(0, puzzle_width)
        y = random.randint(0, puzzle_height)

        # Mock confidence and rotation
        confidence = random.uniform(0.5, 1.0)
        rotation = random.choice([0, 90, 180, 270])

        return PieceResponse(
            position=Position(x=x, y=y), confidence=confidence, rotation=rotation
        )
