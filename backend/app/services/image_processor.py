"""Service module for processing puzzle pieces and matching them to puzzles."""

import random

from app.models.puzzle_model import PieceResponse, Position
from fastapi import UploadFile


class ImageProcessor:
    """Image processing service for puzzle piece detection."""

    def process_piece(self, piece_file: UploadFile) -> PieceResponse:
        """Process a puzzle piece image and predict its position.

        Args:
            piece_file: The puzzle piece image file.

        Returns:
            PieceResponse: Predicted position and confidence.
        """
        # Mock implementation - replace with actual ML model
        # Return normalized coordinates (0.0-1.0) where (0,0) is top-left
        # and (1,1) is bottom-right
        position = Position(
            x=random.uniform(0.1, 0.9),
            y=random.uniform(0.1, 0.9)
        )
        confidence = random.uniform(0.5, 1.0)
        rotation = random.choice([0, 90, 180, 270])

        return PieceResponse(
            position=position,
            confidence=confidence,
            rotation=rotation
        )
