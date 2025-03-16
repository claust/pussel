"""Service module for processing puzzle pieces and matching them to puzzles."""

import random

from fastapi import UploadFile

from app.models.puzzle_model import PieceResponse, Position


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
        position = Position(x=random.uniform(0, 100), y=random.uniform(0, 100))
        confidence = random.uniform(0.5, 1.0)
        rotation = random.choice([0, 90, 180, 270])

        return PieceResponse(
            position=position, confidence=confidence, rotation=rotation
        )
