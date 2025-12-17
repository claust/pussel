"""Service module for processing puzzle pieces and matching them to puzzles.

WARNING: This is a MOCK implementation that returns random values.
There is NO trained machine learning model. This explains poor performance.
See MODEL_ARCHITECTURE_ANALYSIS.md for detailed analysis and recommendations.
"""

import random

from fastapi import UploadFile

from app.models.puzzle_model import PieceResponse, Position


class ImageProcessor:
    """Image processing service for puzzle piece detection.

    IMPORTANT: This is a mock implementation using random values.
    No actual image processing or ML inference occurs.
    Confidence is set to 0.0 to indicate no real prediction.

    TODO: Replace with actual ML model after:
    1. Collecting training dataset (1,000+ puzzles)
    2. Training a proper CNN-based model
    3. See MODEL_ARCHITECTURE_ANALYSIS.md for implementation plan
    """

    def process_piece(self, piece_file: UploadFile) -> PieceResponse:
        """Process a puzzle piece image and predict its position.

        MOCK IMPLEMENTATION: Returns random values, not actual predictions.

        Args:
            piece_file: The puzzle piece image file (currently not analyzed).

        Returns:
            PieceResponse: Random position and rotation with 0.0 confidence
                          to indicate this is not a real prediction.
        """
        # MOCK IMPLEMENTATION - Returns random values
        # The uploaded image is NOT processed or analyzed
        # This is a placeholder until a real ML model is trained

        position = Position(x=random.uniform(0, 100), y=random.uniform(0, 100))
        # Confidence set to 0.0 to indicate no actual ML prediction
        confidence = 0.0
        rotation = random.choice([0, 90, 180, 270])

        return PieceResponse(
            position=position, confidence=confidence, rotation=rotation
        )
