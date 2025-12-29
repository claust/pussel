"""Data models for puzzle-related operations."""

from typing import Optional

from pydantic import BaseModel


class Position(BaseModel):
    """Model representing a position in 2D space."""

    x: float
    y: float


class PieceResponse(BaseModel):
    """Response model for puzzle piece processing."""

    position: Position
    position_confidence: float
    rotation: int
    rotation_confidence: float


class PuzzleResponse(BaseModel):
    """Response model for puzzle upload."""

    puzzle_id: str
    image_url: Optional[str] = None
