"""Data models for puzzle-related operations."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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


class GeneratePieceRequest(BaseModel):
    """Request model for generating a realistic puzzle piece."""

    center_x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate of piece center (0-1)")
    center_y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate of piece center (0-1)")
    piece_size_ratio: float = Field(
        default=0.25, ge=0.1, le=0.5, description="Size of piece relative to puzzle dimensions"
    )


class GeneratePieceResponse(BaseModel):
    """Response model for generated puzzle piece."""

    piece_image: str = Field(..., description="Base64 encoded PNG with transparency")
    piece_config: Dict[str, Any] = Field(..., description="The PieceConfig used for reproducibility")
