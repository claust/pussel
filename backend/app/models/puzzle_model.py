"""Data models for puzzle-related operations."""

from typing import Any, Dict, List, Optional

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
    cleaned_image: Optional[str] = None  # Base64 encoded PNG with background removed


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


# Models for puzzle cutting (play mode)


class CutPuzzleRequest(BaseModel):
    """Request model for cutting a puzzle into pieces."""

    rows: int = Field(..., ge=2, le=10, description="Number of rows in the puzzle grid")
    cols: int = Field(..., ge=2, le=10, description="Number of columns in the puzzle grid")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible edge generation")


class CutPieceInfo(BaseModel):
    """Information about a single cut puzzle piece."""

    id: str = Field(..., description="Unique piece identifier (e.g., 'piece_r0_c0')")
    row: int = Field(..., description="Row position in the grid")
    col: int = Field(..., description="Column position in the grid")
    image: str = Field(..., description="Base64 encoded PNG with transparency")
    correct_x: float = Field(..., description="Normalized x coordinate of correct center position (0-1)")
    correct_y: float = Field(..., description="Normalized y coordinate of correct center position (0-1)")
    width: int = Field(..., description="Width of the piece image in pixels")
    height: int = Field(..., description="Height of the piece image in pixels")


class CutPuzzleResponse(BaseModel):
    """Response model for cutting a puzzle into pieces."""

    pieces: List[CutPieceInfo] = Field(..., description="List of cut puzzle pieces")
    grid: Dict[str, int] = Field(..., description="Grid dimensions {'rows': N, 'cols': M}")
    puzzle_width: int = Field(..., description="Width of the original puzzle in pixels")
    puzzle_height: int = Field(..., description="Height of the original puzzle in pixels")
