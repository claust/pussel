"""Data models for puzzle-related operations."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Model representing a position in 2D space."""

    x: float
    y: float


class PieceSpan(BaseModel):
    """Normalized size of the piece image frame on the puzzle, in [0,1] of puzzle width/height."""

    width: float
    height: float


class PieceResponse(BaseModel):
    """Response model for puzzle piece processing."""

    position: Position
    position_confidence: float
    rotation: int
    rotation_confidence: float
    cleaned_image: Optional[str] = None  # Base64 encoded PNG with background removed
    piece_span: Optional[PieceSpan] = Field(
        default=None,
        description=(
            "Normalized size of the full piece image frame (as sent by the client, in its own "
            "orientation) as a fraction of the puzzle's width/height. Null when no measurement "
            "exists (CNN path, or matcher failure fallback)."
        ),
    )
    grid_row: Optional[int] = Field(
        default=None,
        description=(
            "0-based row of the grid cell nearest to the predicted position. Null when the "
            "puzzle's grid is unknown (no piece_count at upload)."
        ),
    )
    grid_col: Optional[int] = Field(
        default=None, description="0-based column of the nearest grid cell; null when the grid is unknown"
    )
    snapped_position: Optional[Position] = Field(
        default=None,
        description=(
            "Center of the (grid_row, grid_col) cell, normalized to the puzzle image — the "
            "display position, always inside the puzzle. `position` remains the raw prediction. "
            "Null when the grid is unknown."
        ),
    )


class PuzzleResponse(BaseModel):
    """Response model for puzzle upload."""

    puzzle_id: str
    image_url: Optional[str] = None
    piece_count: Optional[int] = Field(
        default=None, description="Client-supplied total piece count, echoed back when provided"
    )
    rows: Optional[int] = Field(default=None, description="Estimated grid rows, present when piece_count was given")
    cols: Optional[int] = Field(default=None, description="Estimated grid cols, present when piece_count was given")


class PuzzleSummary(BaseModel):
    """Summary of one puzzle owned by the caller, as returned by the puzzle-listing endpoint."""

    puzzle_id: str
    rows: Optional[int] = Field(default=None, description="Estimated grid rows, present when known")
    cols: Optional[int] = Field(default=None, description="Estimated grid cols, present when known")


class PuzzleListResponse(BaseModel):
    """Response model for listing the caller's own puzzles."""

    puzzles: List[PuzzleSummary] = Field(..., description="The caller's puzzles, in upload order")


# Models for puzzle frame detection (real mode)


class Corner(BaseModel):
    """A single corner point, normalized to [0, 1] image coordinates."""

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class QuadCorners(BaseModel):
    """Four corners of a quadrilateral in TL/TR/BR/BL order."""

    top_left: Corner
    top_right: Corner
    bottom_right: Corner
    bottom_left: Corner

    def as_points(self) -> List[tuple[float, float]]:
        """Return the corners as a list of (x, y) tuples in TL, TR, BR, BL order."""
        return [
            (self.top_left.x, self.top_left.y),
            (self.top_right.x, self.top_right.y),
            (self.bottom_right.x, self.bottom_right.y),
            (self.bottom_left.x, self.bottom_left.y),
        ]

    @classmethod
    def from_points(cls, points: List[tuple[float, float]]) -> "QuadCorners":
        """Build a QuadCorners from a list of (x, y) tuples in TL, TR, BR, BL order.

        Args:
            points: Exactly four (x, y) tuples ordered TL, TR, BR, BL.

        Returns:
            The corresponding QuadCorners model.
        """
        tl, tr, br, bl = points
        return cls(
            top_left=Corner(x=tl[0], y=tl[1]),
            top_right=Corner(x=tr[0], y=tr[1]),
            bottom_right=Corner(x=br[0], y=br[1]),
            bottom_left=Corner(x=bl[0], y=bl[1]),
        )


class BoundingBox(BaseModel):
    """An axis-aligned bounding box, normalized to [0, 1] image coordinates."""

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)


class PiecePreviewResponse(BaseModel):
    """Response model for live piece-region detection in a camera frame."""

    found: bool = Field(..., description="Whether a piece-like region was detected")
    polygon: List[Corner] = Field(default_factory=list, description="Outline of the detected region, normalized 0-1")
    bbox: Optional[BoundingBox] = Field(default=None, description="Bounding box of the detected region")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How piece-like the detected region is; 0.0 when found is False"
    )
    lockable: Optional[bool] = Field(
        default=None,
        description=(
            "Best-effort piece-geometry quality flag from a quick corner-detection pass on the preview "
            "region; only populated when the request opts in with include_quality=true"
        ),
    )
    corner_disagreement: Optional[bool] = Field(
        default=None,
        description="Whether the quick corner cross-check disagreed; only populated with include_quality=true",
    )


class DetectFrameResponse(BaseModel):
    """Response model for puzzle frame detection."""

    trimmed_image: str = Field(..., description="Base64 data URL (JPEG) of the perspective-corrected puzzle image")
    corners: QuadCorners = Field(..., description="Corners used, normalized 0-1 relative to the EXIF-corrected photo")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence; 1.0 for manual corners")


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
