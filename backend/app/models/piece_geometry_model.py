"""Data models for the piece-geometry API (contour/corner/edge/fingerprint dedupe)."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

EdgeType = Literal["tab", "blank", "flat"]
MatchStatus = Literal["matched", "new", "uncertain"]


class GeometryPoint(BaseModel):
    """A single point, normalized to [0, 1] against the uploaded piece photo's width/height.

    The service extracts geometry within a crop but maps every coordinate back to the full
    uploaded image before normalizing (see `_normalize_points`), so these are full-photo fractions,
    not fractions of a cropped piece image.
    """

    x: float
    y: float


class GeometryEdge(BaseModel):
    """One classified edge, in contour traversal order (orientation is unknown, so not N/E/S/W)."""

    type: EdgeType
    dominant_dev: float = Field(
        ..., description="Signed dominant deviation from the corner-to-corner chord, normalized by chord length"
    )
    polyline: List[GeometryPoint] = Field(default_factory=list, description="100-point edge polyline, normalized 0-1")


class GeometryQuality(BaseModel):
    """Quality gates and metrics for an extracted piece contour (measured within the piece crop)."""

    is_clean: bool = Field(..., description="Contour passed the quality gate (component count, area, solidity)")
    corner_disagreement: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the polydp and curvature corner detectors disagreed by more than the tolerance; "
            "null when corner detection never ran (the flag only asserts what was actually measured)"
        ),
    )
    n_large_components: int
    border_touching: bool
    area_ratio: float
    solidity: float


class PieceGeometryRecordResponse(BaseModel):
    """The geometric record extracted from one piece photo."""

    corners: List[GeometryPoint] = Field(default_factory=list, description="4 corners, normalized 0-1")
    corner_confidences: List[float] = Field(default_factory=list)
    edges: List[GeometryEdge] = Field(default_factory=list, description="4 edges, in contour traversal order")
    contour: Optional[List[GeometryPoint]] = Field(
        default=None, description="Full contour, normalized 0-1; present only when include_contour=true"
    )


class PieceGeometryUploadResponse(BaseModel):
    """Response for POST .../piece/geometry: the extracted record plus the dedupe verdict."""

    piece_id: Optional[str] = Field(
        default=None, description="This photo's piece id (matched/new), or null when uncertain (not enrolled)"
    )
    status: MatchStatus
    match_piece_id: Optional[str] = Field(
        default=None, description="Closest existing piece id, when a comparison was made"
    )
    z_score: Optional[float] = Field(default=None, description="Closest match's combined shape+color z-score")
    lockable: bool = Field(..., description="Clean contour, corner detectors agree, geometry fully computed")
    quality: GeometryQuality
    record: PieceGeometryRecordResponse


class PieceGeometrySummary(BaseModel):
    """One enrolled piece, without polylines (used by the list endpoint)."""

    piece_id: str
    edge_types: List[EdgeType]
    is_clean: bool
    corner_disagreement: bool


class PieceGeometryListResponse(BaseModel):
    """Response for GET .../piece/geometry: all pieces enrolled for a puzzle."""

    puzzle_id: str
    pieces: List[PieceGeometrySummary]
