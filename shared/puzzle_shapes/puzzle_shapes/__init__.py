"""Puzzle shapes - shared library for puzzle piece generation.

This package provides tools for generating realistic jigsaw puzzle piece shapes
using Bezier curves, cutting images into puzzle pieces, and managing puzzle grids.
"""

from .edge_grid import (
    DISTANCE_SAMPLE_POINTS,
    MAX_REGENERATION_ATTEMPTS,
    MIN_BLANK_CURVE_DISTANCE,
    Edge,
    EdgeGrid,
    calculate_grid_dimensions,
    generate_edge_grid,
    get_edge_type_for_piece,
    get_opposite_edge_type,
    get_piece_curves,
    reverse_curves,
    transform_curves,
)
from .geometry import generate_corner_curve, generate_piece_geometry, generate_piece_path, generate_realistic_tab_edge
from .image_masking import (
    CoordinateMapper,
    calculate_piece_bounds,
    create_piece_mask,
    cut_all_pieces,
    cut_piece,
    generate_piece_polygon,
    sample_curve_points,
)
from .models import BezierCurve, PieceConfig, TabParameters

__all__ = [
    # Models
    "BezierCurve",
    "TabParameters",
    "PieceConfig",
    # Geometry
    "generate_realistic_tab_edge",
    "generate_corner_curve",
    "generate_piece_geometry",
    "generate_piece_path",
    # Edge grid
    "MIN_BLANK_CURVE_DISTANCE",
    "DISTANCE_SAMPLE_POINTS",
    "MAX_REGENERATION_ATTEMPTS",
    "Edge",
    "EdgeGrid",
    "calculate_grid_dimensions",
    "generate_edge_grid",
    "get_edge_type_for_piece",
    "get_piece_curves",
    "reverse_curves",
    "transform_curves",
    "get_opposite_edge_type",
    # Image masking
    "CoordinateMapper",
    "sample_curve_points",
    "generate_piece_polygon",
    "create_piece_mask",
    "calculate_piece_bounds",
    "cut_piece",
    "cut_all_pieces",
]
