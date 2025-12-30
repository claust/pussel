"""Edge grid generation for puzzle image cutting.

This module generates a grid of interlocking puzzle edges that can be used
to cut an image into puzzle pieces. Each interior edge is generated once
and shared between adjacent pieces.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from models import BezierCurve, TabParameters


@dataclass
class Edge:
    """A single puzzle edge with its Bezier curve data.

    Attributes:
        edge_type: The type of edge ("tab", "blank", or "flat").
        params: The tab parameters used to generate this edge (None for flat edges).
        curves: The list of Bezier curves defining this edge shape.
    """

    edge_type: Literal["tab", "blank", "flat"]
    params: Optional[TabParameters]
    curves: List[BezierCurve]


@dataclass
class EdgeGrid:
    """Grid of shared edges for a puzzle.

    The grid stores edges in two 2D arrays:
    - horizontal_edges: (rows+1) x cols - edges between vertically adjacent pieces
    - vertical_edges: rows x (cols+1) - edges between horizontally adjacent pieces

    Border edges (top row, bottom row, left col, right col) are flat.
    Interior edges have random tab/blank shapes.
    """

    rows: int
    cols: int
    horizontal_edges: List[List[Edge]]  # [row][col] - (rows+1) x cols
    vertical_edges: List[List[Edge]]  # [row][col] - rows x (cols+1)


def calculate_grid_dimensions(
    image_width: int,
    image_height: int,
    target_pieces: int,
) -> Tuple[int, int]:
    """Calculate optimal grid dimensions for roughly square pieces.

    Args:
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        target_pieces: Target number of pieces.

    Returns:
        Tuple of (rows, cols) where rows * cols is close to target_pieces
        and pieces are roughly square.
    """
    if target_pieces < 4:
        return (2, 2)

    aspect_ratio = image_width / image_height

    # For square pieces: piece_width/piece_height = 1
    # piece_width = image_width/cols, piece_height = image_height/rows
    # So: (image_width/cols) / (image_height/rows) = 1
    # => rows/cols = image_height/image_width = 1/aspect_ratio
    # Also: rows * cols = target_pieces
    # Solving: cols = sqrt(target_pieces * aspect_ratio)
    #          rows = sqrt(target_pieces / aspect_ratio)

    cols_float = math.sqrt(target_pieces * aspect_ratio)
    rows_float = math.sqrt(target_pieces / aspect_ratio)

    # Try nearby integer combinations and pick the best one
    candidates = []
    for rows in [max(2, int(rows_float)), max(2, int(rows_float) + 1)]:
        for cols in [max(2, int(cols_float)), max(2, int(cols_float) + 1)]:
            count = rows * cols
            # Calculate how square the pieces would be
            piece_width = image_width / cols
            piece_height = image_height / rows
            piece_ar = piece_width / piece_height
            squareness = min(piece_ar, 1 / piece_ar)  # 1.0 = perfect square

            # Score: balance squareness with closeness to target
            count_diff = abs(count - target_pieces) / target_pieces
            score = squareness - count_diff * 0.5
            candidates.append((rows, cols, score, count))

    # Return the best candidate
    best = max(candidates, key=lambda x: x[2])
    return (best[0], best[1])


def _generate_flat_edge() -> Edge:
    """Generate a flat (straight) edge."""
    # A flat edge is a straight line from (0, 0) to (1, 0) in normalized coordinates
    # Represented as a degenerate Bezier curve where control points are on the line
    curve = BezierCurve(
        p0=(0.0, 0.0),
        p1=(0.33, 0.0),
        p2=(0.67, 0.0),
        p3=(1.0, 0.0),
    )
    return Edge(edge_type="flat", params=None, curves=[curve])


def _generate_tab_edge(is_tab: bool, params: Optional[TabParameters] = None) -> Edge:
    """Generate a tab or blank edge with random or specified parameters.

    Args:
        is_tab: If True, generate a tab (protrusion). If False, generate a blank (indent).
        params: Optional TabParameters. If None, random parameters are used.

    Returns:
        An Edge object with the generated curves.
    """
    # Import here to avoid circular imports
    from geometry import generate_realistic_tab_edge

    if params is None:
        params = TabParameters.random()

    edge_type: Literal["tab", "blank"] = "tab" if is_tab else "blank"

    # Generate curves in normalized space: start=(0, 0), end=(1, 0)
    curves = generate_realistic_tab_edge(
        start=(0.0, 0.0),
        end=(1.0, 0.0),
        params=params,
        is_blank=not is_tab,
        corner_slope=params.corner_slope,
        edge_type=edge_type,
    )

    return Edge(edge_type=edge_type, params=params, curves=curves)


def generate_edge_grid(
    rows: int,
    cols: int,
    seed: Optional[int] = None,
) -> EdgeGrid:
    """Generate all edges for a puzzle grid.

    Args:
        rows: Number of piece rows.
        cols: Number of piece columns.
        seed: Optional random seed for reproducibility.

    Returns:
        An EdgeGrid containing all horizontal and vertical edges.
    """
    if seed is not None:
        random.seed(seed)

    # Generate horizontal edges: (rows+1) x cols
    # horizontal_edges[r][c] is the edge at the top of piece (r, c)
    # r=0 is the top border (flat), r=rows is the bottom border (flat)
    horizontal_edges: List[List[Edge]] = []
    for r in range(rows + 1):
        row_edges: List[Edge] = []
        for _c in range(cols):
            if r == 0 or r == rows:
                # Border edge - flat
                edge = _generate_flat_edge()
            else:
                # Interior edge - random tab/blank
                is_tab = random.choice([True, False])
                edge = _generate_tab_edge(is_tab)
            row_edges.append(edge)
        horizontal_edges.append(row_edges)

    # Generate vertical edges: rows x (cols+1)
    # vertical_edges[r][c] is the edge at the left of piece (r, c)
    # c=0 is the left border (flat), c=cols is the right border (flat)
    vertical_edges: List[List[Edge]] = []
    for _r in range(rows):
        row_edges = []
        for c in range(cols + 1):
            if c == 0 or c == cols:
                # Border edge - flat
                edge = _generate_flat_edge()
            else:
                # Interior edge - random tab/blank
                is_tab = random.choice([True, False])
                edge = _generate_tab_edge(is_tab)
            row_edges.append(edge)
        vertical_edges.append(row_edges)

    return EdgeGrid(
        rows=rows,
        cols=cols,
        horizontal_edges=horizontal_edges,
        vertical_edges=vertical_edges,
    )


def reverse_curves(curves: List[BezierCurve]) -> List[BezierCurve]:
    """Reverse a list of Bezier curves for traversal in opposite direction.

    Each curve is reversed by swapping p0<->p3 and p1<->p2.
    The list order is also reversed.

    Args:
        curves: List of Bezier curves to reverse.

    Returns:
        New list with reversed curves.
    """
    reversed_list = []
    for curve in reversed(curves):
        reversed_curve = BezierCurve(
            p0=curve.p3,
            p1=curve.p2,
            p2=curve.p1,
            p3=curve.p0,
        )
        reversed_list.append(reversed_curve)
    return reversed_list


def transform_curves(
    curves: List[BezierCurve],
    translate: Tuple[float, float],
    scale: Tuple[float, float],
    rotate_90_ccw: int = 0,
) -> List[BezierCurve]:
    """Transform curves from normalized edge space to piece space.

    Edges are generated in normalized space where the edge goes from (0, 0) to (1, 0).
    This function transforms them to the correct position within a piece.

    Args:
        curves: Curves to transform.
        translate: (x, y) offset to apply after rotation and scaling.
        scale: (sx, sy) scale factors.
        rotate_90_ccw: Number of 90-degree counter-clockwise rotations (0-3).

    Returns:
        Transformed curves.
    """

    def transform_point(p: Tuple[float, float]) -> Tuple[float, float]:
        x, y = p
        # Apply 90-degree CCW rotations
        for _ in range(rotate_90_ccw % 4):
            x, y = -y, x
        # Scale
        x *= scale[0]
        y *= scale[1]
        # Translate
        x += translate[0]
        y += translate[1]
        return (x, y)

    transformed = []
    for curve in curves:
        transformed.append(
            BezierCurve(
                p0=transform_point(curve.p0),
                p1=transform_point(curve.p1),
                p2=transform_point(curve.p2),
                p3=transform_point(curve.p3),
            )
        )
    return transformed


def get_piece_curves(
    edge_grid: EdgeGrid,
    row: int,
    col: int,
) -> List[BezierCurve]:
    """Get all boundary curves for a single piece.

    Returns curves in clockwise order starting from top-left:
    top edge -> right edge -> bottom edge -> left edge

    The curves are in normalized piece coordinates where the piece
    spans (0, 0) to (1, 1), with (0, 0) at bottom-left and (1, 1) at top-right.

    Args:
        edge_grid: The edge grid.
        row: Row index of the piece (0-indexed from top).
        col: Column index of the piece (0-indexed from left).

    Returns:
        List of Bezier curves forming the piece boundary.
    """
    all_curves: List[BezierCurve] = []

    # Original edge space: edge goes from (0, 0) to (1, 0), tab protrudes toward +Y
    #
    # For clockwise boundary:
    # - Top edge: (0, 1) -> (1, 1), tab protrudes +Y (outward)
    # - Right edge: (1, 1) -> (1, 0), tab protrudes +X (outward)
    # - Bottom edge: (1, 0) -> (0, 0), tab protrudes -Y (outward)
    # - Left edge: (0, 0) -> (0, 1), tab protrudes -X (outward)

    # Top edge: horizontal_edges[row][col]
    # Transform: just translate up by 1
    # (0,0)->(1,0) becomes (0,1)->(1,1), tab +Y stays +Y
    top_edge = edge_grid.horizontal_edges[row][col]
    top_curves = transform_curves(
        top_edge.curves,
        translate=(0.0, 1.0),
        scale=(1.0, 1.0),
        rotate_90_ccw=0,
    )
    all_curves.extend(top_curves)

    # Right edge: vertical_edges[row][col+1]
    # Need: (1,1) -> (1,0), tab protrudes +X
    # Transform: rotate 90° CW (= 270° CCW = 3), then translate
    # 90° CW: (x,y) -> (y,-x)
    # (0,0) -> (0,0), (1,0) -> (0,-1), tab direction (0,1) -> (1,0) = +X ✓
    # Then translate (1,1): (0,0) -> (1,1), (0,-1) -> (1,0) ✓
    right_edge = edge_grid.vertical_edges[row][col + 1]
    right_curves = transform_curves(
        right_edge.curves,
        translate=(1.0, 1.0),
        scale=(1.0, 1.0),
        rotate_90_ccw=3,  # 90° CW = 270° CCW
    )
    all_curves.extend(right_curves)

    # Bottom edge: horizontal_edges[row+1][col]
    # Need: (1,0) -> (0,0), tab protrudes -Y
    # Transform: rotate 180° (= 2 x 90° CCW), then translate
    # 180°: (x,y) -> (-x,-y)
    # (0,0) -> (0,0), (1,0) -> (-1,0), tab direction (0,1) -> (0,-1) = -Y ✓
    # Then translate (1,0): (0,0) -> (1,0), (-1,0) -> (0,0) ✓
    bottom_edge = edge_grid.horizontal_edges[row + 1][col]
    bottom_curves = transform_curves(
        bottom_edge.curves,
        translate=(1.0, 0.0),
        scale=(1.0, 1.0),
        rotate_90_ccw=2,  # 180°
    )
    all_curves.extend(bottom_curves)

    # Left edge: vertical_edges[row][col]
    # Need: (0,0) -> (0,1), tab protrudes -X
    # Transform: rotate 90° CCW
    # 90° CCW: (x,y) -> (-y,x)
    # (0,0) -> (0,0), (1,0) -> (0,1), tab direction (0,1) -> (-1,0) = -X ✓
    left_edge = edge_grid.vertical_edges[row][col]
    left_curves = transform_curves(
        left_edge.curves,
        translate=(0.0, 0.0),
        scale=(1.0, 1.0),
        rotate_90_ccw=1,  # 90° CCW
    )
    all_curves.extend(left_curves)

    return all_curves


def get_opposite_edge_type(edge_type: Literal["tab", "blank", "flat"]) -> Literal["tab", "blank", "flat"]:
    """Get the opposite edge type for interlocking.

    Args:
        edge_type: The original edge type.

    Returns:
        The opposite type (tab<->blank, flat stays flat).
    """
    if edge_type == "tab":
        return "blank"
    elif edge_type == "blank":
        return "tab"
    else:
        return "flat"
