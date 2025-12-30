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

# Minimum distance between any two blank curves on a piece.
# This is measured in normalized piece coordinates (piece is 1x1).
# A value of 0.15 means blanks must be at least 15% of piece size apart.
MIN_BLANK_CURVE_DISTANCE = 0.15

# Number of points to sample per curve for distance calculation.
DISTANCE_SAMPLE_POINTS = 10

# Maximum regeneration attempts before giving up on constraint satisfaction.
MAX_REGENERATION_ATTEMPTS = 100


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


def get_edge_type_for_piece(edge: Edge, position: Literal["top", "right", "bottom", "left"]) -> str:
    """Get the effective edge type from a piece's perspective.

    An edge is shared between two adjacent pieces, but they see opposite types:
    - For "top" and "left" edges: the stored edge_type applies directly
    - For "bottom" and "right" edges: the type is inverted (tab<->blank)

    Args:
        edge: The edge to check.
        position: The edge's position relative to the piece ("top", "right", "bottom", "left").

    Returns:
        The effective edge type ("tab", "blank", or "flat") from the piece's perspective.
    """
    if edge.edge_type == "flat":
        return "flat"

    if position in ("top", "left"):
        return edge.edge_type
    else:  # "bottom" or "right"
        return "blank" if edge.edge_type == "tab" else "tab"


def _sample_curve_points(
    curve: BezierCurve,
    num_points: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> List[Tuple[float, float]]:
    """Sample points along a Bezier curve within a t range."""
    points = []
    for i in range(num_points):
        t = t_start + i * (t_end - t_start) / (num_points - 1) if num_points > 1 else (t_start + t_end) / 2
        points.append(curve.evaluate(t))
    return points


def _min_distance_between_point_sets(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
) -> float:
    """Calculate minimum distance between two sets of points."""
    import math

    min_dist = float("inf")
    for p1 in points1:
        for p2 in points2:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            min_dist = min(min_dist, dist)
    return min_dist


def _is_inside_piece(point: Tuple[float, float], margin: float = 0.02) -> bool:
    """Check if a point is inside the piece boundaries (with margin)."""
    x, y = point
    return margin < x < 1.0 - margin and margin < y < 1.0 - margin


def _get_blank_curve_points(
    edge: Edge,
    position: Literal["top", "right", "bottom", "left"],
) -> List[Tuple[float, float]]:
    """Get sampled points from curves that indent INTO this piece.

    The geometry depends on the STORED edge type, not the perspective type:
    - Stored "blank" → curves indent into this piece (after rotation)
    - Stored "tab" → curves protrude out of this piece

    We check stored type "blank" because those curves go INTO the piece.
    """
    if edge.edge_type != "blank":  # Only stored "blank" curves indent into piece
        return []

    # Transform parameters based on edge position
    # Piece coordinates: (0,0) bottom-left, (1,1) top-right
    if position == "top":
        translate, rotate = (0.0, 1.0), 0
    elif position == "left":
        translate, rotate = (0.0, 0.0), 1  # 90° CCW
    elif position == "bottom":
        translate, rotate = (1.0, 0.0), 2  # 180°
    else:  # right
        translate, rotate = (1.0, 1.0), 3  # 270° CCW (90° CW)

    transformed = transform_curves(edge.curves, translate, (1.0, 1.0), rotate)

    # Sample all curves but only keep points INSIDE the piece
    all_points: List[Tuple[float, float]] = []
    for curve in transformed:
        points = _sample_curve_points(curve, DISTANCE_SAMPLE_POINTS, t_start=0.05, t_end=0.95)
        for p in points:
            if _is_inside_piece(p):
                all_points.append(p)

    return all_points


def _check_piece_blank_distances(
    top_edge: Edge,
    left_edge: Edge,
    right_edge: Edge,
    bottom_edge: Edge,
) -> bool:
    """Check if all blank curves on a piece are sufficiently far apart.

    Returns True if constraints are satisfied, False otherwise.
    """
    # Collect points from all blank edges
    blank_point_sets: List[Tuple[str, List[Tuple[float, float]]]] = []

    for edge, pos in [
        (top_edge, "top"),
        (left_edge, "left"),
        (right_edge, "right"),
        (bottom_edge, "bottom"),
    ]:
        pos_literal: Literal["top", "right", "bottom", "left"] = pos  # type: ignore[assignment]
        points = _get_blank_curve_points(edge, pos_literal)
        if points:
            blank_point_sets.append((pos, points))

    # Check distances between all pairs of blank edges
    for i, (_pos1, points1) in enumerate(blank_point_sets):
        for _pos2, points2 in blank_point_sets[i + 1 :]:
            min_dist = _min_distance_between_point_sets(points1, points2)
            if min_dist < MIN_BLANK_CURVE_DISTANCE:
                return False

    return True


def _regenerate_edge_params(edge: Edge, scale: float = 1.0) -> None:
    """Regenerate edge with random parameters, optionally scaled down.

    Args:
        edge: The edge to regenerate.
        scale: Scale factor for height/bulb_width (1.0 = normal, <1.0 = smaller).
    """
    if edge.params is None or edge.edge_type == "flat":
        return

    from geometry import generate_realistic_tab_edge

    new_params = TabParameters.random()
    # Scale down height and bulb_width to make blanks smaller and less likely to overlap
    new_params.height *= scale
    new_params.bulb_width *= scale
    # Center position to keep blanks away from corners
    new_params.position = 0.45 + 0.1 * (new_params.position - 0.4) / 0.2  # Compress to 0.45-0.55

    is_blank = edge.edge_type == "blank"
    edge.params = new_params
    edge.curves = generate_realistic_tab_edge(
        start=(0.0, 0.0),
        end=(1.0, 0.0),
        params=new_params,
        is_blank=is_blank,
        corner_slope=new_params.corner_slope,
        edge_type=edge.edge_type,
    )


def _fix_piece_violations(
    top_edge: Edge,
    left_edge: Edge,
    right_edge: Edge,
    bottom_edge: Edge,
) -> bool:
    """Attempt to fix distance violations for a piece by regenerating edge parameters.

    Returns True if piece now satisfies constraints, False if max attempts exceeded.
    """
    edges_info: List[Tuple[Edge, Literal["top", "right", "bottom", "left"]]] = [
        (top_edge, "top"),
        (left_edge, "left"),
        (right_edge, "right"),
        (bottom_edge, "bottom"),
    ]

    # Get edges with stored type "blank" (these are the curves that indent into pieces)
    blank_edges = [(edge, pos) for edge, pos in edges_info if edge.edge_type == "blank"]

    if len(blank_edges) < 2:
        return True  # No possible violation with < 2 blanks

    for attempt in range(MAX_REGENERATION_ATTEMPTS):
        if _check_piece_blank_distances(top_edge, left_edge, right_edge, bottom_edge):
            return True

        # Progressive scaling: start at 1.0, decrease to 0.5 over attempts
        scale = 1.0 - 0.5 * (attempt / MAX_REGENERATION_ATTEMPTS)
        for edge, _pos in blank_edges:
            _regenerate_edge_params(edge, scale=scale)

    # Final check
    return _check_piece_blank_distances(top_edge, left_edge, right_edge, bottom_edge)


def _apply_distance_constraints(
    horizontal_edges: List[List[Edge]],
    vertical_edges: List[List[Edge]],
    rows: int,
    cols: int,
) -> None:
    """Apply geometric distance constraints using iterative rejection sampling.

    Edges are shared between adjacent pieces, so fixing one piece might break
    another. We iterate until all pieces pass or we hit max iterations.

    Args:
        horizontal_edges: The horizontal edge grid.
        vertical_edges: The vertical edge grid.
        rows: Number of piece rows.
        cols: Number of piece columns.
    """
    max_iterations = 10  # Prevent infinite loops

    for _iteration in range(max_iterations):
        all_passed = True

        for r in range(rows):
            for c in range(cols):
                top = horizontal_edges[r][c]
                left = vertical_edges[r][c]
                right = vertical_edges[r][c + 1]
                bottom = horizontal_edges[r + 1][c]

                if not _check_piece_blank_distances(top, left, right, bottom):
                    all_passed = False
                    _fix_piece_violations(top, left, right, bottom)

        if all_passed:
            break


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

    # Apply geometric distance constraints to prevent blanks from getting too close
    _apply_distance_constraints(horizontal_edges, vertical_edges, rows, cols)

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
