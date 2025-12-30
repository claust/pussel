"""Geometric logic for generating puzzle piece shapes."""

from typing import List, Tuple

import numpy as np

from .models import BezierCurve, PieceConfig, TabParameters


def generate_realistic_tab_edge(
    start: Tuple[float, float],
    end: Tuple[float, float],
    params: TabParameters,
    is_blank: bool = False,
    corner_slope: float = 0.0,
    edge_type: str = "tab",
) -> List[BezierCurve]:
    """Generate a realistic puzzle piece tab using 6 cubic Bezier curves.

    This creates the classic "mushroom" shape with perfectly symmetric curves.
    The bulb is split into two curves using the kappa constant for optimal
    circular arc approximation.

    Args:
        start: Start point of the edge.
        end: End point of the edge.
        params: Tab parameters controlling the shape.
        is_blank: If True, create an indent instead of a protrusion.
        corner_slope: Controls tangent angle at corners.
        edge_type: Either "tab" or "blank".

    Returns:
        List of BezierCurve objects forming the edge.
    """
    direction = 1 if not is_blank else -1
    edge_vec = np.array([end[0] - start[0], end[1] - start[1]])
    edge_length = float(np.linalg.norm(edge_vec))
    edge_unit = edge_vec / edge_length

    # Normal vector pointing toward feature (tab protrusion or blank indent)
    normal = np.array([-edge_unit[1], edge_unit[0]]) * direction

    # Piece's outward normal (always perpendicular, pointing away from piece interior)
    piece_outward_normal = np.array([-edge_unit[1], edge_unit[0]])

    # Corner slope behavior differs for tabs vs blanks:
    # - TABS: Corners curve TOWARD piece interior (acute angle < 90)
    # - BLANKS: Corners curve AWAY from piece interior (obtuse angle > 90)
    if edge_type == "tab":
        corner_sign = -1.0  # Curve toward interior (opposite to outward normal)
    else:  # blank
        corner_sign = 1.0  # Curve toward exterior (same as outward normal)

    # Entry corner offset: applied directly to p1, so tangent = offset direction
    entry_corner_offset = piece_outward_normal * corner_slope * corner_sign
    # Exit corner offset: compensate for (p3-p2) tangent calculation which inverts the sign
    exit_corner_offset = piece_outward_normal * corner_slope * (-corner_sign)

    # Key dimensions (all relative to edge_length)
    full_height = params.height * edge_length
    neck_half = params.neck_width * edge_length * 0.5
    bulb_half = params.bulb_width * edge_length * 0.5

    # Neck height (where the waist is narrowest)
    neck_height = full_height * params.neck_ratio

    # Center of the tab (neck base)
    center = np.array(start) + edge_vec * params.position

    # Asymmetry: shift the bulb horizontally relative to the neck
    bulb_shift = edge_unit * bulb_half * params.asymmetry * 2.0

    # Asymmetric curve factors
    left_curve_factor = 1.0 + params.asymmetry * 0.5
    right_curve_factor = 1.0 - params.asymmetry * 0.5

    curves = []

    # Shoulder offset: displacement of neck base points from the corner-to-corner line
    # The offset opposes the feature direction:
    # - Tabs (normal points outward): neck base moves inward (dip before rising)
    # - Blanks (normal points inward): neck base moves outward (hump before dipping)
    shoulder_offset_vec = -normal * params.shoulder_offset * edge_length

    # Key points for the mushroom shape
    neck_base_left = center - edge_unit * neck_half + shoulder_offset_vec
    neck_base_right = center + edge_unit * neck_half + shoulder_offset_vec

    bulb_center = center + bulb_shift + normal * neck_height
    bulb_base_left = bulb_center - edge_unit * bulb_half
    bulb_base_right = bulb_center + edge_unit * bulb_half

    bulb_mid_left = bulb_base_left + normal * (full_height - neck_height) * 0.5
    bulb_mid_right = bulb_base_right + normal * (full_height - neck_height) * 0.5

    dist_start_to_neck = params.position - params.neck_width * 0.5
    dist_neck_to_end = 1.0 - params.position - params.neck_width * 0.5

    # neck_flare controls shape: positive = pinch inward, negative = flare outward
    flare_amount = (bulb_half - neck_half) * params.neck_flare

    # === G1 Continuity Setup ===
    # Pre-compute C2.P1's offset direction so C1.P2 can be collinear (follow-the-leader)
    # This ensures smooth transition at the C1-C2 junction (neck_base_left)
    c2_p1_offset = edge_unit * flare_amount * left_curve_factor + normal * neck_height * 0.7
    c2_p1_offset_length = float(np.linalg.norm(c2_p1_offset))
    c2_p1_dir = c2_p1_offset / (c2_p1_offset_length + 1e-8)  # Normalized direction

    # Pre-compute C5.P2's offset direction so C6.P1 can be collinear
    # This ensures smooth transition at the C5-C6 junction (neck_base_right)
    c5_p2_offset = -edge_unit * flare_amount * right_curve_factor + normal * neck_height * 0.7
    c5_p2_offset_length = float(np.linalg.norm(c5_p2_offset))
    c5_p2_dir = c5_p2_offset / (c5_p2_offset_length + 1e-8)  # Normalized direction

    # Curve 1: Start to neck base left (flat entry)
    # shoulder_flatness controls how flat the shoulder stays before turning into neck
    shoulder_extend = 0.5 + params.shoulder_flatness * 0.4  # 0.5-0.9: how far p1 extends along edge
    # C1.P2 handle length - controls how far the control point extends from junction
    c1_p2_handle_length = neck_half * 0.8 + neck_height * 0.3  # Blend of horizontal and vertical scale
    p0 = np.array(start)
    p3 = neck_base_left
    p1 = p0 + edge_unit * edge_length * dist_start_to_neck * shoulder_extend + entry_corner_offset * edge_length * 0.3
    # G1 continuity: C1.P2 is collinear with C2.P1, opposite direction from junction
    p2 = p3 - c2_p1_dir * c1_p2_handle_length
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 2: Neck base left up through waist to bulb mid-left
    p0 = curves[-1].p3
    p3 = bulb_mid_left
    # C2.P1 uses the pre-computed offset (this is the "leader")
    p1 = np.array(p0) + c2_p1_offset
    # G1 continuity: p2 must have vertical tangent to match C3's start (no edge_unit offset)
    p2 = np.array(p3) - normal * (full_height - neck_height) * 0.3
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 3a & 3b: Split bulb into two curves for optimal circular arc
    # Using kappa constant (0.5522847498) for quarter-circle approximation
    kappa = 0.5522847498

    # Calculate the geometric apex of the bulb
    bulb_chord_center = (bulb_mid_left + bulb_mid_right) * 0.5
    bulb_radius = bulb_half
    bulb_apex = bulb_chord_center + normal * bulb_radius

    # Curve 3a: Bulb mid-left (Equator) -> Bulb Apex (Top)
    p0 = curves[-1].p3
    p3 = bulb_apex
    # P1: Vertical tangent at equator - extended by squareness for "superellipse" shape
    p1 = np.array(p0) + normal * bulb_radius * kappa * params.curvature * params.squareness
    # P2: Horizontal tangent at apex - extended by squareness for flatter top
    p2 = np.array(p3) - edge_unit * bulb_radius * kappa * params.curvature * params.squareness
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 3b: Bulb Apex (Top) -> Bulb mid-right (Equator)
    p0 = bulb_apex
    p3 = bulb_mid_right
    # P1: Horizontal tangent at apex - extended by squareness for flatter top
    p1 = np.array(p0) + edge_unit * bulb_radius * kappa * params.curvature * params.squareness
    # P2: Vertical tangent at equator - extended by squareness for "superellipse" shape
    p2 = np.array(p3) + normal * bulb_radius * kappa * params.curvature * params.squareness
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 5: Bulb mid-right down through waist to neck base right
    p0 = curves[-1].p3
    p3 = neck_base_right
    # G1 continuity: p1 must have vertical tangent to match C3b's end (no edge_unit offset)
    p1 = np.array(p0) - normal * (full_height - neck_height) * 0.3
    # C5.P2 uses the pre-computed offset (this is the "leader" for C6.P1)
    p2 = np.array(p3) + c5_p2_offset
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 6: Neck base right to end
    # C6.P1 handle length - controls how far the control point extends from junction
    c6_p1_handle_length = neck_half * 0.8 + neck_height * 0.3  # Same as C1.P2 for symmetry
    p0 = curves[-1].p3
    p3 = np.array(end)
    # G1 continuity: C6.P1 is collinear with C5.P2, opposite direction from junction
    p1 = np.array(p0) - c5_p2_dir * c6_p1_handle_length
    p2 = p3 - edge_unit * edge_length * dist_neck_to_end * shoulder_extend + exit_corner_offset * edge_length * 0.3
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    return curves


def generate_corner_curve(
    corner: Tuple[float, float],
    incoming_dir: np.ndarray,
    outgoing_dir: np.ndarray,
    radius: float,
) -> BezierCurve:
    """Generate a rounded corner curve using cubic Bezier approximation of a quarter circle.

    Args:
        corner: The corner vertex position.
        incoming_dir: Unit vector direction of the incoming edge (pointing toward corner).
        outgoing_dir: Unit vector direction of the outgoing edge (pointing away from corner).
        radius: Corner radius.

    Returns:
        A BezierCurve representing the rounded corner.
    """
    kappa = 0.5522847498  # Magic number for quarter-circle approximation

    # Start point: offset from corner along incoming edge direction (backward)
    p0 = np.array(corner) - incoming_dir * radius
    # End point: offset from corner along outgoing edge direction (forward)
    p3 = np.array(corner) + outgoing_dir * radius

    # Control points for quarter-circle approximation
    p1 = p0 + incoming_dir * radius * kappa
    p2 = p3 - outgoing_dir * radius * kappa

    return BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3))  # type: ignore


def generate_piece_geometry(config: PieceConfig) -> List[List[BezierCurve]]:
    """Generate the Bezier curves for all four edges of a puzzle piece.

    This is the primary orchestration function that uses the 6-curve realistic
    algorithm for tabs and blanks, plus corner rounding.

    Args:
        config: The piece configuration specifying edge types and parameters.

    Returns:
        A list of curves in order around the piece:
        [corner0, edge0, corner1, edge1, corner2, edge2, corner3, edge3]
        Each element is a list of BezierCurve objects.
    """
    corners = [
        (0.0, 0.0),  # Bottom-left
        (config.size, 0.0),  # Bottom-right
        (config.size, config.size),  # Top-right
        (0.0, config.size),  # Top-left
    ]

    # Edge direction vectors (unit vectors)
    edge_dirs = [
        np.array([1.0, 0.0]),  # Bottom edge: left to right
        np.array([0.0, 1.0]),  # Right edge: bottom to top
        np.array([-1.0, 0.0]),  # Top edge: right to left
        np.array([0.0, -1.0]),  # Left edge: top to bottom
    ]

    radius = config.corner_radius * config.size
    all_curves: List[List[BezierCurve]] = []

    for i in range(4):
        # Generate corner curve first (corner i connects edge i-1 to edge i)
        prev_edge_dir = edge_dirs[(i - 1) % 4]
        curr_edge_dir = edge_dirs[i]
        corner_curve = generate_corner_curve(
            corners[i],
            incoming_dir=prev_edge_dir,
            outgoing_dir=curr_edge_dir,
            radius=radius,
        )
        all_curves.append([corner_curve])

        # Calculate offset start/end for this edge
        start = tuple((np.array(corners[i]) + curr_edge_dir * radius).tolist())
        end = tuple((np.array(corners[(i + 1) % 4]) - curr_edge_dir * radius).tolist())

        # Get params for this edge
        edge_type = config.edge_types[i]
        params = config.edge_params[i]
        if params is None:
            params = TabParameters.random()

        if edge_type == "flat":
            # Straight line as a single Bezier curve (control points on the line)
            curve = BezierCurve(start, start, end, end)
            all_curves.append([curve])
        else:
            # tab -> is_blank=False, blank -> is_blank=True
            is_blank = edge_type == "blank"
            curves = generate_realistic_tab_edge(
                start, end, params, is_blank=is_blank, corner_slope=params.corner_slope, edge_type=edge_type
            )
            all_curves.append(curves)

    return all_curves


def generate_piece_path(config: PieceConfig, points_per_curve: int = 20) -> Tuple[List[float], List[float]]:
    """Generate the complete path (x, y coordinates) for a puzzle piece.

    Args:
        config: The piece configuration.
        points_per_curve: Number of points to sample from each Bezier curve.

    Returns:
        Tuple of (x_coords, y_coords).
    """
    all_edges = generate_piece_geometry(config)

    all_x: List[float] = []
    all_y: List[float] = []

    for edge in all_edges:
        for curve in edge:
            points = curve.get_points(points_per_curve)
            # Add all points except the last one to avoid duplication
            all_x.extend(points[:-1, 0].tolist())
            all_y.extend(points[:-1, 1].tolist())

    # Close the path by adding the very first point back
    all_x.append(all_x[0])
    all_y.append(all_y[0])

    return all_x, all_y
