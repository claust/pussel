"""Geometric logic for generating puzzle piece shapes."""

from typing import List, Tuple

import numpy as np
from models import BezierCurve, PieceConfig, TabParameters


def generate_realistic_tab_edge(
    start: Tuple[float, float],
    end: Tuple[float, float],
    params: TabParameters,
    is_blank: bool = False,
    corner_slope: float = 0.0,
    edge_type: str = "tab",
) -> List[BezierCurve]:
    """Generate a realistic puzzle piece tab using 5 cubic Bézier curves.

    This creates the classic "mushroom" shape with perfectly symmetric curves.
    """
    direction = 1 if not is_blank else -1
    edge_vec = np.array([end[0] - start[0], end[1] - start[1]])
    edge_length = np.linalg.norm(edge_vec)
    edge_unit = edge_vec / edge_length

    # Normal vector (perpendicular to edge)
    normal = np.array([-edge_unit[1], edge_unit[0]]) * direction

    # Corner slope: adjust tangent direction at corners
    if edge_type == "tab":
        slope_direction = 1.0  # Angle toward the bulge (down into the tab)
    else:  # blank
        slope_direction = -1.0  # Angle away from the indent (up from the blank)

    # The tangent offset at corners - angled slightly toward/away from the feature
    corner_tangent_offset = normal * corner_slope * slope_direction

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

    # Key points for the mushroom shape
    neck_base_left = center - edge_unit * neck_half
    neck_base_right = center + edge_unit * neck_half

    bulb_center = center + bulb_shift + normal * neck_height
    bulb_base_left = bulb_center - edge_unit * bulb_half
    bulb_base_right = bulb_center + edge_unit * bulb_half

    bulb_mid_left = bulb_base_left + normal * (full_height - neck_height) * 0.5
    bulb_mid_right = bulb_base_right + normal * (full_height - neck_height) * 0.5

    dist_start_to_neck = params.position - params.neck_width * 0.5
    dist_neck_to_end = 1.0 - params.position - params.neck_width * 0.5

    # Curve 1: Start to neck base left (flat entry)
    p0 = np.array(start)
    p3 = neck_base_left
    p1 = p0 + edge_unit * edge_length * dist_start_to_neck * 0.6 + corner_tangent_offset * edge_length * 0.3
    p2 = p3 - edge_unit * neck_half * 0.5
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 2: Neck base left up through waist to bulb mid-left
    p0 = curves[-1].p3
    p3 = bulb_mid_left
    pinch_amount = (bulb_half - neck_half) * 0.4
    p1 = np.array(p0) + edge_unit * pinch_amount * left_curve_factor + normal * neck_height * 0.7
    p2 = (
        np.array(p3)
        - normal * (full_height - neck_height) * 0.3
        - edge_unit * (bulb_half - neck_half) * 0.3 * left_curve_factor
    )
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 3: Bulb mid-left up and around to bulb mid-right
    p0 = curves[-1].p3
    p3 = bulb_mid_right
    bulb_radius = bulb_half
    p1 = np.array(p0) + normal * bulb_radius * 0.8 * params.curvature - edge_unit * bulb_radius * 0.1
    p2 = np.array(p3) + normal * bulb_radius * 0.8 * params.curvature + edge_unit * bulb_radius * 0.1
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 4: Bulb mid-right down through waist to neck base right
    p0 = curves[-1].p3
    p3 = neck_base_right
    p1 = (
        np.array(p0)
        - normal * (full_height - neck_height) * 0.3
        + edge_unit * (bulb_half - neck_half) * 0.3 * right_curve_factor
    )
    p2 = np.array(p3) - edge_unit * pinch_amount * right_curve_factor + normal * neck_height * 0.7
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 5: Neck base right to end
    p0 = curves[-1].p3
    p3 = np.array(end)
    p1 = p0 + edge_unit * neck_half * 0.5
    p2 = p3 - edge_unit * edge_length * dist_neck_to_end * 0.6 + corner_tangent_offset * edge_length * 0.3
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    return curves


def generate_piece_geometry(config: PieceConfig) -> List[List[BezierCurve]]:
    """Generate the Bézier curves for all four edges of a puzzle piece.

    This is the primary orchestration function that uses the 5-curve realistic
    algorithm for tabs and blanks.

    Returns:
        A list of 4 edges, where each edge is a list of BezierCurve objects.
        Order: bottom, right, top, left.
    """
    corners = [
        (0, 0),  # Bottom-left
        (config.size, 0),  # Bottom-right
        (config.size, config.size),  # Top-right
        (0, config.size),  # Top-left
    ]

    all_edges: List[List[BezierCurve]] = []

    for i, edge_type in enumerate(config.edge_types):
        start = corners[i]
        end = corners[(i + 1) % 4]

        # Get params for this edge
        params = config.edge_params[i]
        if params is None:
            params = TabParameters.random()

        if edge_type == "flat":
            # Straight line as a single Bézier curve (control points on the line)
            curve = BezierCurve(start, start, end, end)
            all_edges.append([curve])
        else:
            # tab -> is_blank=False, blank -> is_blank=True
            is_blank = edge_type == "blank"
            curves = generate_realistic_tab_edge(
                start, end, params, is_blank=is_blank, corner_slope=params.corner_slope, edge_type=edge_type
            )
            all_edges.append(curves)

    return all_edges


def generate_piece_path(config: PieceConfig, points_per_curve: int = 20) -> Tuple[List[float], List[float]]:
    """Generate the complete path (x, y coordinates) for a puzzle piece.

    Args:
        config: The piece configuration.
        points_per_curve: Number of points to sample from each Bézier curve.

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
