"""Legacy edge generation algorithms for puzzle pieces."""

from typing import List, Tuple

import numpy as np
from models import BezierCurve, TabParameters


def generate_tab_edge_2_curves(
    start: Tuple[float, float], end: Tuple[float, float], params: TabParameters, is_blank: bool = False
) -> List[BezierCurve]:
    """Generate a puzzle piece edge with tab/blank using 2 cubic Bézier curves."""
    direction = 1 if not is_blank else -1
    edge_vec = np.array([end[0] - start[0], end[1] - start[1]])
    edge_length = np.linalg.norm(edge_vec)
    edge_unit = edge_vec / edge_length

    # Normal vector (perpendicular to edge)
    normal = np.array([-edge_unit[1], edge_unit[0]]) * direction

    # Key points
    tab_center = np.array(start) + edge_vec * params.position
    tab_top = tab_center + normal * edge_length * params.height

    # Control points for left curve
    p0 = start
    p1 = np.array(start) + edge_vec * (params.position - params.neck_width)
    p2 = tab_top - edge_unit * edge_length * params.bulb_width * 0.5
    p3 = tab_top

    curve1 = BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3))  # type: ignore

    # Control points for right curve
    p0 = tab_top
    p1 = tab_top + edge_unit * edge_length * params.bulb_width * 0.5
    p2 = np.array(end) - edge_vec * (1 - params.position - params.neck_width)
    p3 = end

    curve2 = BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3))  # type: ignore

    return [curve1, curve2]


def generate_tab_edge_4_curves(
    start: Tuple[float, float], end: Tuple[float, float], params: TabParameters, is_blank: bool = False
) -> List[BezierCurve]:
    """Generate a puzzle piece edge with tab/blank using 4 cubic Bézier curves."""
    direction = 1 if not is_blank else -1
    edge_vec = np.array([end[0] - start[0], end[1] - start[1]])
    edge_length = np.linalg.norm(edge_vec)
    edge_unit = edge_vec / edge_length

    # Normal vector (perpendicular to edge)
    normal = np.array([-edge_unit[1], edge_unit[0]]) * direction

    # Key positions along the edge
    neck_start_pos = params.position - params.neck_width
    neck_end_pos = params.position + params.neck_width
    bulb_start_pos = params.position - params.bulb_width * 0.5
    bulb_end_pos = params.position + params.bulb_width * 0.5

    # Key heights
    neck_height = params.height * params.neck_ratio
    full_height = params.height

    curves = []

    # Curve 1: Start to neck start (mostly straight)
    p0 = np.array(start)
    p3 = np.array(start) + edge_vec * neck_start_pos
    p1 = p0 + edge_vec * neck_start_pos * 0.5
    p2 = p3 - edge_vec * neck_start_pos * 0.3
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 2: Neck start to bulb start (going up/out)
    p0 = curves[-1].p3
    p3 = np.array(start) + edge_vec * bulb_start_pos + normal * edge_length * full_height
    p1 = np.array(p0) + normal * edge_length * neck_height * params.curvature
    p2 = p3 - edge_unit * edge_length * params.bulb_width * 0.3 - normal * edge_length * full_height * 0.3
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 3: Bulb (rounded top)
    p0 = curves[-1].p3
    p3 = np.array(start) + edge_vec * bulb_end_pos + normal * edge_length * full_height
    # Control points create the rounded top
    p1 = (
        np.array(p0)
        + edge_unit * edge_length * params.bulb_width * 0.4
        + normal * edge_length * full_height * 0.2 * params.curvature
    )
    p2 = (
        np.array(p3)
        - edge_unit * edge_length * params.bulb_width * 0.4
        + normal * edge_length * full_height * 0.2 * params.curvature
    )
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 4: Bulb end to neck end to end (going back down)
    p0 = curves[-1].p3
    neck_end_point = np.array(start) + edge_vec * neck_end_pos
    p3 = np.array(end)
    p1 = np.array(p0) + edge_unit * edge_length * params.bulb_width * 0.3 - normal * edge_length * full_height * 0.3
    p2 = neck_end_point + normal * edge_length * neck_height * params.curvature
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    return curves


def generate_tab_edge_3_curves(
    start: Tuple[float, float], end: Tuple[float, float], params: TabParameters, is_blank: bool = False
) -> List[BezierCurve]:
    """Generate a puzzle piece edge with tab/blank using 3 cubic Bézier curves."""
    direction = 1 if not is_blank else -1
    edge_vec = np.array([end[0] - start[0], end[1] - start[1]])
    edge_length = np.linalg.norm(edge_vec)
    edge_unit = edge_vec / edge_length

    # Normal vector (perpendicular to edge)
    normal = np.array([-edge_unit[1], edge_unit[0]]) * direction

    # Key heights
    full_height = params.height

    curves = []

    # Key points
    bulb_left = (
        np.array(start) + edge_vec * (params.position - params.bulb_width * 0.5) + normal * edge_length * full_height
    )
    bulb_right = (
        np.array(start) + edge_vec * (params.position + params.bulb_width * 0.5) + normal * edge_length * full_height
    )

    # Curve 1: Start to left side of bulb
    p0 = np.array(start)
    p3 = bulb_left
    p1 = p0 + edge_vec * (params.position - params.neck_width) * 0.7
    p2 = p3 - edge_unit * edge_length * params.neck_width - normal * edge_length * full_height * 0.5 * params.curvature
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 2: Bulb top (left to right)
    p0 = curves[-1].p3
    p3 = bulb_right
    p1 = (
        np.array(p0)
        + edge_unit * edge_length * params.bulb_width * 0.5
        + normal * edge_length * full_height * 0.3 * params.curvature
    )
    p2 = (
        np.array(p3)
        - edge_unit * edge_length * params.bulb_width * 0.5
        + normal * edge_length * full_height * 0.3 * params.curvature
    )
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 3: Right side of bulb to end
    p0 = curves[-1].p3
    p3 = np.array(end)
    p1 = (
        np.array(p0)
        + edge_unit * edge_length * params.neck_width
        - normal * edge_length * full_height * 0.5 * params.curvature
    )
    p2 = p3 - edge_vec * (1 - params.position - params.neck_width) * 0.7
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    return curves


def generate_full_piece(
    size: float = 1.0, edge_types: List[str] | None = None, params: TabParameters | None = None
) -> List[List[BezierCurve]]:
    """Generate a complete puzzle piece with 4 edges."""
    if edge_types is None:
        edge_types = ["tab", "blank", "tab", "blank"]
    if params is None:
        params = TabParameters()

    corners = [
        (0, 0),  # Bottom-left
        (size, 0),  # Bottom-right
        (size, size),  # Top-right
        (0, size),  # Top-left
    ]

    edges = []
    for i, edge_type in enumerate(edge_types):
        start = corners[i]
        end = corners[(i + 1) % 4]

        if edge_type == "flat":
            # Just a straight line
            curve = BezierCurve(start, start, end, end)
            edges.append([curve])
        elif edge_type == "tab":
            edges.append(generate_tab_edge_3_curves(start, end, params, is_blank=False))
        else:  # blank
            edges.append(generate_tab_edge_3_curves(start, end, params, is_blank=True))

    return edges
