"""PyTorch-based geometric logic for generating puzzle piece shapes.

This module provides differentiable Bezier curve generation for use with
gradient-based optimization. All operations use PyTorch tensors and maintain
gradients through the computation graph.
"""

from typing import List, Tuple

import torch
from models import PieceConfig, TabParameters

# Kappa constant for quarter-circle approximation with cubic Bezier
KAPPA = 0.5522847498


def bezier_curve_torch(
    control_points: torch.Tensor,
    num_points: int = 20,
) -> torch.Tensor:
    """Evaluate a cubic Bezier curve at uniformly spaced parameter values.

    Args:
        control_points: (4, 2) tensor of control points [p0, p1, p2, p3].
        num_points: Number of points to sample along the curve.

    Returns:
        (num_points, 2) tensor of curve points.
    """
    device = control_points.device
    dtype = control_points.dtype

    t = torch.linspace(0, 1, num_points, device=device, dtype=dtype)
    t = t.unsqueeze(1)  # (N, 1)
    omt = 1 - t  # one minus t

    # Bernstein basis polynomials for cubic Bezier
    coeffs = torch.cat(
        [
            omt**3,
            3 * omt**2 * t,
            3 * omt * t**2,
            t**3,
        ],
        dim=1,
    )  # (N, 4)

    return coeffs @ control_points  # (N, 2)


def generate_tab_edge_torch(
    start: torch.Tensor,
    end: torch.Tensor,
    params: torch.Tensor,
    is_blank: bool = False,
    edge_type: str = "tab",
    num_points_per_curve: int = 20,
) -> torch.Tensor:
    """Generate a puzzle tab/blank edge using 6 cubic Bezier curves.

    This is the PyTorch equivalent of generate_realistic_tab_edge.

    Args:
        start: (2,) tensor for edge start point.
        end: (2,) tensor for edge end point.
        params: (12,) tensor of tab parameters in order:
            [position, neck_width, bulb_width, height, neck_ratio,
             curvature, asymmetry, corner_slope, squareness, neck_flare,
             shoulder_offset, shoulder_flatness]
        is_blank: Whether this is a blank (indent) instead of tab (protrusion).
        edge_type: "tab" or "blank" for corner slope direction.
        num_points_per_curve: Points to sample per Bezier curve.

    Returns:
        (N, 2) tensor of contour points for this edge.
    """
    device = start.device
    dtype = start.dtype

    # Unpack parameters
    position = params[0]
    neck_width = params[1]
    bulb_width = params[2]
    height = params[3]
    neck_ratio = params[4]
    curvature = params[5]
    asymmetry = params[6]
    corner_slope = params[7]
    squareness = params[8]
    neck_flare = params[9]
    shoulder_offset = params[10]
    shoulder_flatness = params[11]

    direction = 1.0 if not is_blank else -1.0

    edge_vec = end - start
    edge_length = torch.norm(edge_vec)
    edge_unit = edge_vec / (edge_length + 1e-8)

    # Normal vector pointing toward feature
    normal = torch.tensor([-edge_unit[1], edge_unit[0]], device=device, dtype=dtype) * direction

    # Piece's outward normal
    piece_outward_normal = torch.tensor([-edge_unit[1], edge_unit[0]], device=device, dtype=dtype)

    # Corner slope direction differs for tabs vs blanks
    corner_sign = -1.0 if edge_type == "tab" else 1.0

    entry_corner_offset = piece_outward_normal * corner_slope * corner_sign
    exit_corner_offset = piece_outward_normal * corner_slope * (-corner_sign)

    # Key dimensions (relative to edge_length)
    full_height = height * edge_length
    neck_half = neck_width * edge_length * 0.5
    bulb_half = bulb_width * edge_length * 0.5
    neck_height = full_height * neck_ratio

    # Center of the tab
    center = start + edge_vec * position

    # Asymmetry: shift the bulb horizontally
    bulb_shift = edge_unit * bulb_half * asymmetry * 2.0

    # Asymmetric curve factors
    left_curve_factor = 1.0 + asymmetry * 0.5
    right_curve_factor = 1.0 - asymmetry * 0.5

    # Shoulder offset: displacement of neck base points from the corner-to-corner line
    # The offset opposes the feature direction:
    # - Tabs (normal points outward): neck base moves inward (dip before rising)
    # - Blanks (normal points inward): neck base moves outward (hump before dipping)
    shoulder_offset_vec = -normal * shoulder_offset * edge_length

    # Key points for the mushroom shape
    neck_base_left = center - edge_unit * neck_half + shoulder_offset_vec
    neck_base_right = center + edge_unit * neck_half + shoulder_offset_vec

    bulb_center = center + bulb_shift + normal * neck_height
    bulb_base_left = bulb_center - edge_unit * bulb_half
    bulb_base_right = bulb_center + edge_unit * bulb_half

    bulb_mid_left = bulb_base_left + normal * (full_height - neck_height) * 0.5
    bulb_mid_right = bulb_base_right + normal * (full_height - neck_height) * 0.5

    dist_start_to_neck = position - neck_width * 0.5
    dist_neck_to_end = 1.0 - position - neck_width * 0.5

    # Neck flare amount
    flare_amount = (bulb_half - neck_half) * neck_flare

    all_points = []

    # Curve 1: Start to neck base left
    # shoulder_flatness controls how flat the shoulder stays before turning into neck
    # Higher flatness = longer flat section + sharper "armpit" turn
    shoulder_extend = 0.5 + shoulder_flatness * 0.4  # 0.5-0.9: how far p1 extends along edge
    neck_turn_tightness = 0.5 * (1.0 - shoulder_flatness * 0.7)  # 0.5-0.15: how tight the turn
    p0 = start
    p3 = neck_base_left
    p1 = p0 + edge_unit * edge_length * dist_start_to_neck * shoulder_extend + entry_corner_offset * edge_length * 0.3
    p2 = p3 - edge_unit * neck_half * neck_turn_tightness
    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points_per_curve)
    all_points.append(pts[:-1])  # Exclude last to avoid duplication

    # Curve 2: Neck base left up through waist to bulb mid-left
    p0 = p3
    p3 = bulb_mid_left
    p1 = p0 + edge_unit * flare_amount * left_curve_factor + normal * neck_height * 0.7
    p2 = p3 - normal * (full_height - neck_height) * 0.3
    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points_per_curve)
    all_points.append(pts[:-1])

    # Curve 3a: Bulb mid-left to apex
    bulb_chord_center = (bulb_mid_left + bulb_mid_right) * 0.5
    bulb_radius = bulb_half
    bulb_apex = bulb_chord_center + normal * bulb_radius

    p0 = p3
    p3 = bulb_apex
    p1 = p0 + normal * bulb_radius * KAPPA * curvature * squareness
    p2 = p3 - edge_unit * bulb_radius * KAPPA * curvature * squareness
    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points_per_curve)
    all_points.append(pts[:-1])

    # Curve 3b: Bulb apex to bulb mid-right
    p0 = p3
    p3 = bulb_mid_right
    p1 = p0 + edge_unit * bulb_radius * KAPPA * curvature * squareness
    p2 = p3 + normal * bulb_radius * KAPPA * curvature * squareness
    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points_per_curve)
    all_points.append(pts[:-1])

    # Curve 5: Bulb mid-right down through waist to neck base right
    p0 = p3
    p3 = neck_base_right
    p1 = p0 - normal * (full_height - neck_height) * 0.3
    p2 = p3 - edge_unit * flare_amount * right_curve_factor + normal * neck_height * 0.7
    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points_per_curve)
    all_points.append(pts[:-1])

    # Curve 6: Neck base right to end
    # Mirror of Curve 1 - use same shoulder_flatness for symmetric appearance
    p0 = p3
    p3 = end
    p1 = p0 + edge_unit * neck_half * neck_turn_tightness
    p2 = p3 - edge_unit * edge_length * dist_neck_to_end * shoulder_extend + exit_corner_offset * edge_length * 0.3
    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points_per_curve)
    all_points.append(pts[:-1])

    return torch.cat(all_points, dim=0)


def generate_corner_curve_torch(
    corner: torch.Tensor,
    incoming_dir: torch.Tensor,
    outgoing_dir: torch.Tensor,
    radius: torch.Tensor,
    num_points: int = 10,
) -> torch.Tensor:
    """Generate a rounded corner using cubic Bezier approximation.

    Args:
        corner: (2,) tensor for corner position.
        incoming_dir: (2,) unit vector of incoming edge direction.
        outgoing_dir: (2,) unit vector of outgoing edge direction.
        radius: Scalar corner radius.
        num_points: Points to sample for the corner curve.

    Returns:
        (N, 2) tensor of corner curve points.
    """
    # Start point: offset from corner along incoming edge direction (backward)
    p0 = corner - incoming_dir * radius
    # End point: offset from corner along outgoing edge direction (forward)
    p3 = corner + outgoing_dir * radius

    # Control points for quarter-circle approximation
    p1 = p0 + incoming_dir * radius * KAPPA
    p2 = p3 - outgoing_dir * radius * KAPPA

    ctrl_pts = torch.stack([p0, p1, p2, p3])
    pts = bezier_curve_torch(ctrl_pts, num_points)

    return pts[:-1]  # Exclude last to avoid duplication


def generate_flat_edge_torch(
    start: torch.Tensor,
    end: torch.Tensor,
    num_points: int = 10,
) -> torch.Tensor:
    """Generate a flat edge as a straight line.

    Args:
        start: (2,) tensor for edge start.
        end: (2,) tensor for edge end.
        num_points: Points to sample.

    Returns:
        (N, 2) tensor of edge points.
    """
    device = start.device
    dtype = start.dtype

    t = torch.linspace(0, 1, num_points, device=device, dtype=dtype)
    t = t.unsqueeze(1)  # (N, 1)

    points = start.unsqueeze(0) * (1 - t) + end.unsqueeze(0) * t
    return points[:-1]  # Exclude last to avoid duplication


def generate_piece_path_torch(
    edge_params_list: List[torch.Tensor],
    config: PieceConfig,
    device: torch.device,
    points_per_curve: int = 20,
) -> torch.Tensor:
    """Generate the complete contour path for a puzzle piece.

    Args:
        edge_params_list: List of (11,) tensors for each edge that has tab/blank.
                         Order: [edge0_params, edge1_params, ...] for non-flat edges.
        config: PieceConfig with edge_types, size, corner_radius.
        device: Torch device.
        points_per_curve: Points per Bezier curve segment.

    Returns:
        (N, 2) tensor of contour points forming a closed path.
    """
    dtype = torch.float32

    size = config.size
    radius = config.corner_radius * size

    # Define corners
    corners = torch.tensor(
        [
            [0.0, 0.0],  # Bottom-left
            [size, 0.0],  # Bottom-right
            [size, size],  # Top-right
            [0.0, size],  # Top-left
        ],
        device=device,
        dtype=dtype,
    )

    # Edge direction vectors
    edge_dirs = torch.tensor(
        [
            [1.0, 0.0],  # Bottom edge: left to right
            [0.0, 1.0],  # Right edge: bottom to top
            [-1.0, 0.0],  # Top edge: right to left
            [0.0, -1.0],  # Left edge: top to bottom
        ],
        device=device,
        dtype=dtype,
    )

    all_points = []
    param_idx = 0  # Track which parameter tensor to use

    for i in range(4):
        # Generate corner curve
        prev_edge_dir = edge_dirs[(i - 1) % 4]
        curr_edge_dir = edge_dirs[i]

        corner_pts = generate_corner_curve_torch(
            corners[i],
            incoming_dir=prev_edge_dir,
            outgoing_dir=curr_edge_dir,
            radius=torch.tensor(radius, device=device, dtype=dtype),
            num_points=10,
        )
        all_points.append(corner_pts)

        # Calculate edge start/end (offset by corner radius)
        edge_start = corners[i] + curr_edge_dir * radius
        edge_end = corners[(i + 1) % 4] - curr_edge_dir * radius

        edge_type = config.edge_types[i]

        if edge_type == "flat":
            flat_pts = generate_flat_edge_torch(edge_start, edge_end, num_points=20)
            all_points.append(flat_pts)
        else:
            # tab or blank
            params = edge_params_list[param_idx]
            param_idx += 1

            is_blank = edge_type == "blank"
            edge_pts = generate_tab_edge_torch(
                edge_start,
                edge_end,
                params,
                is_blank=is_blank,
                edge_type=edge_type,
                num_points_per_curve=points_per_curve,
            )
            all_points.append(edge_pts)

    contour = torch.cat(all_points, dim=0)

    # Close the path by appending the first point
    contour = torch.cat([contour, contour[:1]], dim=0)

    return contour


def params_to_tensor(
    config: PieceConfig,
    param_names: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """Convert PieceConfig parameters to a flat tensor for optimization.

    Args:
        config: PieceConfig with edge_params.
        param_names: Names of parameters to extract (e.g., ["neck_width", "height", ...]).
        device: Torch device.

    Returns:
        Tuple of:
            - (K,) tensor of parameter values (K = num_non_flat_edges * len(param_names))
            - List of edge indices that have tab/blank (for reconstruction)
    """
    params = []
    edge_indices = []

    for i, edge_params in enumerate(config.edge_params):
        if config.edge_types[i] != "flat" and edge_params is not None:
            edge_indices.append(i)
            for name in param_names:
                params.append(getattr(edge_params, name))

    return torch.tensor(params, device=device, dtype=torch.float32), edge_indices


def tensor_to_edge_params_list(
    vector: torch.Tensor,
    config: PieceConfig,
    param_names: List[str],
    edge_indices: List[int],
) -> List[torch.Tensor]:
    """Convert flat parameter vector to list of edge parameter tensors.

    This function properly maintains gradient connections by using torch operations
    instead of creating new tensors.

    Args:
        vector: Flat parameter tensor of shape (K,) with gradients.
        config: Original PieceConfig for defaults.
        param_names: Names of parameters in the vector.
        edge_indices: Which edges have parameters in the vector.

    Returns:
        List of (10,) tensors, one per non-flat edge, with gradients preserved.
    """
    device = vector.device
    dtype = vector.dtype

    num_params = len(param_names)
    edge_params_list = []

    all_param_names = [
        "position",
        "neck_width",
        "bulb_width",
        "height",
        "neck_ratio",
        "curvature",
        "asymmetry",
        "corner_slope",
        "squareness",
        "neck_flare",
        "shoulder_offset",
        "shoulder_flatness",
    ]

    # Create a mapping from param name to index in param_names
    param_name_to_idx = {name: i for i, name in enumerate(param_names)}

    for idx, edge_idx in enumerate(edge_indices):
        # Start with default values from config
        edge_params = config.edge_params[edge_idx]
        if edge_params is None:
            edge_params = TabParameters()

        # Build full 10-param tensor by stacking individual values
        # This preserves gradients for optimized parameters
        full_params_list = []
        param_offset = idx * num_params

        for name in all_param_names:
            if name in param_name_to_idx:
                # Use optimized value - slice to preserve gradient
                name_idx = param_name_to_idx[name]
                full_params_list.append(vector[param_offset + name_idx : param_offset + name_idx + 1])
            else:
                # Use default from config as a tensor
                default_val = getattr(edge_params, name)
                full_params_list.append(torch.tensor([default_val], device=device, dtype=dtype))

        # Stack all params - this preserves gradients
        full_params = torch.cat(full_params_list, dim=0)
        edge_params_list.append(full_params)

    return edge_params_list
