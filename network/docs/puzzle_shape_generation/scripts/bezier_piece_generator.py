#!/usr/bin/env python3
"""Bézier Curve Puzzle Piece Shape Generator.

This script generates realistic puzzle piece shapes using cubic Bézier curves.

Usage:
    # Generate a random puzzle piece PNG:
    python bezier_piece_generator.py

    # Generate with specific output path:
    python bezier_piece_generator.py --output my_piece.png

    # Generate comparison visualizations (original behavior):
    python bezier_piece_generator.py --compare
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


@dataclass
class BezierCurve:
    """A cubic Bézier curve defined by 4 control points."""

    p0: Tuple[float, float]  # Start point
    p1: Tuple[float, float]  # Control point 1
    p2: Tuple[float, float]  # Control point 2
    p3: Tuple[float, float]  # End point

    def evaluate(self, t: float) -> Tuple[float, float]:
        """Evaluate the curve at parameter t (0 to 1)."""
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * self.p0[0] + 3 * mt2 * t * self.p1[0] + 3 * mt * t2 * self.p2[0] + t3 * self.p3[0]
        y = mt3 * self.p0[1] + 3 * mt2 * t * self.p1[1] + 3 * mt * t2 * self.p2[1] + t3 * self.p3[1]
        return (x, y)

    def get_points(self, num_points: int = 50) -> np.ndarray:
        """Generate points along the curve."""
        t_values = np.linspace(0, 1, num_points)
        points = [self.evaluate(t) for t in t_values]
        return np.array(points)


@dataclass
class TabParameters:
    """Parameters controlling the shape of a puzzle piece tab/blank."""

    # Position along edge (0 to 1, typically 0.5 for center)
    position: float = 0.5

    # Width of the neck where tab connects to edge (relative to edge length)
    neck_width: float = 0.15

    # Width of the bulb at its widest point (relative to edge length)
    bulb_width: float = 0.25

    # Height of the tab (relative to edge length, positive = outward)
    height: float = 0.2

    # Neck height before bulb starts (relative to height)
    neck_ratio: float = 0.3

    # Curvature of the bulb (0 = pointed, 1 = very round)
    curvature: float = 0.7

    # Asymmetry factor (-1 to 1, 0 = symmetric)
    asymmetry: float = 0.0

    @classmethod
    def random(cls) -> "TabParameters":
        """Generate random but realistic parameters for a puzzle piece tab."""
        # Generate bulb_width first, then derive neck_width proportionally
        bulb_width = random.uniform(0.18, 0.32)
        # Neck should be 40-65% of bulb width to look properly attached
        neck_ratio_to_bulb = random.uniform(0.40, 0.65)
        neck_width = bulb_width * neck_ratio_to_bulb

        return cls(
            position=random.uniform(0.40, 0.60),  # Keep tabs more centered
            neck_width=neck_width,
            bulb_width=bulb_width,
            height=random.uniform(0.15, 0.28),  # Short to tall tabs
            neck_ratio=random.uniform(0.25, 0.45),  # Affects waist position
            curvature=random.uniform(0.5, 1.0),  # Rounder bulbs look more realistic
            asymmetry=random.uniform(-0.12, 0.12),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TabParameters":
        """Create from dictionary."""
        return cls(**data)


# Hardcoded color for all pieces
PIECE_COLOR = "#32CD32"  # Bright green (lime green)


@dataclass
class PieceConfig:
    """Configuration for a complete puzzle piece."""

    # Edge types: 'tab', 'blank', or 'flat'
    edge_types: List[str] = field(default_factory=lambda: ["tab", "blank", "tab", "blank"])
    # Parameters for each edge (if None, random params are used)
    edge_params: List[TabParameters | None] = field(default_factory=lambda: [None, None, None, None])
    # Piece size
    size: float = 1.0

    @classmethod
    def random(cls) -> "PieceConfig":
        """Generate a random piece configuration."""
        # Randomly choose edge types (but not all flat)
        edge_options = ["tab", "blank", "flat"]
        edge_types = [random.choice(edge_options) for _ in range(4)]

        # Ensure at least one non-flat edge
        if all(e == "flat" for e in edge_types):
            edge_types[random.randint(0, 3)] = random.choice(["tab", "blank"])

        # Generate random params for each edge
        edge_params: List[TabParameters | None] = [TabParameters.random() for _ in range(4)]

        return cls(
            edge_types=edge_types,
            edge_params=edge_params,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "edge_types": self.edge_types,
            "edge_params": [p.to_dict() if p else None for p in self.edge_params],
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PieceConfig":
        """Create from dictionary."""
        edge_params = [TabParameters.from_dict(p) if p else None for p in data.get("edge_params", [None] * 4)]
        return cls(
            edge_types=data.get("edge_types", ["tab", "blank", "tab", "blank"]),
            edge_params=edge_params,
            size=data.get("size", 1.0),
        )


def load_pieces_from_json(json_path: str | Path) -> List[PieceConfig]:
    """Load piece configurations from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    pieces = []
    for piece_data in data.get("pieces", []):
        pieces.append(PieceConfig.from_dict(piece_data))

    return pieces


def save_pieces_to_json(pieces: List[PieceConfig], json_path: str | Path) -> None:
    """Save piece configurations to a JSON file."""
    data = {"pieces": [p.to_dict() for p in pieces]}

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def generate_pieces_from_json(
    json_path: str | Path,
    output_dir: str | Path = ".",
    size_px: int = 512,
    transparent_bg: bool = True,
) -> List[Path]:
    """Generate PNG images for all pieces defined in a JSON file."""
    pieces = load_pieces_from_json(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for i, config in enumerate(pieces):
        output_path = output_dir / f"piece_{i + 1}.png"
        render_piece_to_png(config, output_path, size_px, transparent_bg)
        output_paths.append(output_path)
        print(f"Generated: {output_path}")

    return output_paths


def generate_tab_edge_2_curves(
    start: Tuple[float, float], end: Tuple[float, float], params: TabParameters, is_blank: bool = False
) -> List[BezierCurve]:
    """Generate a puzzle piece edge with tab/blank using 2 cubic Bézier curves.

    This is the simplest approach:
    - One curve for left half (start to top of tab)
    - One curve for right half (top of tab to end)
    """
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
    """Generate a puzzle piece edge with tab/blank using 4 cubic Bézier curves.

    More realistic approach:
    - Curve 1: Straight segment to neck start
    - Curve 2: Neck (transition from edge to bulb)
    - Curve 3: Bulb (rounded top)
    - Curve 4: Neck back down and straight to end
    """
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
    """Generate a puzzle piece edge with tab/blank using 3 cubic Bézier curves.

    Middle ground approach:
    - Curve 1: Start + neck going up
    - Curve 2: Bulb (rounded part)
    - Curve 3: Coming down + end
    """
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


def plot_curves(curves: List[BezierCurve], ax: Axes, color: str = "blue", label: str = "") -> None:
    """Plot a list of Bézier curves."""
    all_points = []
    for curve in curves:
        points = curve.get_points(30)
        all_points.append(points)

    # Plot the combined curve
    combined = np.vstack(all_points)
    ax.plot(combined[:, 0], combined[:, 1], color=color, linewidth=2, label=label)

    # Plot control points for visualization
    for _i, curve in enumerate(curves):
        control_x = [curve.p0[0], curve.p1[0], curve.p2[0], curve.p3[0]]
        control_y = [curve.p0[1], curve.p1[1], curve.p2[1], curve.p3[1]]
        ax.plot(control_x, control_y, "o--", color=color, alpha=0.3, markersize=4)


def generate_full_piece(
    size: float = 1.0, edge_types: List[str] | None = None, params: TabParameters | None = None
) -> List[List[BezierCurve]]:
    """Generate a complete puzzle piece with 4 edges.

    Args:
        size: Size of the piece (edge length)
        edge_types: List of 4 edge types ('tab', 'blank', or 'flat')
        params: Parameters for the tabs/blanks
    """
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


def generate_realistic_tab_edge(
    start: Tuple[float, float], end: Tuple[float, float], params: TabParameters, is_blank: bool = False
) -> List[BezierCurve]:
    """Generate a realistic puzzle piece tab using 5 cubic Bézier curves.

    This creates the classic "mushroom" shape with perfectly symmetric curves:
    - Curve 1: Edge to neck base left (flat entry)
    - Curve 2: Neck base left to bulb left (going up through waist)
    - Curve 3: Bulb left to bulb right (semicircular top)
    - Curve 4: Bulb right to neck base right (exact mirror of curve 2)
    - Curve 5: Neck base right to edge end (exact mirror of curve 1)

    The asymmetry parameter controls the tilt of the tab:
    - asymmetry > 0: bulb tilts right (in the direction of the edge)
    - asymmetry < 0: bulb tilts left (against the direction of the edge)
    - asymmetry = 0: perfectly symmetric tab
    """
    direction = 1 if not is_blank else -1
    edge_vec = np.array([end[0] - start[0], end[1] - start[1]])
    edge_length = np.linalg.norm(edge_vec)
    edge_unit = edge_vec / edge_length

    # Normal vector (perpendicular to edge)
    normal = np.array([-edge_unit[1], edge_unit[0]]) * direction

    # Key dimensions (all relative to edge_length)
    full_height = params.height * edge_length
    neck_half = params.neck_width * edge_length * 0.5
    bulb_half = params.bulb_width * edge_length * 0.5

    # Neck height (where the waist is narrowest) - this is key for the mushroom look
    neck_height = full_height * params.neck_ratio

    # Center of the tab (neck base)
    center = np.array(start) + edge_vec * params.position

    # Asymmetry: shift the bulb horizontally relative to the neck
    # Scale asymmetry by bulb_half to make the shift proportional to bulb size
    bulb_shift = edge_unit * bulb_half * params.asymmetry * 2.0

    # Asymmetric curve factors - one side curves more, the other less
    # This creates a more natural tilt appearance
    left_curve_factor = 1.0 + params.asymmetry * 0.5
    right_curve_factor = 1.0 - params.asymmetry * 0.5

    curves = []

    # Key points for the mushroom shape
    # The neck/waist points (symmetric relative to center)
    neck_base_left = center - edge_unit * neck_half
    neck_base_right = center + edge_unit * neck_half

    # The bulb points - shifted by asymmetry to create tilt
    bulb_center = center + bulb_shift + normal * neck_height
    bulb_base_left = bulb_center - edge_unit * bulb_half
    bulb_base_right = bulb_center + edge_unit * bulb_half

    # Bulb midpoints (where curves 2/4 meet curve 3)
    bulb_mid_left = bulb_base_left + normal * (full_height - neck_height) * 0.5
    bulb_mid_right = bulb_base_right + normal * (full_height - neck_height) * 0.5

    # Distance from start to neck and from neck to end (for symmetric flat sections)
    dist_start_to_neck = params.position - params.neck_width * 0.5
    dist_neck_to_end = 1.0 - params.position - params.neck_width * 0.5

    # Curve 1: Start to neck base left (flat entry)
    p0 = np.array(start)
    p3 = neck_base_left
    p1 = p0 + edge_unit * edge_length * dist_start_to_neck * 0.6
    p2 = p3 - edge_unit * neck_half * 0.5
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 2: Neck base left up through waist to bulb mid-left
    # This creates the critical S-curve: first INWARD (toward center), then OUTWARD (to bulb)
    p0 = curves[-1].p3
    p3 = bulb_mid_left

    # Calculate the waist pinch amount - how much the curve goes inward
    # More pinch for thinner necks relative to bulb (creates locking mechanism)
    pinch_amount = (bulb_half - neck_half) * 0.4

    # p1: Pull curve INWARD toward center and UP toward waist
    # This creates the first half of the S-curve (the inward pinch)
    p1 = (
        np.array(p0)
        + edge_unit * pinch_amount * left_curve_factor  # inward toward center
        + normal * neck_height * 0.7  # up toward waist height
    )

    # p2: Pull curve OUTWARD toward bulb edge
    # This creates the second half of the S-curve (outward to bulb)
    p2 = (
        np.array(p3)
        - normal * (full_height - neck_height) * 0.3  # down from bulb midpoint
        - edge_unit * (bulb_half - neck_half) * 0.3 * left_curve_factor  # slight outward bias
    )
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 3: Bulb mid-left up and around to bulb mid-right (semicircular)
    p0 = curves[-1].p3
    p3 = bulb_mid_right
    # Control points for a nice round bulb - approximating a semicircle
    bulb_radius = bulb_half
    p1 = np.array(p0) + normal * bulb_radius * 0.8 * params.curvature - edge_unit * bulb_radius * 0.1
    p2 = np.array(p3) + normal * bulb_radius * 0.8 * params.curvature + edge_unit * bulb_radius * 0.1
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 4: Bulb mid-right down through waist to neck base right (EXACT mirror of curve 2)
    # Mirror of the S-curve: first away from bulb, then INWARD toward center, then down to neck
    p0 = curves[-1].p3
    p3 = neck_base_right

    # p1: Mirror of curve 2's p2 - pull away from bulb
    p1 = (
        np.array(p0)
        - normal * (full_height - neck_height) * 0.3  # down from bulb midpoint
        + edge_unit * (bulb_half - neck_half) * 0.3 * right_curve_factor  # slight outward bias (mirrored)
    )

    # p2: Mirror of curve 2's p1 - pull INWARD toward center and down toward neck
    p2 = (
        np.array(p3)
        - edge_unit * pinch_amount * right_curve_factor  # inward toward center (mirrored direction)
        + normal * neck_height * 0.7  # up toward waist height
    )
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    # Curve 5: Neck base right to end (EXACT mirror of curve 1)
    p0 = curves[-1].p3
    p3 = np.array(end)
    # Mirror the control points of curve 1:
    # curve 1 p1 was: start + edge_unit * dist_start_to_neck * 0.6
    # curve 1 p2 was: neck_base_left - edge_unit * neck_half * 0.5
    # For the mirror, we reverse the structure
    p1 = p0 + edge_unit * neck_half * 0.5
    p2 = p3 - edge_unit * edge_length * dist_neck_to_end * 0.6
    curves.append(BezierCurve(tuple(p0), tuple(p1), tuple(p2), tuple(p3)))  # type: ignore

    return curves


def generate_piece_path(config: PieceConfig) -> Tuple[List[float], List[float]]:
    """Generate the complete path (x, y coordinates) for a puzzle piece.

    Returns:
        Tuple of (x_coords, y_coords) lists defining the piece outline.
    """
    corners = [
        (0, 0),  # Bottom-left
        (config.size, 0),  # Bottom-right
        (config.size, config.size),  # Top-right
        (0, config.size),  # Top-left
    ]

    all_x: List[float] = []
    all_y: List[float] = []

    for i, edge_type in enumerate(config.edge_types):
        start = corners[i]
        end = corners[(i + 1) % 4]

        # Get params for this edge
        params = config.edge_params[i]
        if params is None:
            params = TabParameters.random()

        if edge_type == "flat":
            # Straight line - just add a few points
            all_x.extend([start[0], end[0]])
            all_y.extend([start[1], end[1]])
        else:
            # Note: is_blank=True makes the shape go OUTWARD (tab), is_blank=False goes INWARD (blank)
            # So we invert: "tab" -> is_blank=True (outward), "blank" -> is_blank=False (inward)
            is_blank = edge_type == "tab"
            curves = generate_realistic_tab_edge(start, end, params, is_blank=is_blank)

            for curve in curves:
                points = curve.get_points(20)
                all_x.extend(points[:, 0].tolist())
                all_y.extend(points[:, 1].tolist())

    # Close the path
    all_x.append(all_x[0])
    all_y.append(all_y[0])

    return all_x, all_y


def render_piece_to_png(
    config: PieceConfig | None = None,
    output_path: str | Path = "piece.png",
    size_px: int = 512,
    transparent_bg: bool = True,
) -> Path:
    """Render a puzzle piece to a PNG file.

    Args:
        config: Piece configuration (if None, random config is generated)
        output_path: Path to save the PNG file
        size_px: Size of the output image in pixels
        transparent_bg: Whether to use transparent background

    Returns:
        Path to the saved PNG file.
    """
    if config is None:
        config = PieceConfig.random()

    # Generate the piece outline
    x_coords, y_coords = generate_piece_path(config)

    # Create figure with appropriate size and DPI
    dpi = 100
    fig_size = size_px / dpi
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)

    if transparent_bg:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    # Fill the piece
    ax.fill(x_coords, y_coords, color=PIECE_COLOR, edgecolor=PIECE_COLOR, linewidth=2)

    # Set equal aspect ratio and remove axes
    ax.set_aspect("equal")
    ax.axis("off")

    # Calculate bounds with some padding
    margin = 0.35 * config.size  # Account for tabs/blanks
    ax.set_xlim(-margin, config.size + margin)
    ax.set_ylim(-margin, config.size + margin)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.1,
        transparent=transparent_bg,
    )
    plt.close(fig)

    return output_path


def generate_random_piece(output_path: str | Path = "piece.png", size_px: int = 512) -> Path:
    """Generate a random puzzle piece and save it to a PNG file.

    This is the main entry point for generating random pieces.

    Args:
        output_path: Path to save the PNG file
        size_px: Size of the output image in pixels

    Returns:
        Path to the saved PNG file.
    """
    config = PieceConfig.random()
    return render_piece_to_png(config, output_path, size_px)


# Reference piece configurations (manually tuned to match pieces_2.webp)
# Edge order: bottom, right, top, left
REFERENCE_PIECES: dict[str, PieceConfig] = {
    "ref1": PieceConfig(
        edge_types=["tab", "blank", "blank", "tab"],
        edge_params=[
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.09, bulb_width=0.22, height=0.20, neck_ratio=0.40, curvature=0.80),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.22, height=0.20, neck_ratio=0.38, curvature=0.82),
        ],
    ),
    "ref2": PieceConfig(
        edge_types=["blank", "tab", "blank", "blank"],
        edge_params=[
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.26, height=0.24, neck_ratio=0.32, curvature=0.90),
            TabParameters(position=0.5, neck_width=0.09, bulb_width=0.22, height=0.18, neck_ratio=0.40, curvature=0.78),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.20, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.38, curvature=0.80),
        ],
    ),
    "ref3": PieceConfig(
        edge_types=["tab", "tab", "blank", "tab"],
        edge_params=[
            TabParameters(position=0.5, neck_width=0.09, bulb_width=0.22, height=0.20, neck_ratio=0.38, curvature=0.82),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.88),
            TabParameters(position=0.5, neck_width=0.11, bulb_width=0.26, height=0.24, neck_ratio=0.30, curvature=0.90),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.22, height=0.20, neck_ratio=0.40, curvature=0.80),
        ],
    ),
    "ref4": PieceConfig(
        edge_types=["tab", "tab", "blank", "tab"],
        edge_params=[
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.26, height=0.24, neck_ratio=0.32, curvature=0.88),
            TabParameters(position=0.5, neck_width=0.09, bulb_width=0.22, height=0.20, neck_ratio=0.38, curvature=0.82),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
        ],
    ),
    "ref5": PieceConfig(
        edge_types=["tab", "blank", "blank", "tab"],
        edge_params=[
            TabParameters(position=0.5, neck_width=0.11, bulb_width=0.28, height=0.26, neck_ratio=0.30, curvature=0.92),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.26, height=0.24, neck_ratio=0.32, curvature=0.88),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
        ],
    ),
    "ref6": PieceConfig(
        edge_types=["tab", "tab", "tab", "tab"],
        edge_params=[
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.22, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.09, bulb_width=0.22, height=0.18, neck_ratio=0.40, curvature=0.78),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.24, height=0.20, neck_ratio=0.35, curvature=0.85),
            TabParameters(position=0.5, neck_width=0.10, bulb_width=0.22, height=0.20, neck_ratio=0.38, curvature=0.82),
        ],
    ),
}


def _extract_and_normalize_contour(image_path: Path) -> Tuple[np.ndarray, np.ndarray] | None:
    """Extract contour from reference image and normalize to unit coordinates.

    Returns:
        Tuple of (x_coords, y_coords) normalized to roughly [0, 1] range, or None if failed.
    """
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Get alpha channel or convert to grayscale
    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    contour = largest.reshape(-1, 2).astype(np.float64)

    # Normalize: center and scale to fit in [0, 1] range
    min_pt = contour.min(axis=0)
    max_pt = contour.max(axis=0)
    size = (max_pt - min_pt).max()

    if size > 0:
        # Center the contour
        center = (min_pt + max_pt) / 2
        contour = contour - center
        # Scale to unit size
        contour = contour / size
        # Shift to [0, 1] centered at 0.5
        contour = contour + 0.5

    # Flip Y axis (image coordinates are inverted)
    contour[:, 1] = 1.0 - contour[:, 1]

    return contour[:, 0], contour[:, 1]


def generate_reference_comparison(
    output_path: str | Path = "../outputs/reference_comparison.png",
    json_path: str | Path | None = None,
) -> Path:
    """Generate a side-by-side comparison of reference pieces and generated matches.

    Args:
        output_path: Path to save the comparison image
        json_path: Path to JSON config file. If None, uses ../reference_pieces.json
    """
    from PIL import Image

    output_path = Path(output_path)
    ref_dir = Path(__file__).parent.parent / "reference_images" / "standardized"

    # Load pieces from JSON file
    if json_path is None:
        json_path = Path(__file__).parent.parent / "reference_pieces.json"
    pieces = load_pieces_from_json(json_path)

    num_pieces = len(pieces)
    fig, axes = plt.subplots(3, num_pieces, figsize=(3 * num_pieces, 9))
    fig.suptitle("Reference vs Generated Pieces", fontsize=16, fontweight="bold")

    for i, config in enumerate(pieces):
        # Top row: reference pieces
        ref_img_path = ref_dir / f"piece_{i + 1}.png"
        if ref_img_path.exists():
            ref_img = Image.open(ref_img_path)
            axes[0, i].imshow(ref_img)
        axes[0, i].set_title(f"Reference {i + 1}")
        axes[0, i].axis("off")

        # Middle row: generated pieces from JSON
        x_coords, y_coords = generate_piece_path(config)

        axes[1, i].fill(x_coords, y_coords, color=PIECE_COLOR, edgecolor=PIECE_COLOR, linewidth=2)
        axes[1, i].set_aspect("equal")
        axes[1, i].axis("off")
        margin = 0.35 * config.size
        axes[1, i].set_xlim(-margin, config.size + margin)
        axes[1, i].set_ylim(-margin, config.size + margin)
        axes[1, i].set_title(f"Generated {i + 1}")

        # Bottom row: overlapped comparison
        ax = axes[2, i]

        # Normalize generated piece to [0, 1] range for comparison
        gen_x = np.array(x_coords)
        gen_y = np.array(y_coords)
        gen_min = min(gen_x.min(), gen_y.min())
        gen_max = max(gen_x.max(), gen_y.max())
        gen_size = gen_max - gen_min
        if gen_size > 0:
            gen_center_x = (gen_x.min() + gen_x.max()) / 2
            gen_center_y = (gen_y.min() + gen_y.max()) / 2
            gen_x_norm = (gen_x - gen_center_x) / gen_size + 0.5
            gen_y_norm = (gen_y - gen_center_y) / gen_size + 0.5
        else:
            gen_x_norm, gen_y_norm = gen_x, gen_y

        # Extract and normalize reference contour
        ref_contour = _extract_and_normalize_contour(ref_img_path)

        if ref_contour is not None:
            ref_x, ref_y = ref_contour
            # Plot reference shape (blue, semi-transparent)
            ax.fill(ref_x, ref_y, color="blue", alpha=0.4, label="Reference")
            ax.plot(ref_x, ref_y, color="blue", linewidth=1.5, alpha=0.8)

        # Plot generated shape (red, semi-transparent)
        ax.fill(gen_x_norm, gen_y_norm, color="red", alpha=0.4, label="Generated")
        ax.plot(gen_x_norm, gen_y_norm, color="red", linewidth=1.5, alpha=0.8)

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title(f"Overlap {i + 1}")

    # Add legend to first overlap plot
    axes[2, 0].legend(loc="upper left", fontsize=8)

    axes[0, 0].set_ylabel("Reference", fontsize=12)
    axes[1, 0].set_ylabel("Generated", fontsize=12)
    axes[2, 0].set_ylabel("Overlap", fontsize=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def main() -> None:
    """Generate a random puzzle piece and save it to a PNG file."""
    parser = argparse.ArgumentParser(
        description="Generate realistic puzzle piece shapes using Bézier curves.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bezier_piece_generator.py                    # Generate random piece
  python bezier_piece_generator.py -o my_piece.png    # Custom output path
  python bezier_piece_generator.py --size 256         # Smaller image
  python bezier_piece_generator.py --preset ref1      # Use reference preset
  python bezier_piece_generator.py --compare          # Compare with references
  python bezier_piece_generator.py --json pieces.json # Generate from JSON config
  python bezier_piece_generator.py --export-json out.json  # Export presets to JSON
""",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="piece.png",
        help="Output path for the generated piece (default: piece.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated pieces when using --json",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Size of the output image in pixels (default: 512)",
    )
    parser.add_argument(
        "--no-transparent",
        action="store_true",
        help="Use white background instead of transparent",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(REFERENCE_PIECES.keys()),
        help="Use a preset configuration (ref1-ref6)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison with reference pieces",
    )
    parser.add_argument(
        "--json",
        type=str,
        metavar="FILE",
        help="Generate pieces from a JSON configuration file",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        metavar="FILE",
        help="Export the reference presets to a JSON file for editing",
    )

    args = parser.parse_args()

    if args.export_json:
        pieces = [REFERENCE_PIECES[f"ref{i}"] for i in range(1, 7)]
        save_pieces_to_json(pieces, args.export_json)
        print(f"Exported {len(pieces)} piece configurations to: {args.export_json}")
    elif args.json:
        generate_pieces_from_json(
            args.json,
            output_dir=args.output_dir,
            size_px=args.size,
            transparent_bg=not args.no_transparent,
        )
    elif args.compare:
        output_path = generate_reference_comparison()
        print(f"Generated reference comparison: {output_path}")
    else:
        config = REFERENCE_PIECES.get(args.preset) if args.preset else None
        output_path = render_piece_to_png(
            config=config,
            output_path=args.output,
            size_px=args.size,
            transparent_bg=not args.no_transparent,
        )
        print(f"Generated puzzle piece: {output_path}")


if __name__ == "__main__":
    main()
