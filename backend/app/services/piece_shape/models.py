"""Data models for Bezier puzzle pieces.

This module is ported from network/docs/puzzle_shape_generation/scripts/models.py
"""

import random
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class BezierCurve:
    """A cubic Bezier curve defined by 4 control points."""

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

    # Corner slope: controls tangent angle at corners (0 = 90 degrees, higher = more slope)
    # Direction is automatic: tabs slope down toward bulge, blanks slope up from indent
    corner_slope: float = 0.10

    # Squareness: how flat/square the bulb top is (1.0 = circular, >1.0 = flatter/squarer)
    squareness: float = 1.0

    # Neck flare: controls neck shape (-1 = max flare outward, 0 = straight, 1 = max pinch inward)
    neck_flare: float = 0.4

    # Shoulder offset: vertical displacement of neck base from corner-to-corner line
    # For tabs: creates a "dip" before the tab rises (neck base moves inward)
    # For blanks: creates a "hump" before the indent dips (neck base moves outward)
    # The sign is automatically applied based on feature type.
    shoulder_offset: float = 0.04

    # Shoulder flatness: controls how flat the shoulder stays before sharply turning
    # into the neck. Higher values = flatter shoulder + sharper "armpit" transition.
    # 0.0 = smooth flowing curve, 1.0 = very flat with sharp turn into neck
    shoulder_flatness: float = 0.5

    @classmethod
    def random(cls) -> "TabParameters":
        """Generate random but realistic parameters for a puzzle piece tab."""
        # Generate bulb_width first, then derive neck_width proportionally
        bulb_width = random.uniform(0.25, 0.42)  # Increased for larger tabs
        neck_ratio_to_bulb = random.uniform(0.50, 0.7)
        neck_width = bulb_width * neck_ratio_to_bulb

        return cls(
            position=random.uniform(0.40, 0.60),  # Keep tabs more centered
            neck_width=neck_width,
            bulb_width=bulb_width,
            height=random.uniform(0.18, 0.38),  # Increased for taller tabs
            neck_ratio=random.uniform(0.25, 0.45),  # Affects waist position
            curvature=random.uniform(0.5, 1.0),  # Rounder bulbs look more realistic
            asymmetry=random.uniform(-0.12, 0.12),
            corner_slope=random.uniform(0.05, 0.20),
            squareness=random.uniform(1.0, 1.4),  # Slightly flat to more square tops
            neck_flare=random.uniform(-0.3, 0.5),  # Flare outward to pinch inward
            shoulder_offset=random.uniform(0.03, 0.06),  # Subtle dip/hump at neck base
            shoulder_flatness=random.uniform(0.4, 0.8),  # Flatter shoulder, sharper armpit
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TabParameters":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PieceConfig:
    """Configuration for a complete puzzle piece."""

    # Edge types: 'tab', 'blank', or 'flat'
    edge_types: List[str] = field(default_factory=lambda: ["tab", "blank", "tab", "blank"])
    # Parameters for each edge (if None, random params are used)
    edge_params: List[Optional[TabParameters]] = field(default_factory=lambda: [None, None, None, None])
    # Piece size
    size: float = 1.0
    # Corner radius for rounded corners (relative to piece size)
    corner_radius: float = 0.04

    @classmethod
    def random(cls) -> "PieceConfig":
        """Generate a random piece configuration."""
        # Randomly choose edge types (but not all flat)
        edge_options = ["tab", "blank", "flat"]
        edge_types = [random.choice(edge_options) for _ in range(4)]

        # Ensure valid flat edge configuration:
        # - Maximum 2 flat edges
        # - If 2 flat edges, they must be adjacent (indices differ by 1, or 0 and 3)
        flat_indices = [i for i, e in enumerate(edge_types) if e == "flat"]

        while len(flat_indices) > 2 or (
            len(flat_indices) == 2 and abs(flat_indices[1] - flat_indices[0]) not in (1, 3)
        ):
            idx = random.choice(flat_indices)
            edge_types[idx] = random.choice(["tab", "blank"])
            flat_indices.remove(idx)

        # Generate random params for each edge
        edge_params: List[Optional[TabParameters]] = [TabParameters.random() for _ in range(4)]

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
            "corner_radius": self.corner_radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PieceConfig":
        """Create from dictionary."""
        edge_params = [TabParameters.from_dict(p) if p else None for p in data.get("edge_params", [None] * 4)]
        return cls(
            edge_types=data.get("edge_types", ["tab", "blank", "tab", "blank"]),
            edge_params=edge_params,
            size=data.get("size", 1.0),
            corner_radius=data.get("corner_radius", 0.04),
        )
