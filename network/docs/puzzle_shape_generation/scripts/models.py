"""Data models for Bézier puzzle pieces."""

import random
from dataclasses import asdict, dataclass, field
from typing import Any, List, Tuple

import numpy as np


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

    # Corner slope: controls tangent angle at corners (0 = 90°, higher = more slope)
    # Direction is automatic: tabs slope down toward bulge, blanks slope up from indent
    corner_slope: float = 0.10

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
            corner_slope=random.uniform(0.05, 0.15),
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
