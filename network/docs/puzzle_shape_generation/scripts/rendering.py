"""Rendering and visualization for puzzle pieces."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from io_utils import load_pieces_from_json
from matplotlib.axes import Axes
from puzzle_shapes import BezierCurve, PieceConfig, generate_piece_path

# Hardcoded color for all pieces
PIECE_COLOR = "#32CD32"  # Bright green (lime green)


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


def render_piece_to_png(
    config: PieceConfig | None = None,
    output_path: str | Path = "piece.png",
    size_px: int = 512,
    transparent_bg: bool = True,
) -> Path:
    """Render a puzzle piece to a PNG file."""
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
    """Generate a random puzzle piece and save it to a PNG file."""
    config = PieceConfig.random()
    return render_piece_to_png(config, output_path, size_px)


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
