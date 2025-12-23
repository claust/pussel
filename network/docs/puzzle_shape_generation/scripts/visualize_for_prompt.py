#!/usr/bin/env python3
"""Script to generate an illustration of the Bézier curve structure for a puzzle piece tab."""
import matplotlib.pyplot as plt
import numpy as np
from geometry import generate_realistic_tab_edge
from models import TabParameters


def create_prompt_illustration():
    """Create and save an illustration showing how Bézier curves form a puzzle tab."""
    # Parameters similar to a reference piece
    params = TabParameters(
        position=0.5,
        neck_width=0.10,
        bulb_width=0.32,
        height=0.30,
        neck_ratio=0.35,
        curvature=0.95,
        asymmetry=-0.05,
        corner_slope=0.12,
        shoulder_offset=0.04,  # Dip before tab rises
        shoulder_flatness=0.6,  # Flatter shoulder, sharper armpit
    )

    curves = generate_realistic_tab_edge((0, 0), (1, 0), params, is_blank=False)

    fig, ax = plt.subplots(figsize=(14, 10))

    colors = ["#FF9999", "#66B2FF", "#99FF99", "#90EE90", "#FFCC99", "#BC8F8F"]
    edge_color = "#333333"

    # Plot the full piece edge for context
    ax.axhline(y=0, color="black", linewidth=2, alpha=0.3)

    for i, curve in enumerate(curves):
        points = curve.get_points(100)
        # Plot the curve itself
        ax.plot(points[:, 0], points[:, 1], color=edge_color, linewidth=4, zorder=3)

        # Plot control polygon
        ctrl = np.array([curve.p0, curve.p1, curve.p2, curve.p3])
        ax.plot(ctrl[:, 0], ctrl[:, 1], "--", color=colors[i], alpha=0.6, linewidth=1.5, zorder=2)

        # Plot control points
        # p0 and p3 are anchor points (shared between curves)
        # p1 and p2 are handle points
        s_anchor = 100
        s_handle = 60

        ax.scatter(
            [curve.p0[0], curve.p3[0]],
            [curve.p0[1], curve.p3[1]],
            color=colors[i],
            s=s_anchor,
            edgecolors="black",
            zorder=5,
            label=f"Curve {i + 1} Anchors" if i == 0 else "",
        )
        ax.scatter(
            [curve.p1[0], curve.p2[0]],
            [curve.p1[1], curve.p2[1]],
            color=colors[i],
            s=s_handle,
            marker="s",
            edgecolors="black",
            alpha=0.8,
            zorder=4,
            label=f"Curve {i + 1} Handles" if i == 0 else "",
        )

        # Labels
        for j, (x, y) in enumerate(ctrl):
            label = f"C{i + 1}.P{j}"
            ax.annotate(
                label,
                (x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=edge_color,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
            )

    # Highlight specific parameter effects
    # Neck width
    ax.annotate(
        "",
        xy=(0.5 - params.neck_width / 2, 0.02),
        xytext=(0.5 + params.neck_width / 2, 0.02),
        arrowprops={"arrowstyle": "<->", "color": "blue"},
    )
    ax.text(0.5, 0.03, "neck_width", ha="center", color="blue", fontweight="bold")

    # Bulb width
    bulb_y = params.height * 0.7
    ax.annotate(
        "",
        xy=(0.5 - params.bulb_width / 2 + params.asymmetry * params.bulb_width, bulb_y),
        xytext=(0.5 + params.bulb_width / 2 + params.asymmetry * params.bulb_width, bulb_y),
        arrowprops={"arrowstyle": "<->", "color": "red"},
    )
    ax.text(
        0.5 + params.asymmetry * params.bulb_width,
        bulb_y + 0.01,
        "bulb_width",
        ha="center",
        color="red",
        fontweight="bold",
    )

    # Shoulder offset (dip below baseline for tabs)
    neck_base_y = -params.shoulder_offset  # For tabs, neck base is below baseline
    ax.annotate(
        "",
        xy=(0.35, 0),
        xytext=(0.35, neck_base_y),
        arrowprops={"arrowstyle": "<->", "color": "green"},
    )
    ax.text(0.32, neck_base_y / 2, "shoulder_offset", ha="right", color="green", fontweight="bold", fontsize=9)

    # Draw a dashed line at baseline for reference
    ax.axhline(y=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(0.02, 0.01, "baseline (corner-to-corner)", fontsize=8, color="gray", alpha=0.7)

    ax.set_title("Current Puzzle Piece Generation Logic (6 Cubic Bézier Curves)", fontsize=16, pad=20)
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 0.45)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path = "../outputs/bezier_structure_for_prompt.png"
    plt.savefig(output_path, dpi=200)
    print(f"Saved illustration to: {output_path}")


if __name__ == "__main__":
    create_prompt_illustration()
