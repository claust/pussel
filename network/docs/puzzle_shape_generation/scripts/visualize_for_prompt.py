#!/usr/bin/env python3
"""Script to generate an illustration of the Bézier curve structure for a puzzle piece tab."""
import matplotlib.pyplot as plt
import numpy as np
from puzzle_shapes import TabParameters, generate_realistic_tab_edge


def create_prompt_illustration():
    """Create and save an illustration showing how Bézier curves form a puzzle tab."""
    # Parameters that showcase all features clearly
    params = TabParameters(
        position=0.5,
        neck_width=0.12,
        bulb_width=0.32,
        height=0.32,
        neck_ratio=0.35,
        curvature=0.95,
        asymmetry=0.0,  # Keep symmetric for clearer illustration
        corner_slope=0.10,
        squareness=1.3,
        neck_flare=0.2,
        shoulder_offset=0.03,
        shoulder_flatness=0.5,
    )

    curves = generate_realistic_tab_edge((0, 0), (1, 0), params, is_blank=False)

    fig, ax = plt.subplots(figsize=(16, 12))

    # Curve colors - distinct for each of the 6 curves
    curve_colors = [
        "#E74C3C",  # C1: Red - shoulder entry
        "#3498DB",  # C2: Blue - neck left
        "#2ECC71",  # C3a: Green - bulb left
        "#27AE60",  # C3b: Dark green - bulb right (was C4)
        "#F39C12",  # C5: Orange - neck right
        "#9B59B6",  # C6: Purple - shoulder exit
    ]
    curve_names = [
        "C1: Shoulder Entry",
        "C2: Neck Left",
        "C3a: Bulb Left",
        "C3b: Bulb Right",
        "C5: Neck Right",
        "C6: Shoulder Exit",
    ]
    edge_color = "#2C3E50"

    # Draw baseline
    ax.axhline(y=0, color="gray", linewidth=2, linestyle="--", alpha=0.5)
    ax.text(0.02, 0.008, "baseline (edge line)", fontsize=9, color="gray", alpha=0.8)

    # Plot each curve
    for i, curve in enumerate(curves):
        points = curve.get_points(100)
        # Plot the curve itself
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=edge_color,
            linewidth=4,
            zorder=3,
            solid_capstyle="round",
        )

        # Plot control handles (P0→P1 and P2→P3 only, not the full polygon)
        # This matches how vector graphics tools visualize Bézier curves
        ctrl = np.array([curve.p0, curve.p1, curve.p2, curve.p3])
        # Handle at start: P0 → P1
        ax.plot(
            [ctrl[0, 0], ctrl[1, 0]],
            [ctrl[0, 1], ctrl[1, 1]],
            "--",
            color=curve_colors[i],
            alpha=0.7,
            linewidth=1.5,
            zorder=2,
        )
        # Handle at end: P2 → P3
        ax.plot(
            [ctrl[2, 0], ctrl[3, 0]],
            [ctrl[2, 1], ctrl[3, 1]],
            "--",
            color=curve_colors[i],
            alpha=0.7,
            linewidth=1.5,
            zorder=2,
        )

        # Plot control points
        s_anchor = 120  # Size for anchor points (p0, p3)
        s_handle = 70  # Size for handle points (p1, p2)

        # Anchor points (circles) - shared between curves
        ax.scatter(
            [curve.p0[0], curve.p3[0]],
            [curve.p0[1], curve.p3[1]],
            color=curve_colors[i],
            s=s_anchor,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        # Handle points (squares)
        ax.scatter(
            [curve.p1[0], curve.p2[0]],
            [curve.p1[1], curve.p2[1]],
            color=curve_colors[i],
            s=s_handle,
            marker="s",
            edgecolors="black",
            linewidths=1,
            alpha=0.9,
            zorder=4,
        )

        # Label control points
        for j, (x, y) in enumerate(ctrl):
            # Use curve naming that matches our code
            curve_num = ["1", "2", "3a", "3b", "5", "6"][i]
            label = f"C{curve_num}.P{j}"

            # Offset labels to avoid overlap
            offset_x, offset_y = 8, 8
            if j == 2:  # P2 points - offset differently
                offset_x, offset_y = -8, 8
            if i >= 3 and j in [1, 2]:  # Right side curves
                offset_x = -offset_x

            ax.annotate(
                label,
                (x, y),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=edge_color,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
            )

    # === Parameter Annotations ===
    annotation_style = {"fontsize": 10, "fontweight": "bold"}
    arrow_style = {"arrowstyle": "<->", "lw": 1.5}

    # 1. neck_width
    neck_base_y = -params.shoulder_offset
    neck_left_x = 0.5 - params.neck_width / 2
    neck_right_x = 0.5 + params.neck_width / 2
    ax.annotate(
        "",
        xy=(neck_left_x, neck_base_y - 0.02),
        xytext=(neck_right_x, neck_base_y - 0.02),
        arrowprops={**arrow_style, "color": "#16A085"},
    )
    ax.text(0.5, neck_base_y - 0.04, "neck_width", ha="center", color="#16A085", **annotation_style)

    # 2. bulb_width
    bulb_y = params.height * params.neck_ratio + (params.height * (1 - params.neck_ratio)) * 0.5
    bulb_left_x = 0.5 - params.bulb_width / 2
    bulb_right_x = 0.5 + params.bulb_width / 2
    ax.annotate(
        "",
        xy=(bulb_left_x - 0.01, bulb_y),
        xytext=(bulb_right_x + 0.01, bulb_y),
        arrowprops={**arrow_style, "color": "#E74C3C"},
    )
    ax.text(0.5, bulb_y + 0.015, "bulb_width", ha="center", color="#E74C3C", **annotation_style)

    # 3. height
    ax.annotate("", xy=(0.85, 0), xytext=(0.85, params.height), arrowprops={**arrow_style, "color": "#8E44AD"})
    ax.text(0.87, params.height / 2, "height", ha="left", va="center", color="#8E44AD", **annotation_style)

    # 4. neck_ratio (portion of height where neck ends)
    neck_height = params.height * params.neck_ratio
    ax.annotate("", xy=(0.92, 0), xytext=(0.92, neck_height), arrowprops={**arrow_style, "color": "#D35400"})
    ax.text(
        0.94,
        neck_height / 2,
        "neck_ratio\n(of height)",
        ha="left",
        va="center",
        color="#D35400",
        fontsize=9,
        fontweight="bold",
    )

    # 5. shoulder_offset
    ax.annotate("", xy=(0.25, 0), xytext=(0.25, neck_base_y), arrowprops={**arrow_style, "color": "#27AE60"})
    ax.text(
        0.23,
        neck_base_y / 2,
        "shoulder\noffset",
        ha="right",
        va="center",
        color="#27AE60",
        fontsize=9,
        fontweight="bold",
    )

    # 6. shoulder_flatness indicator
    ax.annotate(
        "shoulder_flatness\n(how flat before turn)",
        xy=(0.2, 0.0),
        xytext=(0.02, 0.12),
        fontsize=9,
        fontweight="bold",
        color="#3498DB",
        arrowprops={"arrowstyle": "->", "color": "#3498DB", "lw": 1.2},
    )

    # 8. squareness indicator (at bulb top)
    ax.annotate(
        "squareness\n(flat-top bulb)",
        xy=(0.5, params.height),
        xytext=(0.5, params.height + 0.06),
        fontsize=9,
        fontweight="bold",
        color="#9B59B6",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#9B59B6", "lw": 1.2},
    )

    # 9. curvature indicator
    ax.annotate(
        "curvature\n(bulb roundness)",
        xy=(0.5 - params.bulb_width / 2 - 0.02, params.height * 0.7),
        xytext=(0.08, 0.28),
        fontsize=9,
        fontweight="bold",
        color="#E67E22",
        arrowprops={"arrowstyle": "->", "color": "#E67E22", "lw": 1.2},
    )

    # 10. neck_flare indicator
    ax.annotate(
        "neck_flare\n(waist shape)",
        xy=(neck_left_x + 0.02, neck_height * 0.5),
        xytext=(0.18, 0.18),
        fontsize=9,
        fontweight="bold",
        color="#2980B9",
        arrowprops={"arrowstyle": "->", "color": "#2980B9", "lw": 1.2},
    )

    # Legend for curves
    legend_y = 0.42
    ax.text(0.02, legend_y + 0.03, "Curves:", fontsize=11, fontweight="bold", color=edge_color)
    for i, (name, color) in enumerate(zip(curve_names, curve_colors)):
        ax.plot([0.02, 0.06], [legend_y - i * 0.025, legend_y - i * 0.025], color=color, linewidth=3)
        ax.text(0.07, legend_y - i * 0.025, name, fontsize=9, va="center", color=edge_color)

    # Title
    ax.set_title(
        "Puzzle Piece Tab Generation: 6 Cubic Bézier Curves\n(12 Parameters)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.08, 0.52)
    ax.grid(True, linestyle=":", alpha=0.4)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path = "../outputs/bezier_structure_for_prompt.png"
    plt.savefig(output_path, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"Saved illustration to: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_prompt_illustration()
