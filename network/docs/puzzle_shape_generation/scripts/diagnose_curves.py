#!/usr/bin/env python3
"""Diagnostic tool to visualize Bézier curve control points and structure."""

import matplotlib.pyplot as plt
import numpy as np
from bezier_piece_generator import TabParameters, generate_realistic_tab_edge
from matplotlib.figure import Figure


def visualize_single_tab(params: TabParameters, title: str = "Tab Structure") -> Figure:
    """Visualize a single tab with control points."""
    curves = generate_realistic_tab_edge((0, 0), (1, 0), params, is_blank=False)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ["red", "blue", "green", "purple", "orange"]
    labels = ["C1: Start→Neck", "C2: Neck→Bulb", "C3: Bulb top", "C4: Bulb→Neck", "C5: Neck→End"]

    # Plot each curve
    for i, curve in enumerate(curves):
        points = curve.get_points(50)
        ax.plot(points[:, 0], points[:, 1], color=colors[i], linewidth=3, label=labels[i])

        # Plot control polygon
        ctrl = np.array([curve.p0, curve.p1, curve.p2, curve.p3])
        ax.plot(ctrl[:, 0], ctrl[:, 1], "--", color=colors[i], alpha=0.4, linewidth=1)
        ax.scatter(ctrl[:, 0], ctrl[:, 1], color=colors[i], s=50, zorder=5)

        # Label control points
        for j, (x, y) in enumerate(ctrl):
            ax.annotate(f"C{i + 1}.p{j}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Add key reference points
    center_x = params.position
    neck_half = params.neck_width * 0.5
    bulb_half = params.bulb_width * 0.5
    neck_height = params.height * params.neck_ratio
    full_height = params.height

    # Vertical reference lines
    ax.axvline(x=center_x, color="gray", linestyle=":", alpha=0.5, label="Center")
    ax.axvline(x=center_x - neck_half, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=center_x + neck_half, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=center_x - bulb_half, color="gray", linestyle="-.", alpha=0.3)
    ax.axvline(x=center_x + bulb_half, color="gray", linestyle="-.", alpha=0.3)

    # Horizontal reference lines
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axhline(y=neck_height, color="gray", linestyle="--", alpha=0.3, label=f"Neck height ({neck_height:.3f})")
    ax.axhline(y=full_height, color="gray", linestyle=":", alpha=0.3, label=f"Full height ({full_height:.3f})")

    ax.set_title(
        f"{title}\nneck_width={params.neck_width:.3f}, bulb_width={params.bulb_width:.3f}, "
        f"ratio={params.neck_width / params.bulb_width:.1%}"
    )
    ax.set_xlabel("X (along edge)")
    ax.set_ylabel("Y (perpendicular to edge)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_neck_ratios() -> None:
    """Compare different neck/bulb ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Effect of neck/bulb ratio on tab shape", fontsize=14)

    # Test different ratios
    ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    bulb_width = 0.25

    for idx, ratio in enumerate(ratios):
        ax = axes[idx // 3, idx % 3]
        neck_width = bulb_width * ratio

        params = TabParameters(
            position=0.5,
            neck_width=neck_width,
            bulb_width=bulb_width,
            height=0.25,
            neck_ratio=0.35,
            curvature=0.8,
            asymmetry=0.0,
        )

        curves = generate_realistic_tab_edge((0, 0), (1, 0), params, is_blank=False)

        colors = ["red", "blue", "green", "purple", "orange"]
        for i, curve in enumerate(curves):
            points = curve.get_points(50)
            ax.plot(points[:, 0], points[:, 1], color=colors[i], linewidth=2)

        ax.set_title(f"Ratio: {ratio:.0%} (neck={neck_width:.3f})")
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.35)
        ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    plt.savefig("../outputs/neck_ratio_comparison.png", dpi=150)
    print("Saved: ../outputs/neck_ratio_comparison.png")


def main() -> None:
    """Run diagnostics."""
    # Example with thin neck (typical of reference pieces)
    params_thin = TabParameters(
        position=0.5,
        neck_width=0.07,  # ~25% of bulb
        bulb_width=0.28,
        height=0.28,
        neck_ratio=0.35,
        curvature=0.85,
        asymmetry=0.0,
    )

    fig = visualize_single_tab(params_thin, "Thin Neck Tab (25% ratio)")
    fig.savefig("../outputs/tab_structure_thin.png", dpi=150)
    print("Saved: ../outputs/tab_structure_thin.png")
    plt.close()

    # Compare different ratios
    compare_neck_ratios()


if __name__ == "__main__":
    main()
