#!/usr/bin/env python3
"""Visual comparison of reference vs generated puzzle pieces.

Shows overlaid contours to identify where shape differences occur.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bezier_piece_generator import generate_piece_path, load_pieces_from_json
from shape_comparator import extract_contour_from_image, normalize_contour, resample_contour


def visualize_piece_comparison(
    piece_num: int,
    ref_dir: Path,
    json_path: Path,
    output_path: Path | None = None,
) -> None:
    """Generate visual comparison for a single piece."""
    piece_index = piece_num - 1

    # Load reference contour
    ref_image_path = ref_dir / f"piece_{piece_num}.png"
    ref_contour = extract_contour_from_image(ref_image_path)

    # Generate contour from config
    pieces = load_pieces_from_json(json_path)
    config = pieces[piece_index]
    x_coords, y_coords = generate_piece_path(config)
    gen_contour = np.array(list(zip(x_coords, y_coords)))

    # Normalize both
    ref_norm = normalize_contour(ref_contour)
    gen_norm = normalize_contour(gen_contour)

    # Resample for distance calculation
    ref_res = resample_contour(ref_norm, 500)
    gen_res = resample_contour(gen_norm, 500)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Piece {piece_num} - Reference vs Generated Comparison", fontsize=14)

    # Plot 1: Overlaid contours
    ax = axes[0]
    ax.plot(ref_norm[:, 0], ref_norm[:, 1], "b-", linewidth=2, label="Reference", alpha=0.7)
    ax.plot(gen_norm[:, 0], gen_norm[:, 1], "r-", linewidth=2, label="Generated", alpha=0.7)
    ax.set_title("Overlaid Contours")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Distance heatmap on generated contour
    ax = axes[1]
    # Compute distance from each generated point to nearest reference point
    distances = []
    for pt in gen_res:
        dists = np.sqrt(((ref_res - pt) ** 2).sum(axis=1))
        distances.append(dists.min())
    distances = np.array(distances)

    # Color by distance
    scatter = ax.scatter(gen_res[:, 0], gen_res[:, 1], c=distances, cmap="RdYlGn_r", s=10, vmin=0, vmax=0.1)
    ax.plot(ref_norm[:, 0], ref_norm[:, 1], "b-", linewidth=1, alpha=0.3, label="Reference")
    ax.set_title("Distance Heatmap (generated→reference)")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="Distance")

    # Plot 3: Show the original reference image
    ax = axes[2]
    ref_img = cv2.imread(str(ref_image_path), cv2.IMREAD_UNCHANGED)
    if ref_img is not None:
        # Convert BGRA to RGBA
        if ref_img.shape[2] == 4:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2RGBA)
        else:
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ax.imshow(ref_img)
    ax.set_title("Reference Image")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_all_pieces(ref_dir: Path, json_path: Path, output_dir: Path) -> None:
    """Generate comparison for all pieces in a grid."""
    pieces = load_pieces_from_json(json_path)
    num_pieces = len(pieces)

    fig, axes = plt.subplots(2, num_pieces, figsize=(4 * num_pieces, 8))
    fig.suptitle("Reference vs Generated - All Pieces", fontsize=16)

    for i in range(num_pieces):
        piece_num = i + 1

        # Load reference
        ref_image_path = ref_dir / f"piece_{piece_num}.png"
        try:
            ref_contour = extract_contour_from_image(ref_image_path)
        except Exception:
            continue

        # Generate contour
        config = pieces[i]
        x_coords, y_coords = generate_piece_path(config)
        gen_contour = np.array(list(zip(x_coords, y_coords)))

        # Normalize
        ref_norm = normalize_contour(ref_contour)
        gen_norm = normalize_contour(gen_contour)

        # Top row: Reference image
        ax = axes[0, i]
        ref_img = cv2.imread(str(ref_image_path), cv2.IMREAD_UNCHANGED)
        if ref_img is not None:
            if ref_img.shape[2] == 4:
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2RGBA)
            else:
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ax.imshow(ref_img)
        ax.set_title(f"Ref {piece_num}")
        ax.axis("off")

        # Bottom row: Overlaid contours
        ax = axes[1, i]
        ax.fill(ref_norm[:, 0], ref_norm[:, 1], color="blue", alpha=0.3, label="Reference")
        ax.fill(gen_norm[:, 0], gen_norm[:, 1], color="red", alpha=0.3, label="Generated")
        ax.plot(ref_norm[:, 0], ref_norm[:, 1], "b-", linewidth=1)
        ax.plot(gen_norm[:, 0], gen_norm[:, 1], "r-", linewidth=1)
        ax.set_title(f"Overlay {piece_num}")
        ax.set_aspect("equal")
        ax.axis("off")

    # Add legend to first bottom plot
    axes[1, 0].legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    output_path = output_dir / "all_pieces_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visual comparison of reference vs generated pieces")
    parser.add_argument("piece", type=int, nargs="?", help="Piece number (1-6), or omit for all")
    parser.add_argument("-o", "--output", type=str, help="Output path")

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    ref_dir = script_dir.parent / "reference_images" / "standardized"
    json_path = script_dir.parent / "reference_pieces.json"
    output_dir = script_dir.parent / "outputs"

    if args.piece:
        output_path = Path(args.output) if args.output else output_dir / f"comparison_piece_{args.piece}.png"
        visualize_piece_comparison(args.piece, ref_dir, json_path, output_path)
    else:
        visualize_all_pieces(ref_dir, json_path, output_dir)


if __name__ == "__main__":
    main()
