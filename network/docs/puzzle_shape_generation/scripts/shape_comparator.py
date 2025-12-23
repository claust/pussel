#!/usr/bin/env python3
"""Shape Comparator for Puzzle Pieces.

Compares generated puzzle pieces against reference images using contour-based metrics.

Usage:
    # Compare a single piece (1-6):
    python shape_comparator.py 1

    # Compare all pieces:
    python shape_comparator.py --all

    # Verbose output with debug visualization:
    python shape_comparator.py 1 --verbose
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from geometry import generate_piece_path
from io_utils import load_pieces_from_json
from scipy.spatial.distance import directed_hausdorff


def extract_contour_from_image(image_path: Path) -> np.ndarray:
    """Extract the contour from a reference piece image.

    Args:
        image_path: Path to the PNG image with transparent background.

    Returns:
        Nx2 array of contour points.
    """
    # Read image with alpha channel
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Handle different image formats
    if img.shape[2] == 4:
        # Use alpha channel to create mask
        alpha = img[:, :, 3]
        mask = (alpha > 128).astype(np.uint8) * 255
    else:
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray > 0).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError(f"No contours found in image: {image_path}")

    # Get the largest contour (the piece)
    largest = max(contours, key=cv2.contourArea)

    # Reshape to Nx2
    contour = largest.reshape(-1, 2).astype(np.float64)

    # Flip Y coordinates to convert from image coordinates (Y down) to
    # mathematical coordinates (Y up) to match generated contours
    contour[:, 1] = img.shape[0] - contour[:, 1]

    return contour


def generate_contour_from_config(piece_index: int, json_path: Path) -> np.ndarray:
    """Generate contour points from a piece configuration.

    Args:
        piece_index: 0-based index of the piece in the JSON file.
        json_path: Path to the JSON configuration file.

    Returns:
        Nx2 array of contour points.
    """
    pieces = load_pieces_from_json(json_path)

    if piece_index >= len(pieces):
        raise ValueError(f"Piece index {piece_index} out of range (max {len(pieces) - 1})")

    config = pieces[piece_index]
    x_coords, y_coords = generate_piece_path(config)

    return np.array(list(zip(x_coords, y_coords)))


def normalize_contour(contour: np.ndarray) -> np.ndarray:
    """Normalize contour to unit scale centered at origin.

    Args:
        contour: Nx2 array of contour points.

    Returns:
        Normalized Nx2 array.
    """
    # Center at origin
    centroid = contour.mean(axis=0)
    centered = contour - centroid

    # Scale to unit size (using bounding box diagonal)
    min_pt = centered.min(axis=0)
    max_pt = centered.max(axis=0)
    scale = np.linalg.norm(max_pt - min_pt)

    if scale > 0:
        centered = centered / scale

    return centered


def resample_contour(contour: np.ndarray, num_points: int = 500) -> np.ndarray:
    """Resample contour to a fixed number of equally-spaced points.

    Args:
        contour: Nx2 array of contour points.
        num_points: Number of points to resample to.

    Returns:
        Resampled Nx2 array.
    """
    # Compute cumulative arc length
    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]

    if total_length == 0:
        return contour[:num_points]

    # Generate equally spaced parameter values
    target_lengths = np.linspace(0, total_length, num_points)

    # Interpolate x and y separately
    resampled_x = np.interp(target_lengths, cumulative_length, contour[:, 0])
    resampled_y = np.interp(target_lengths, cumulative_length, contour[:, 1])

    return np.column_stack([resampled_x, resampled_y])


def compute_iou(contour1: np.ndarray, contour2: np.ndarray, resolution: int = 256) -> float:
    """Compute Intersection over Union between two contours.

    Assumes both contours are already normalized to the same scale.

    Args:
        contour1: First contour (Nx2), normalized.
        contour2: Second contour (Mx2), normalized.
        resolution: Resolution of the rasterization grid.

    Returns:
        IoU score between 0 and 1.
    """
    # Create masks for both contours
    mask1 = np.zeros((resolution, resolution), dtype=np.uint8)
    mask2 = np.zeros((resolution, resolution), dtype=np.uint8)

    # Find common bounding box for both contours
    all_points = np.vstack([contour1, contour2])
    min_pt = all_points.min(axis=0)
    max_pt = all_points.max(axis=0)

    # Scale to fit in the resolution grid with margin
    margin = 0.1
    usable_size = resolution * (1 - 2 * margin)
    size = (max_pt - min_pt).max()
    if size == 0:
        size = 1

    def scale_contour(contour: np.ndarray) -> np.ndarray:
        scaled = (contour - min_pt) / size * usable_size + resolution * margin
        return scaled.astype(np.int32)

    pts1 = scale_contour(contour1)
    pts2 = scale_contour(contour2)

    cv2.fillPoly(mask1, [pts1], (255,))
    cv2.fillPoly(mask2, [pts2], (255,))

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def _one_way_contour_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Compute one-way distance from points on c1 to nearest points on c2."""
    distances = []
    for pt in c1:
        dists = np.sqrt(((c2 - pt) ** 2).sum(axis=1))
        distances.append(dists.min())
    return float(np.mean(distances))


def compute_mean_contour_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """Compute mean distance from points on contour1 to nearest points on contour2.

    Args:
        contour1: First contour (Nx2).
        contour2: Second contour (Mx2).

    Returns:
        Mean distance (symmetric average of both directions).
    """
    d1 = _one_way_contour_distance(contour1, contour2)
    d2 = _one_way_contour_distance(contour2, contour1)

    return (d1 + d2) / 2


def compute_hausdorff_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    """Compute Hausdorff distance between two contours.

    Args:
        contour1: First contour (Nx2).
        contour2: Second contour (Mx2).

    Returns:
        Hausdorff distance (maximum of directed distances).
    """
    d1 = directed_hausdorff(contour1, contour2)[0]
    d2 = directed_hausdorff(contour2, contour1)[0]
    return float(max(d1, d2))


def compare_piece(
    piece_num: int,
    ref_dir: Path,
    json_path: Path,
    verbose: bool = False,
) -> dict:
    """Compare a single generated piece against its reference.

    Args:
        piece_num: Piece number (1-6).
        ref_dir: Directory containing reference images.
        json_path: Path to the JSON configuration file.
        verbose: Whether to print detailed output.

    Returns:
        Dictionary with comparison metrics.
    """
    piece_index = piece_num - 1  # Convert to 0-based

    # Load reference contour
    ref_image_path = ref_dir / f"piece_{piece_num}.png"
    ref_contour = extract_contour_from_image(ref_image_path)

    # Generate contour from config
    gen_contour = generate_contour_from_config(piece_index, json_path)

    # Normalize both contours
    ref_norm = normalize_contour(ref_contour)
    gen_norm = normalize_contour(gen_contour)

    # Resample to same number of points
    num_points = 500
    ref_resampled = resample_contour(ref_norm, num_points)
    gen_resampled = resample_contour(gen_norm, num_points)

    # Compute metrics
    iou = compute_iou(ref_norm, gen_norm)
    mean_dist = compute_mean_contour_distance(ref_resampled, gen_resampled)
    hausdorff = compute_hausdorff_distance(ref_resampled, gen_resampled)

    results = {
        "piece_num": piece_num,
        "iou": iou,
        "mean_contour_distance": mean_dist,
        "hausdorff_distance": hausdorff,
    }

    if verbose:
        print(f"\nPiece {piece_num} Details:")
        print(f"  Reference contour points: {len(ref_contour)}")
        print(f"  Generated contour points: {len(gen_contour)}")
        print(f"  Normalized & resampled to: {num_points} points")

    return results


def print_results(results: dict) -> None:
    """Print comparison results in a formatted way."""
    print(f"\nPiece {results['piece_num']}:")
    print(f"  IoU:                    {results['iou']:.3f}  (1.0 = perfect overlap)")
    print(f"  Mean Contour Distance:  {results['mean_contour_distance']:.4f}  (0 = perfect match)")
    print(f"  Hausdorff Distance:     {results['hausdorff_distance']:.4f}  (0 = perfect match)")


def main() -> None:
    """Main entry point for shape comparison."""
    parser = argparse.ArgumentParser(
        description="Compare generated puzzle pieces against reference images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shape_comparator.py 1           # Compare piece 1
  python shape_comparator.py --all       # Compare all pieces
  python shape_comparator.py 3 --verbose # Verbose output for piece 3
""",
    )

    parser.add_argument(
        "piece",
        type=int,
        nargs="?",
        default=None,
        help="Piece number to compare (1-6)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all 6 pieces",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    ref_dir = script_dir.parent / "reference_images" / "standardized"
    json_path = script_dir.parent / "reference_pieces.json"

    # Validate paths
    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}")
        return

    if not json_path.exists():
        print(f"Error: JSON config not found: {json_path}")
        return

    # Determine which pieces to compare
    if args.all:
        pieces = list(range(1, 7))
    elif args.piece is not None:
        if args.piece < 1 or args.piece > 6:
            print("Error: Piece number must be between 1 and 6")
            return
        pieces = [args.piece]
    else:
        print("Error: Please specify a piece number (1-6) or use --all")
        return

    # Run comparisons
    print("=" * 50)
    print("Shape Comparison: Generated vs Reference")
    print("=" * 50)

    all_results = []
    for piece_num in pieces:
        try:
            results = compare_piece(piece_num, ref_dir, json_path, args.verbose)
            print_results(results)
            all_results.append(results)
        except Exception as e:
            print(f"\nPiece {piece_num}: Error - {e}")

    # Print summary if comparing multiple pieces
    if len(all_results) > 1:
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)

        avg_iou = np.mean([r["iou"] for r in all_results])
        avg_mean_dist = np.mean([r["mean_contour_distance"] for r in all_results])
        avg_hausdorff = np.mean([r["hausdorff_distance"] for r in all_results])

        print(f"  Average IoU:                    {avg_iou:.3f}")
        print(f"  Average Mean Contour Distance:  {avg_mean_dist:.4f}")
        print(f"  Average Hausdorff Distance:     {avg_hausdorff:.4f}")

        # Find best and worst pieces by IoU
        best = max(all_results, key=lambda r: r["iou"])
        worst = min(all_results, key=lambda r: r["iou"])
        print(f"\n  Best match:  Piece {best['piece_num']} (IoU: {best['iou']:.3f})")
        print(f"  Worst match: Piece {worst['piece_num']} (IoU: {worst['iou']:.3f})")


if __name__ == "__main__":
    main()
