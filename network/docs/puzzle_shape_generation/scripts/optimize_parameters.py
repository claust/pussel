#!/usr/bin/env python3
"""Parameter Optimizer for Puzzle Piece Generation.

Optimizes the parameters in reference_pieces.json to minimize the difference
between generated pieces and reference images using shape comparison metrics.

Usage:
    # Optimize a single piece (1-6):
    python optimize_parameters.py 1

    # Optimize all pieces:
    python optimize_parameters.py --all

    # Dry run (show results without saving):
    python optimize_parameters.py 1 --dry-run

    # Use different optimization method:
    python optimize_parameters.py 1 --method powell

    # Higher precision (more iterations):
    python optimize_parameters.py 1 --max-iter 500
"""

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from bezier_piece_generator import PieceConfig, load_pieces_from_json
from scipy.optimize import differential_evolution, minimize
from shape_comparator import (
    compute_hausdorff_distance,
    compute_iou,
    compute_mean_contour_distance,
    extract_contour_from_image,
    normalize_contour,
    resample_contour,
)

# Parameter bounds (from README)
PARAM_BOUNDS = {
    "position": (0.35, 0.65),
    "neck_width": (0.06, 0.12),
    "bulb_width": (0.20, 0.32),
    "height": (0.12, 0.30),
    "neck_ratio": (0.15, 0.55),
    "curvature": (0.30, 1.0),
    "asymmetry": (-0.15, 0.15),
}

# Parameters to optimize (exclude position by default for stability)
OPTIMIZABLE_PARAMS = ["neck_width", "bulb_width", "height", "neck_ratio", "curvature", "asymmetry"]


@dataclass
class OptimizationResult:
    """Result of parameter optimization for a single piece."""

    piece_num: int
    initial_iou: float
    final_iou: float
    initial_mean_dist: float
    final_mean_dist: float
    initial_hausdorff: float
    final_hausdorff: float
    optimized_config: dict[str, Any]
    iterations: int
    success: bool


def config_to_vector(config: PieceConfig, param_names: list[str]) -> np.ndarray:
    """Convert piece config to a flat parameter vector for optimization."""
    params = []
    for edge_params in config.edge_params:
        if edge_params is not None:
            for name in param_names:
                params.append(getattr(edge_params, name))
    return np.array(params)


def vector_to_config(
    vector: np.ndarray,
    original_config: PieceConfig,
    param_names: list[str],
) -> PieceConfig:
    """Convert parameter vector back to a piece config."""
    config = copy.deepcopy(original_config)
    idx = 0
    for _, edge_params in enumerate(config.edge_params):
        if edge_params is not None:
            for name in param_names:
                setattr(edge_params, name, float(vector[idx]))
                idx += 1
    return config


def get_bounds_for_config(config: PieceConfig, param_names: list[str]) -> list[tuple[float, float]]:
    """Get optimization bounds for the parameter vector."""
    bounds = []
    for edge_params in config.edge_params:
        if edge_params is not None:
            for name in param_names:
                bounds.append(PARAM_BOUNDS[name])
    return bounds


def compute_objective(
    vector: np.ndarray,
    original_config: PieceConfig,
    param_names: list[str],
    ref_contour_norm: np.ndarray,
    ref_resampled: np.ndarray,
    weights: dict[str, float],
) -> float:
    """Compute the objective function (lower is better).

    Combines IoU (inverted), mean contour distance, and Hausdorff distance.
    """
    try:
        # Convert vector to config
        config = vector_to_config(vector, original_config, param_names)

        # Generate contour from config
        from bezier_piece_generator import generate_piece_path

        x_coords, y_coords = generate_piece_path(config)
        gen_contour = np.array(list(zip(x_coords, y_coords)))

        # Normalize and resample
        gen_norm = normalize_contour(gen_contour)
        gen_resampled = resample_contour(gen_norm, 500)

        # Compute metrics
        iou = compute_iou(ref_contour_norm, gen_norm)
        mean_dist = compute_mean_contour_distance(ref_resampled, gen_resampled)
        hausdorff = compute_hausdorff_distance(ref_resampled, gen_resampled)

        # Combine into objective (lower is better)
        # IoU is inverted (1 - iou) so lower is better
        objective = weights["iou"] * (1 - iou) + weights["mean_dist"] * mean_dist + weights["hausdorff"] * hausdorff

        return float(objective)

    except Exception:
        # Return high penalty for invalid configurations
        return 1000.0


def optimize_piece(
    piece_num: int,
    ref_dir: Path,
    json_path: Path,
    param_names: list[str] = OPTIMIZABLE_PARAMS,
    method: str = "L-BFGS-B",
    max_iter: int = 200,
    weights: dict[str, float] | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Optimize parameters for a single piece.

    Args:
        piece_num: Piece number (1-6).
        ref_dir: Directory containing reference images.
        json_path: Path to the JSON configuration file.
        param_names: Names of parameters to optimize.
        method: Optimization method ('L-BFGS-B', 'Powell', 'Nelder-Mead', 'differential_evolution').
        max_iter: Maximum iterations.
        weights: Weights for different metrics in objective function.
        verbose: Print progress information.

    Returns:
        OptimizationResult with initial and final metrics.
    """
    if weights is None:
        # Default weights emphasizing IoU
        weights = {"iou": 10.0, "mean_dist": 100.0, "hausdorff": 50.0}

    piece_index = piece_num - 1

    # Load current config
    pieces = load_pieces_from_json(json_path)
    original_config = pieces[piece_index]

    # Load reference contour
    ref_image_path = ref_dir / f"piece_{piece_num}.png"
    ref_contour = extract_contour_from_image(ref_image_path)
    ref_contour_norm = normalize_contour(ref_contour)
    ref_resampled = resample_contour(ref_contour_norm, 500)

    # Get initial metrics
    initial_vector = config_to_vector(original_config, param_names)

    def objective(v: np.ndarray) -> float:
        return compute_objective(v, original_config, param_names, ref_contour_norm, ref_resampled, weights)

    initial_obj = objective(initial_vector)

    # Compute initial metrics separately
    from bezier_piece_generator import generate_piece_path

    x, y = generate_piece_path(original_config)
    init_gen = np.array(list(zip(x, y)))
    init_gen_norm = normalize_contour(init_gen)
    init_gen_res = resample_contour(init_gen_norm, 500)
    initial_iou = compute_iou(ref_contour_norm, init_gen_norm)
    initial_mean_dist = compute_mean_contour_distance(ref_resampled, init_gen_res)
    initial_hausdorff = compute_hausdorff_distance(ref_resampled, init_gen_res)

    if verbose:
        print(f"\nPiece {piece_num} - Starting optimization...")
        print(f"  Initial IoU: {initial_iou:.4f}")
        print(f"  Initial Mean Dist: {initial_mean_dist:.4f}")
        print(f"  Initial Hausdorff: {initial_hausdorff:.4f}")
        print(f"  Initial Objective: {initial_obj:.4f}")

    # Get bounds
    bounds = get_bounds_for_config(original_config, param_names)

    # Run optimization
    if method.lower() == "differential_evolution":
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            seed=42,  # type: ignore[call-arg]
            disp=verbose,
            polish=True,
        )
        optimal_vector = result.x
        success = result.success
        iterations = result.nit
    else:
        result = minimize(
            objective,
            initial_vector,
            method=method,
            bounds=bounds,
            options={"maxiter": max_iter, "disp": verbose},
        )
        optimal_vector = result.x
        success = result.success
        iterations = result.nit if hasattr(result, "nit") else 0

    # Compute final metrics
    optimized_config = vector_to_config(optimal_vector, original_config, param_names)
    x, y = generate_piece_path(optimized_config)
    final_gen = np.array(list(zip(x, y)))
    final_gen_norm = normalize_contour(final_gen)
    final_gen_res = resample_contour(final_gen_norm, 500)
    final_iou = compute_iou(ref_contour_norm, final_gen_norm)
    final_mean_dist = compute_mean_contour_distance(ref_resampled, final_gen_res)
    final_hausdorff = compute_hausdorff_distance(ref_resampled, final_gen_res)

    if verbose:
        print(f"\n  Final IoU: {final_iou:.4f} (delta: {final_iou - initial_iou:+.4f})")
        print(f"  Final Mean Dist: {final_mean_dist:.4f} (delta: {final_mean_dist - initial_mean_dist:+.4f})")
        print(f"  Final Hausdorff: {final_hausdorff:.4f} (delta: {final_hausdorff - initial_hausdorff:+.4f})")
        print(f"  Iterations: {iterations}, Success: {success}")

    return OptimizationResult(
        piece_num=piece_num,
        initial_iou=initial_iou,
        final_iou=final_iou,
        initial_mean_dist=initial_mean_dist,
        final_mean_dist=final_mean_dist,
        initial_hausdorff=initial_hausdorff,
        final_hausdorff=final_hausdorff,
        optimized_config=optimized_config.to_dict(),
        iterations=iterations,
        success=success,
    )


def save_optimized_json(
    results: list[OptimizationResult],
    original_json_path: Path,
    output_json_path: Path | None = None,
) -> Path:
    """Save optimized configurations to JSON file.

    Args:
        results: List of optimization results.
        original_json_path: Path to the original JSON file.
        output_json_path: Path to save optimized JSON. If None, overwrites original.

    Returns:
        Path to saved JSON file.
    """
    # Load original JSON
    with open(original_json_path) as f:
        data = json.load(f)

    # Update with optimized configs
    for result in results:
        idx = result.piece_num - 1
        data["pieces"][idx] = result.optimized_config

    # Save
    output_path = output_json_path or original_json_path
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def validate_paths(ref_dir: Path, json_path: Path) -> bool:
    """Validate that required paths exist.

    Returns:
        True if all paths are valid, False otherwise.
    """
    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}")
        print("Run standardize_references.py first to create standardized reference images.")
        return False

    if not json_path.exists():
        print(f"Error: JSON config not found: {json_path}")
        return False

    return True


def get_pieces_to_optimize(args: argparse.Namespace) -> list[int] | None:
    """Determine which pieces to optimize based on arguments.

    Returns:
        List of piece numbers, or None if arguments are invalid.
    """
    if args.all:
        return list(range(1, 7))
    elif args.piece is not None:
        if args.piece < 1 or args.piece > 6:
            print("Error: Piece number must be between 1 and 6")
            return None
        return [args.piece]
    else:
        print("Error: Please specify a piece number (1-6) or use --all")
        return None


def print_results_table(results: list[OptimizationResult]) -> None:
    """Print a formatted table of optimization results."""
    print("\n" + "=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print(f"{'Piece':<6} {'Init IoU':<10} {'Final IoU':<10} {'Delta':<10} {'Init Dist':<12} {'Final Dist':<12}")
    print("-" * 80)

    for r in results:
        iou_delta = r.final_iou - r.initial_iou
        print(
            f"{r.piece_num:<6} {r.initial_iou:<10.4f} {r.final_iou:<10.4f} "
            f"{iou_delta:+<10.4f} {r.initial_mean_dist:<12.4f} {r.final_mean_dist:<12.4f}"
        )

    if len(results) > 1:
        print("-" * 80)
        avg_init_iou = np.mean([r.initial_iou for r in results])
        avg_final_iou = np.mean([r.final_iou for r in results])
        avg_delta = avg_final_iou - avg_init_iou
        avg_init_dist = np.mean([r.initial_mean_dist for r in results])
        avg_final_dist = np.mean([r.final_mean_dist for r in results])
        print(
            f"{'Avg':<6} {avg_init_iou:<10.4f} {avg_final_iou:<10.4f} "
            f"{avg_delta:+<10.4f} {avg_init_dist:<12.4f} {avg_final_dist:<12.4f}"
        )


def main() -> None:
    """Main entry point for parameter optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize puzzle piece parameters to match reference images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_parameters.py 1               # Optimize piece 1
  python optimize_parameters.py --all           # Optimize all pieces
  python optimize_parameters.py 1 --dry-run     # Preview without saving
  python optimize_parameters.py 1 --method powell  # Use Powell method
  python optimize_parameters.py --all --max-iter 500  # More iterations
""",
    )

    parser.add_argument(
        "piece",
        type=int,
        nargs="?",
        default=None,
        help="Piece number to optimize (1-6)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Optimize all 6 pieces",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show results without saving to JSON",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="L-BFGS-B",
        choices=["L-BFGS-B", "Powell", "Nelder-Mead", "differential_evolution"],
        help="Optimization method (default: L-BFGS-B)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum iterations (default: 200)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON path (default: overwrite original)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--include-position",
        action="store_true",
        help="Also optimize the 'position' parameter",
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    ref_dir = script_dir.parent / "reference_images" / "standardized"
    json_path = script_dir.parent / "reference_pieces.json"

    # Validate paths
    if not validate_paths(ref_dir, json_path):
        return

    # Determine which pieces to optimize
    pieces = get_pieces_to_optimize(args)
    if pieces is None:
        return

    # Determine which parameters to optimize
    param_names = list(OPTIMIZABLE_PARAMS)
    if args.include_position:
        param_names.append("position")

    print(f"Optimizing parameters: {', '.join(param_names)}")
    print(f"Method: {args.method}, Max iterations: {args.max_iter}")
    print(f"Pieces: {pieces}")

    # Run optimization
    results = []
    for piece_num in pieces:
        try:
            result = optimize_piece(
                piece_num=piece_num,
                ref_dir=ref_dir,
                json_path=json_path,
                param_names=param_names,
                method=args.method,
                max_iter=args.max_iter,
                verbose=args.verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"\nPiece {piece_num}: Error - {e}")
            import traceback

            traceback.print_exc()

    # Print results table
    if results:
        print_results_table(results)

        # Save if not dry run
        if not args.dry_run:
            output_path = Path(args.output) if args.output else json_path
            saved_path = save_optimized_json(results, json_path, output_path)
            print(f"\nSaved optimized parameters to: {saved_path}")
        else:
            print("\n[Dry run - results not saved]")


if __name__ == "__main__":
    main()
