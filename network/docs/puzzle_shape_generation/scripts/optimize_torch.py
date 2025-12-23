#!/usr/bin/env python3
"""PyTorch-based Parameter Optimizer for Puzzle Piece Generation.

Uses gradient-based optimization with differentiable loss functions
for significantly faster convergence compared to scipy-based optimization.

Supports macOS MPS (Metal) GPU acceleration.

Usage:
    # Optimize a single piece (1-6):
    python optimize_torch.py 1

    # Optimize all pieces:
    python optimize_torch.py --all

    # Dry run (show results without saving):
    python optimize_torch.py 1 --dry-run

    # Custom learning rate and iterations:
    python optimize_torch.py 1 --lr 0.1 --num-iter 200

    # Force CPU (useful for debugging):
    python optimize_torch.py 1 --device cpu
"""

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from geometry_torch import generate_piece_path_torch, params_to_tensor, tensor_to_edge_params_list
from io_utils import load_pieces_from_json
from losses_torch import chamfer_distance, soft_hausdorff
from models import PieceConfig, TabParameters
from shape_comparator import compute_iou, extract_contour_from_image, normalize_contour, resample_contour

# Parameter bounds for per-edge TabParameters (same as scipy version)
PARAM_BOUNDS = {
    "position": (0.35, 0.65),
    "neck_width": (0.12, 0.25),
    "bulb_width": (0.25, 0.45),
    "height": (0.15, 0.40),
    "neck_ratio": (0.15, 0.55),
    "curvature": (0.30, 1.0),
    "asymmetry": (-0.15, 0.15),
    "corner_slope": (0.0, 0.25),
    "squareness": (1.0, 1.5),
    "neck_flare": (-0.5, 0.6),
}

# Parameters to optimize (exclude position by default for stability)
OPTIMIZABLE_PARAMS = [
    "neck_width",
    "bulb_width",
    "height",
    "neck_ratio",
    "curvature",
    "asymmetry",
    "corner_slope",
    "squareness",
    "neck_flare",
]


def get_device(device_name: str = "auto") -> torch.device:
    """Get the appropriate torch device.

    Note: For this workload (many small tensor operations), CPU is typically
    faster than GPU due to dispatch overhead. Use --device mps/cuda explicitly
    if you want to test GPU performance.

    Args:
        device_name: "auto", "mps", "cuda", or "cpu".

    Returns:
        Torch device to use.
    """
    if device_name == "auto":
        # CPU is actually faster for this workload due to small tensor operations
        # GPU dispatch overhead dominates for many small operations
        return torch.device("cpu")
    return torch.device(device_name)


class BoundedParameters(torch.nn.Module):
    """Parameter module with bounded values via sigmoid transformation."""

    def __init__(
        self,
        initial_values: torch.Tensor,
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor,
    ):
        """Initialize bounded parameters.

        Args:
            initial_values: Initial parameter values (within bounds).
            min_bounds: Minimum allowed values.
            max_bounds: Maximum allowed values.
        """
        super().__init__()

        # Transform initial values to unbounded space (inverse sigmoid)
        # normalized = (initial - min) / (max - min)
        # raw = log(normalized / (1 - normalized)) = logit(normalized)
        normalized = (initial_values - min_bounds) / (max_bounds - min_bounds + 1e-8)
        normalized = torch.clamp(normalized, 0.01, 0.99)  # Avoid inf
        raw = torch.log(normalized / (1 - normalized))

        self.raw_params = torch.nn.Parameter(raw)
        self.register_buffer("min_bounds", min_bounds)
        self.register_buffer("max_bounds", max_bounds)

    def forward(self) -> torch.Tensor:
        """Get bounded parameter values."""
        normalized = torch.sigmoid(self.raw_params)
        return self.min_bounds + normalized * (self.max_bounds - self.min_bounds)


@dataclass
class OptimizationResult:
    """Result of parameter optimization for a single piece."""

    piece_num: int
    initial_chamfer: float
    final_chamfer: float
    initial_iou: float
    final_iou: float
    optimized_config: dict[str, Any]
    iterations: int
    time_seconds: float


def get_bounds_tensors(
    config: PieceConfig,
    param_names: List[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Get min/max bounds tensors for all optimizable parameters.

    Returns:
        Tuple of (min_bounds, max_bounds, edge_indices).
    """
    min_vals = []
    max_vals = []
    edge_indices = []

    for i, edge_params in enumerate(config.edge_params):
        if config.edge_types[i] != "flat" and edge_params is not None:
            edge_indices.append(i)
            for name in param_names:
                min_val, max_val = PARAM_BOUNDS[name]
                min_vals.append(min_val)
                max_vals.append(max_val)

    return (
        torch.tensor(min_vals, device=device, dtype=torch.float32),
        torch.tensor(max_vals, device=device, dtype=torch.float32),
        edge_indices,
    )


def normalize_contour_torch(contour: torch.Tensor) -> torch.Tensor:
    """Normalize a contour by centering and scaling.

    Args:
        contour: Contour tensor of shape (N, 2).

    Returns:
        Normalized contour tensor.
    """
    contour_centered = contour - contour.mean(dim=0)
    min_pt = contour_centered.min(dim=0).values
    max_pt = contour_centered.max(dim=0).values
    scale = torch.norm(max_pt - min_pt)
    if scale > 0:
        return contour_centered / scale
    return contour_centered


def run_optimization_loop(
    params_module: BoundedParameters,
    original_config: PieceConfig,
    param_names: List[str],
    edge_indices: List[int],
    ref_tensor: torch.Tensor,
    device: torch.device,
    num_iterations: int,
    lr: float,
    chamfer_weight: float,
    hausdorff_weight: float,
    verbose: bool,
) -> torch.Tensor:
    """Run the gradient descent optimization loop.

    Args:
        params_module: Bounded parameters module.
        original_config: Original piece configuration.
        param_names: Names of parameters being optimized.
        edge_indices: Indices of edges being optimized.
        ref_tensor: Reference contour tensor.
        device: Torch device.
        num_iterations: Maximum iterations.
        lr: Learning rate.
        chamfer_weight: Weight for Chamfer distance.
        hausdorff_weight: Weight for soft-Hausdorff.
        verbose: Print progress.

    Returns:
        Best parameters tensor found during optimization.
    """
    optimizer = torch.optim.Adam(params_module.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=lr * 0.01
    )

    best_loss = float("inf")
    best_params = params_module().detach().clone()
    patience_counter = 0
    early_stop_patience = 30

    for epoch in range(num_iterations):
        optimizer.zero_grad()

        current_params = params_module()
        edge_params_list = tensor_to_edge_params_list(current_params, original_config, param_names, edge_indices)
        contour = generate_piece_path_torch(edge_params_list, original_config, device, points_per_curve=20)
        contour_norm = normalize_contour_torch(contour)

        chamfer = chamfer_distance(contour_norm, ref_tensor)
        hausdorff = soft_hausdorff(contour_norm, ref_tensor)
        loss = chamfer_weight * chamfer + hausdorff_weight * hausdorff

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params_module().detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch + 1}")
            break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {current_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

    return best_params


def optimize_piece_torch(
    piece_num: int,
    ref_dir: Path,
    json_path: Path,
    param_names: List[str] = OPTIMIZABLE_PARAMS,
    num_iterations: int = 100,
    lr: float = 0.05,
    device: torch.device | None = None,
    verbose: bool = False,
    chamfer_weight: float = 1.0,
    hausdorff_weight: float = 0.3,
) -> OptimizationResult:
    """Optimize parameters for a single piece using gradient descent.

    Args:
        piece_num: Piece number (1-6).
        ref_dir: Directory containing reference images.
        json_path: Path to the JSON configuration file.
        param_names: Names of parameters to optimize.
        num_iterations: Maximum gradient descent iterations.
        lr: Learning rate for Adam optimizer.
        device: Torch device (auto-detected if None).
        verbose: Print progress information.
        chamfer_weight: Weight for Chamfer distance in loss.
        hausdorff_weight: Weight for soft-Hausdorff in loss.

    Returns:
        OptimizationResult with metrics and optimized config.
    """
    from geometry import generate_piece_path

    if device is None:
        device = get_device()

    start_time = time.time()

    # Load config and reference
    pieces = load_pieces_from_json(json_path)
    original_config = pieces[piece_num - 1]

    ref_image_path = ref_dir / f"piece_{piece_num}.png"
    ref_contour = extract_contour_from_image(ref_image_path)
    ref_contour_norm = normalize_contour(ref_contour)
    ref_resampled = resample_contour(ref_contour_norm, 500)
    ref_tensor = torch.from_numpy(ref_resampled).float().to(device)

    # Set up parameters
    initial_vector, edge_indices = params_to_tensor(original_config, param_names, device)
    min_bounds, max_bounds, _ = get_bounds_tensors(original_config, param_names, device)
    params_module = BoundedParameters(initial_vector, min_bounds, max_bounds).to(device)

    # Compute initial metrics
    with torch.no_grad():
        edge_params_list = tensor_to_edge_params_list(params_module(), original_config, param_names, edge_indices)
        initial_contour = generate_piece_path_torch(edge_params_list, original_config, device, points_per_curve=20)
        initial_chamfer = chamfer_distance(normalize_contour_torch(initial_contour), ref_tensor).item()

    x, y = generate_piece_path(original_config)
    init_gen_norm = normalize_contour(np.array(list(zip(x, y))))
    initial_iou = compute_iou(ref_contour_norm, init_gen_norm)

    if verbose:
        print(f"\nPiece {piece_num} - Starting PyTorch optimization...")
        print(f"  Device: {device}")
        print(f"  Initial Chamfer: {initial_chamfer:.6f}")
        print(f"  Initial IoU: {initial_iou:.4f}")

    # Run optimization
    best_params = run_optimization_loop(
        params_module,
        original_config,
        param_names,
        edge_indices,
        ref_tensor,
        device,
        num_iterations,
        lr,
        chamfer_weight,
        hausdorff_weight,
        verbose,
    )

    # Compute final metrics
    with torch.no_grad():
        edge_params_list = tensor_to_edge_params_list(best_params, original_config, param_names, edge_indices)
        final_contour = generate_piece_path_torch(edge_params_list, original_config, device, points_per_curve=20)
        final_chamfer = chamfer_distance(normalize_contour_torch(final_contour), ref_tensor).item()

    # Build optimized config
    optimized_config = copy.deepcopy(original_config)
    best_params_np = best_params.cpu().numpy()
    num_params = len(param_names)

    for idx, edge_idx in enumerate(edge_indices):
        edge_params = optimized_config.edge_params[edge_idx]
        if edge_params is None:
            edge_params = TabParameters()
            optimized_config.edge_params[edge_idx] = edge_params
        for j, name in enumerate(param_names):
            setattr(edge_params, name, float(best_params_np[idx * num_params + j]))

    # Compute final IoU
    x, y = generate_piece_path(optimized_config)
    final_gen_norm = normalize_contour(np.array(list(zip(x, y))))
    final_iou = compute_iou(ref_contour_norm, final_gen_norm)

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n  Final Chamfer: {final_chamfer:.6f} (delta: {final_chamfer - initial_chamfer:+.6f})")
        print(f"  Final IoU: {final_iou:.4f} (delta: {final_iou - initial_iou:+.4f})")
        print(f"  Time: {elapsed:.2f}s")

    return OptimizationResult(
        piece_num=piece_num,
        initial_chamfer=initial_chamfer,
        final_chamfer=final_chamfer,
        initial_iou=initial_iou,
        final_iou=final_iou,
        optimized_config=optimized_config.to_dict(),
        iterations=num_iterations,
        time_seconds=elapsed,
    )


def save_optimized_json(
    results: List[OptimizationResult],
    original_json_path: Path,
    output_json_path: Path | None = None,
) -> Path:
    """Save optimized configurations to JSON file."""
    with open(original_json_path) as f:
        data = json.load(f)

    for result in results:
        idx = result.piece_num - 1
        data["pieces"][idx] = result.optimized_config

    output_path = output_json_path or original_json_path
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    return output_path


def validate_paths(ref_dir: Path, json_path: Path) -> bool:
    """Validate that required paths exist."""
    if not ref_dir.exists():
        print(f"Error: Reference directory not found: {ref_dir}")
        print("Run standardize_references.py first to create standardized reference images.")
        return False

    if not json_path.exists():
        print(f"Error: JSON config not found: {json_path}")
        return False

    return True


def get_pieces_to_optimize(args: argparse.Namespace) -> List[int] | None:
    """Determine which pieces to optimize based on arguments."""
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


def print_results_table(results: List[OptimizationResult]) -> None:
    """Print a formatted table of optimization results."""
    print("\n" + "=" * 90)
    print("PyTorch Optimization Results")
    print("=" * 90)
    print(
        f"{'Piece':<6} {'Init IoU':<10} {'Final IoU':<10} {'Delta':<10} "
        f"{'Init Chamfer':<14} {'Final Chamfer':<14} {'Time':<8}"
    )
    print("-" * 90)

    for r in results:
        iou_delta = r.final_iou - r.initial_iou
        print(
            f"{r.piece_num:<6} {r.initial_iou:<10.4f} {r.final_iou:<10.4f} "
            f"{iou_delta:+<10.4f} {r.initial_chamfer:<14.6f} {r.final_chamfer:<14.6f} "
            f"{r.time_seconds:<8.2f}s"
        )

    if len(results) > 1:
        print("-" * 90)
        avg_init_iou = np.mean([r.initial_iou for r in results])
        avg_final_iou = np.mean([r.final_iou for r in results])
        avg_delta = avg_final_iou - avg_init_iou
        avg_init_chamfer = np.mean([r.initial_chamfer for r in results])
        avg_final_chamfer = np.mean([r.final_chamfer for r in results])
        total_time = sum(r.time_seconds for r in results)
        print(
            f"{'Avg':<6} {avg_init_iou:<10.4f} {avg_final_iou:<10.4f} "
            f"{avg_delta:+<10.4f} {avg_init_chamfer:<14.6f} {avg_final_chamfer:<14.6f} "
            f"{total_time:<8.2f}s"
        )


def main() -> None:
    """Main entry point for PyTorch parameter optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize puzzle piece parameters using PyTorch gradient descent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_torch.py 1               # Optimize piece 1
  python optimize_torch.py --all           # Optimize all pieces
  python optimize_torch.py 1 --dry-run     # Preview without saving
  python optimize_torch.py 1 --lr 0.1      # Higher learning rate
  python optimize_torch.py --all --device cpu  # Force CPU
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
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for Adam optimizer (default: 0.05)",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=100,
        help="Maximum iterations (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use (default: auto)",
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

    # Get device
    device = get_device(args.device)

    print("PyTorch Optimizer")
    print(f"  Device: {device}")
    print(f"  Parameters: {', '.join(param_names)}")
    print(f"  Learning rate: {args.lr}, Max iterations: {args.num_iter}")
    print(f"  Pieces: {pieces}")

    # Run optimization
    results = []
    total_start = time.time()

    for piece_num in pieces:
        try:
            result = optimize_piece_torch(
                piece_num=piece_num,
                ref_dir=ref_dir,
                json_path=json_path,
                param_names=param_names,
                num_iterations=args.num_iter,
                lr=args.lr,
                device=device,
                verbose=args.verbose,
            )
            results.append(result)
        except Exception as e:
            print(f"\nPiece {piece_num}: Error - {e}")
            import traceback

            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # Print results table
    if results:
        print_results_table(results)
        print(f"\nTotal time: {total_elapsed:.2f}s")

        # Save if not dry run
        if not args.dry_run:
            output_path = Path(args.output) if args.output else json_path
            saved_path = save_optimized_json(results, json_path, output_path)
            print(f"\nSaved optimized parameters to: {saved_path}")
        else:
            print("\n[Dry run - results not saved]")


if __name__ == "__main__":
    main()
