"""Learning rate finder for exp18.

Sweeps through learning rates exponentially and plots loss vs LR
to help identify the optimal learning rate range.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import create_datasets
from .model import FastBackboneModel


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_finder(
    model: FastBackboneModel,
    train_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
    smooth_factor: float = 0.05,
) -> tuple[list[float], list[float], list[float]]:
    """Run learning rate finder.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        device: Device to use.
        start_lr: Starting learning rate.
        end_lr: Ending learning rate.
        num_iter: Number of iterations.
        smooth_factor: Smoothing factor for loss.

    Returns:
        Tuple of (learning_rates, raw_losses, smoothed_losses).
    """
    # Save initial model state
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Setup optimizer with start_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)

    # Calculate multiplicative factor for LR increase
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    learning_rates: list[float] = []
    raw_losses: list[float] = []
    smoothed_losses: list[float] = []

    model.train()
    data_iter = iter(train_loader)
    avg_loss = 0.0
    best_loss = float("inf")

    print(f"Running LR finder: {start_lr:.2e} -> {end_lr:.2e} over {num_iter} steps")
    print("-" * 60)

    for iteration in range(num_iter):
        # Get batch (cycle through data if needed)
        try:
            pieces, puzzles, targets, _, rotations = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            pieces, puzzles, targets, _, rotations = next(data_iter)

        pieces = pieces.to(device)
        puzzles = puzzles.to(device)
        targets = targets.to(device)
        rotations = rotations.to(device)

        # Forward pass
        optimizer.zero_grad()
        preds, rotation_logits, _ = model(pieces, puzzles)

        # Combined loss
        pos_loss = F.mse_loss(preds, targets)
        rot_loss = F.cross_entropy(rotation_logits, rotations)
        loss = pos_loss + rot_loss

        # Check for divergence
        if math.isnan(loss.item()) or loss.item() > 100:
            print(f"Stopping early at iter {iteration}: loss diverged")
            break

        # Backward pass
        loss.backward()
        optimizer.step()

        # Record
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)
        raw_losses.append(loss.item())

        # Smoothed loss (exponential moving average)
        if iteration == 0:
            avg_loss = loss.item()
        else:
            avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss
        smoothed_losses.append(avg_loss)

        # Track best loss for early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss

        # Stop if loss is diverging (4x best loss)
        if avg_loss > 4 * best_loss and iteration > 10:
            print(f"Stopping at iter {iteration}: loss diverging")
            break

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_mult

        # Progress
        if (iteration + 1) % 20 == 0:
            print(
                f"Iter {iteration + 1:3d}/{num_iter}: "
                f"lr={current_lr:.2e}, loss={loss.item():.4f}, "
                f"smoothed={avg_loss:.4f}"
            )

    # Restore initial model state
    model.load_state_dict(initial_state)

    return learning_rates, raw_losses, smoothed_losses


def plot_lr_finder(
    learning_rates: list[float],
    raw_losses: list[float],
    smoothed_losses: list[float],
    output_path: Path,
) -> None:
    """Plot learning rate finder results.

    Args:
        learning_rates: List of learning rates.
        raw_losses: List of raw losses.
        smoothed_losses: List of smoothed losses.
        output_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Full range (log scale)
    ax1 = axes[0]
    ax1.plot(learning_rates, raw_losses, "b-", alpha=0.3, label="Raw loss")
    ax1.plot(learning_rates, smoothed_losses, "r-", linewidth=2, label="Smoothed loss")
    ax1.set_xscale("log")
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Loss")
    ax1.set_title("LR Finder - Full Range")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Find suggested LR (point of steepest descent)
    if len(smoothed_losses) > 10:
        # Calculate gradient of smoothed loss
        gradients = []
        for i in range(1, len(smoothed_losses)):
            grad = (smoothed_losses[i] - smoothed_losses[i - 1]) / (
                math.log10(learning_rates[i]) - math.log10(learning_rates[i - 1])
            )
            gradients.append(grad)

        # Find minimum gradient (steepest descent)
        min_grad_idx = gradients.index(min(gradients))
        suggested_lr = learning_rates[min_grad_idx]

        # Also find the point where loss starts increasing significantly
        min_loss_idx = smoothed_losses.index(min(smoothed_losses))
        min_loss_lr = learning_rates[min_loss_idx]

        ax1.axvline(
            x=suggested_lr,
            color="green",
            linestyle="--",
            label=f"Steepest: {suggested_lr:.2e}",
        )
        ax1.axvline(
            x=min_loss_lr,
            color="orange",
            linestyle="--",
            label=f"Min loss: {min_loss_lr:.2e}",
        )
        ax1.legend()
    else:
        suggested_lr = None
        min_loss_lr = None

    # Plot 2: Zoomed view with annotations
    ax2 = axes[1]
    ax2.plot(learning_rates, smoothed_losses, "r-", linewidth=2, label="Smoothed loss")
    ax2.set_xscale("log")
    ax2.set_xlabel("Learning Rate")
    ax2.set_ylabel("Loss (smoothed)")
    ax2.set_title("LR Finder - Suggested Range")
    ax2.grid(True, alpha=0.3)

    # Add suggested range
    if suggested_lr and min_loss_lr:
        # Suggested range: 1/10 of steepest descent point to min loss point
        range_low = suggested_lr / 10
        range_high = min_loss_lr

        ax2.axvspan(range_low, range_high, alpha=0.2, color="green", label="Suggested range")
        ax2.axvline(x=suggested_lr, color="green", linestyle="--")
        ax2.axvline(x=min_loss_lr, color="orange", linestyle="--")

        # Add text annotations
        ax2.text(
            0.05,
            0.95,
            f"Suggested LR range:\n{range_low:.2e} - {range_high:.2e}\n\n" f"Recommended: {suggested_lr:.2e}",
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    ax2.legend()

    plt.suptitle("Learning Rate Finder - Exp18 (3x3 Grid, 20K Puzzles)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved LR finder plot to {output_path}")

    if suggested_lr:
        print(f"\n{'=' * 60}")
        print("LEARNING RATE RECOMMENDATIONS")
        print("=" * 60)
        print(f"Steepest descent point: {suggested_lr:.2e}")
        print(f"Minimum loss point: {min_loss_lr:.2e}")
        print(f"Suggested range: {suggested_lr / 10:.2e} - {min_loss_lr:.2e}")
        print(f"\nRecommended starting LR: {suggested_lr:.2e}")
        print(f"(Use 1/10 of this for backbone: {suggested_lr / 10:.2e})")


def main(
    n_train: int = 1000,
    batch_size: int = 64,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
) -> None:
    """Run learning rate finder.

    Args:
        n_train: Number of training puzzles.
        batch_size: Batch size.
        start_lr: Starting learning rate.
        end_lr: Ending learning rate.
        num_iter: Number of iterations.
    """
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER - Exp18 (3x3 Grid, 20K Puzzles)")
    print("=" * 60)
    print(f"Start LR: {start_lr:.2e}")
    print(f"End LR: {end_lr:.2e}")
    print(f"Iterations: {num_iter}")
    print(f"Training puzzles: {n_train}")
    print(f"Batch size: {batch_size}")

    device = get_device()
    print(f"\nDevice: {device}")

    # Create dataset
    print("\nLoading dataset...")
    train_dataset, _ = create_datasets(
        n_train_puzzles=n_train,
        n_test_puzzles=50,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    print("Creating model...")
    model = FastBackboneModel(
        backbone_name="shufflenet_v2_x0_5",
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    # Run LR finder
    print("\n")
    learning_rates, raw_losses, smoothed_losses = lr_finder(
        model=model,
        train_loader=train_loader,
        device=device,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
    )

    # Plot results
    plot_lr_finder(
        learning_rates=learning_rates,
        raw_losses=raw_losses,
        smoothed_losses=smoothed_losses,
        output_path=output_dir / "lr_finder.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning rate finder for exp18")
    parser.add_argument("--n-train", type=int, default=1000, help="Training puzzles")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--start-lr", type=float, default=1e-7, help="Starting LR")
    parser.add_argument("--end-lr", type=float, default=1.0, help="Ending LR")
    parser.add_argument("--num-iter", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()

    main(
        n_train=args.n_train,
        batch_size=args.batch_size,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
    )
