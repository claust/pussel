"""Training script for single puzzle overfit experiment.

Follows the same verification checklist as baseline_sanity:
1. Overfit 1 piece first
2. Overfit 10 pieces
3. Print predictions every epoch
4. Gradient check every N steps
5. Visualize every 10 epochs + final

This tests whether a network can memorize piece->position mappings
for real texture images from a single puzzle.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import SinglePuzzleDataset, compute_dataset_statistics
from .model import count_parameters, get_model
from .visualize import create_grid_visualization, save_prediction_overlay


def print_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Print and return gradient norms for all layers."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            grad_norms[name] = norm
            print(f"  {name}: grad_norm = {norm:.6f}")
        else:
            grad_norms[name] = 0.0
            print(f"  {name}: grad = None")
    return grad_norms


def overfit_single_piece(
    model: torch.nn.Module,
    dataset: SinglePuzzleDataset,
    device: torch.device,
    max_steps: int = 2000,
) -> bool:
    """Test 1: Overfit on a single piece.

    Should reach near-zero loss quickly. If this doesn't work on real
    texture data, the problem is fundamental.

    Args:
        model: Model to train.
        dataset: Dataset providing piece images.
        device: Computation device.
        max_steps: Maximum optimization steps.

    Returns:
        True if successfully overfits, False otherwise.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Overfit single piece")
    print("=" * 60)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Get single piece
    piece, target = dataset[0]
    piece = piece.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    info = dataset.get_sample_info(0)
    print(f"Piece: {info['piece_id']}")
    print(f"Target: ({target[0, 0].item():.4f}, {target[0, 1].item():.4f})")

    loss = torch.tensor(float("inf"))

    for step in range(max_steps):
        optimizer.zero_grad()
        pred = model(piece)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(
                f"Step {step:4d}: loss = {loss.item():.6f}, "
                f"pred = ({pred[0, 0].item():.4f}, {pred[0, 1].item():.4f})"
            )

        if loss.item() < 1e-5:
            print(f"\nSUCCESS: Converged at step {step} with loss = {loss.item():.8f}")
            return True

    final_loss = loss.item()
    if final_loss < 0.001:
        print(f"\nSUCCESS: Final loss = {final_loss:.6f}")
        return True
    else:
        print(f"\nFAILURE: Did not converge. Final loss = {final_loss:.6f}")
        return False


def overfit_ten_pieces(
    model: torch.nn.Module,
    dataset: SinglePuzzleDataset,
    device: torch.device,
    max_epochs: int = 500,
) -> bool:
    """Test 2: Overfit on 10 pieces.

    Tests whether the model can memorize a small set of real pieces.

    Args:
        model: Model to train.
        dataset: Dataset providing piece images.
        device: Computation device.
        max_epochs: Maximum epochs to train.

    Returns:
        True if successfully overfits, False otherwise.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Overfit 10 pieces")
    print("=" * 60)

    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create subset of 10 pieces
    subset = Subset(dataset, list(range(10)))
    loader = DataLoader(subset, batch_size=10, shuffle=False)

    loss = torch.tensor(float("inf"))
    preds = torch.zeros(10, 2)
    targets = torch.zeros(10, 2)

    for epoch in range(max_epochs):
        for pieces, targets in loader:
            pieces = pieces.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(pieces)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: loss = {loss.item():.6f}")

            # Print a few predictions
            with torch.no_grad():
                for i in range(min(3, len(preds))):
                    print(
                        f"  Piece {i}: pred=({preds[i, 0].item():.3f}, {preds[i, 1].item():.3f}) "
                        f"target=({targets[i, 0].item():.3f}, {targets[i, 1].item():.3f})"
                    )

        if loss.item() < 0.001:
            print(f"\nSUCCESS: Converged at epoch {epoch} with loss = {loss.item():.6f}")
            return True

    final_loss = loss.item()
    if final_loss < 0.01:
        print(f"\nSUCCESS: Final loss = {final_loss:.6f}")
        return True
    else:
        print(f"\nFAILURE: Did not converge. Final loss = {final_loss:.6f}")
        return False


_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def full_training(
    model: torch.nn.Module,
    dataset: SinglePuzzleDataset,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> dict:
    """Full training on all pieces from the puzzle.

    This is a pure overfit test - we train on all pieces and evaluate
    on the same pieces. Success means the network can memorize the
    piece->position mapping for real texture data.

    Args:
        model: Neural network model to train.
        dataset: SinglePuzzleDataset with all pieces.
        device: Computation device.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        output_dir: Directory for saving outputs.

    Returns:
        Training history dict with loss values.
    """
    print("\n" + "=" * 60)
    print("FULL TRAINING (overfit all pieces)")
    print("=" * 60)
    print(f"Dataset size: {len(dataset)} pieces")

    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {"train_loss": []}
    step = 0

    # Storage for visualization
    all_preds: list[tuple[float, float]] = []
    all_targets: list[tuple[float, float]] = []
    all_pieces: list[torch.Tensor] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for pieces, targets in loader:
            pieces = pieces.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(pieces)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            # Gradient check every 100 steps (first few epochs only)
            if step % 100 == 0 and epoch < 3:
                print(f"\n[Step {step}] Gradient norms:")
                print_gradient_norms(model)

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        # Evaluate on all data (since this is overfit test)
        model.eval()
        all_preds.clear()
        all_targets.clear()
        all_pieces.clear()

        eval_loss = 0.0
        eval_batches = 0

        with torch.no_grad():
            for pieces, targets in loader:
                pieces = pieces.to(device)
                targets = targets.to(device)

                preds = model(pieces)
                loss = F.mse_loss(preds, targets)
                eval_loss += loss.item()
                eval_batches += 1

                # Store for visualization (first 8 only)
                if len(all_pieces) < 8:
                    for i in range(min(8 - len(all_pieces), len(pieces))):
                        all_pieces.append(pieces[i].cpu())
                        all_preds.append((preds[i, 0].item(), preds[i, 1].item()))
                        all_targets.append((targets[i, 0].item(), targets[i, 1].item()))

        avg_eval_loss = eval_loss / eval_batches

        # Print epoch summary
        print(f"\nEpoch {epoch + 1:3d}/{epochs}: loss = {avg_loss:.6f}, eval_loss = {avg_eval_loss:.6f}")

        # Print sample predictions
        print("Sample predictions:")
        for i in range(min(4, len(all_preds))):
            pred = all_preds[i]
            tgt = all_targets[i]
            error = ((pred[0] - tgt[0]) ** 2 + (pred[1] - tgt[1]) ** 2) ** 0.5
            print(f"  pred=({pred[0]:.3f}, {pred[1]:.3f}) target=({tgt[0]:.3f}, {tgt[1]:.3f}) error={error:.4f}")

        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Grid visualization of pieces
            vis_path = output_dir / f"epoch_{epoch + 1:03d}.png"
            create_grid_visualization(all_pieces, all_preds, all_targets, vis_path)
            print(f"Saved piece grid to {vis_path}")

            # Overlay visualization on puzzle
            overlay_path = output_dir / f"overlay_{epoch + 1:03d}.png"
            puzzle_tensor = dataset.get_puzzle_image()
            save_prediction_overlay(
                puzzle_tensor,
                all_preds[:8],
                all_targets[:8],
                overlay_path,
            )
            print(f"Saved puzzle overlay to {overlay_path}")

        # Early stopping
        if avg_eval_loss < 0.001:
            print(f"\nSUCCESS: Reached target loss at epoch {epoch + 1}")
            break

    # Final visualizations
    vis_path = output_dir / "final_pieces.png"
    create_grid_visualization(all_pieces, all_preds, all_targets, vis_path)
    print(f"Saved final piece grid to {vis_path}")

    overlay_path = output_dir / "final_overlay.png"
    puzzle_tensor = dataset.get_puzzle_image()
    save_prediction_overlay(puzzle_tensor, all_preds[:8], all_targets[:8], overlay_path)
    print(f"Saved final puzzle overlay to {overlay_path}")

    return history


def main(model_type: str = "piece_loc", epochs: int = 200):
    """Run all verification checks.

    Args:
        model_type: Model to use ("piece_loc", "piece_loc_large", "dual_encoder").
        epochs: Number of training epochs for full training.

    Returns:
        Results dictionary with pass/fail for each test.
    """
    print("=" * 60)
    print("SINGLE PUZZLE OVERFIT EXPERIMENT")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = get_model(model_type).to(device)
    print(f"Model: {model_type}")
    print(f"Parameters: {count_parameters(model):,}")

    # Create dataset
    dataset = SinglePuzzleDataset(
        puzzle_id="puzzle_001",
        piece_size=64,
        puzzle_size=256,
        unrotate_pieces=True,
    )

    print(f"Dataset size: {len(dataset)} pieces")

    # Print dataset statistics
    stats = compute_dataset_statistics(dataset)
    print(f"Target cx range: [{stats['cx_min']:.3f}, {stats['cx_max']:.3f}]")
    print(f"Target cy range: [{stats['cy_min']:.3f}, {stats['cy_max']:.3f}]")

    # Run verification checks
    results: dict[str, bool | float] = {}

    # Test 1: Overfit single piece
    results["overfit_1"] = overfit_single_piece(model, dataset, device)

    # Test 2: Overfit 10 pieces
    results["overfit_10"] = overfit_ten_pieces(model, dataset, device)

    # Test 3: Full training (only if basic tests pass)
    if results["overfit_1"] and results["overfit_10"]:
        print("\nBasic tests passed. Starting full training...")
        history = full_training(
            model,
            dataset,
            device,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            output_dir=_DEFAULT_OUTPUT_DIR,
        )
        results["final_loss"] = history["train_loss"][-1]
        results["training_success"] = history["train_loss"][-1] < 0.01
    else:
        print("\nBasic tests FAILED. Skipping full training.")
        print("There is a fundamental problem that needs investigation.")
        results["training_success"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overfit 1 piece: {'PASS' if results['overfit_1'] else 'FAIL'}")
    print(f"Overfit 10 pieces: {'PASS' if results['overfit_10'] else 'FAIL'}")
    if "final_loss" in results:
        print(f"Final training loss: {results['final_loss']:.6f}")
    print(f"Overall success: {'PASS' if results.get('training_success', False) else 'FAIL'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single puzzle overfit experiment")
    parser.add_argument(
        "--model",
        type=str,
        default="piece_loc",
        choices=["piece_loc", "piece_loc_large", "dual_encoder"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs for full training",
    )
    args = parser.parse_args()

    main(model_type=args.model, epochs=args.epochs)
