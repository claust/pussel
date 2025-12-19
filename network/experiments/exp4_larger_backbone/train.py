"""Training script for larger backbone cell classification experiment.

This experiment tests Option A from exp3: Increase Backbone Capacity Moderately.

Key changes from exp3:
- Larger backbone: 3→32→64→128→256 (256-dim features vs 64-dim)
- MPS (Metal Performance Shaders) support for macOS GPU acceleration
- Same training procedure for fair comparison
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import CellClassificationDataset, verify_cell_indices
from .model import CellClassifierLargerBackbone, count_backbone_parameters, count_parameters
from .visualize import create_grid_comparison, save_accuracy_grid, save_heatmap_overlay


def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU).

    On macOS, MPS (Metal Performance Shaders) provides GPU acceleration.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute top-1 and top-5 accuracy.

    Args:
        logits: Model output logits (batch, num_cells).
        targets: Target cell indices (batch,).

    Returns:
        Dictionary with accuracy metrics.
    """
    batch_size = targets.size(0)

    # Top-1
    preds = torch.argmax(logits, dim=1)
    top1_correct = (preds == targets).sum().item()

    # Top-5
    _, top5_indices = torch.topk(logits, k=min(5, logits.size(1)), dim=1)
    top5_correct = sum(targets[i].item() in top5_indices[i].tolist() for i in range(batch_size))

    return {
        "top1": top1_correct / batch_size,
        "top5": top5_correct / batch_size,
    }


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
    dataset: CellClassificationDataset,
    device: torch.device,
    max_steps: int = 2000,
) -> bool:
    """Test 1: Overfit on a single piece.

    Should reach 100% accuracy quickly on a single sample.

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
    print(f"Target cell: {target.item()} (col={info['col']}, row={info['row']})")

    for step in range(max_steps):
        optimizer.zero_grad()
        logits = model(piece)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=1).item()
        correct = pred == target.item()

        if step % 100 == 0:
            probs = F.softmax(logits, dim=1)
            target_prob = probs[0, int(target.item())].item()
            print(
                f"Step {step:4d}: loss = {loss.item():.6f}, "
                f"pred = {pred}, target_prob = {target_prob:.4f}, "
                f"correct = {correct}"
            )

        if correct and loss.item() < 0.01:
            print(f"\nSUCCESS: Converged at step {step} with loss = {loss.item():.6f}")
            return True

    final_loss = loss.item()  # type: ignore[possibly-undefined]
    pred = torch.argmax(logits, dim=1).item()  # type: ignore[possibly-undefined]
    correct = pred == target.item()
    if correct:
        print(f"\nSUCCESS: Correct prediction, final loss = {final_loss:.6f}")
        return True
    else:
        print(f"\nFAILURE: Did not converge. Final loss = {final_loss:.6f}, pred = {pred}")
        return False


def overfit_ten_pieces(
    model: torch.nn.Module,
    dataset: CellClassificationDataset,
    device: torch.device,
    max_epochs: int = 500,
) -> bool:
    """Test 2: Overfit on 10 pieces.

    Tests whether the model can memorize a small set.

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

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        total = 0

        for pieces, targets in loader:
            pieces = pieces.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(pieces)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == targets).sum().item()
            total += targets.size(0)

        accuracy = epoch_correct / total

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: loss = {epoch_loss:.6f}, accuracy = {accuracy:.1%}")

        if accuracy >= 1.0 and epoch_loss < 0.1:
            print(f"\nSUCCESS: 100% accuracy at epoch {epoch}, loss = {epoch_loss:.6f}")
            return True

    if accuracy >= 0.9:  # type: ignore[possibly-undefined]
        print(f"\nSUCCESS: {accuracy:.1%} accuracy, loss = {epoch_loss:.6f}")  # type: ignore[possibly-undefined]
        return True
    else:
        print(f"\nFAILURE: Only {accuracy:.1%} accuracy")  # type: ignore[possibly-undefined]
        return False


_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def full_training(
    model: torch.nn.Module,
    dataset: CellClassificationDataset,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
) -> dict:
    """Full training on all pieces from the puzzle.

    Args:
        model: Neural network model to train.
        dataset: CellClassificationDataset with all pieces.
        device: Computation device.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        lr: Learning rate.
        output_dir: Directory for saving outputs.

    Returns:
        Training history dict with loss and accuracy values.
    """
    print("\n" + "=" * 60)
    print("FULL TRAINING (overfit all pieces)")
    print("=" * 60)
    print(f"Dataset size: {len(dataset)} pieces")
    print(f"Number of cells: {dataset.num_cells}")

    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Optimize DataLoader for better GPU utilization
    # - num_workers: parallel data loading (0 for MPS compatibility)
    # - pin_memory: faster CPU->GPU transfer
    # - persistent_workers: keep workers alive between epochs
    num_workers = 4 if device.type == "cuda" else 0  # MPS works best with 0 workers
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),  # Only for CUDA, not MPS
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_top5": [],
    }
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_top5_correct = 0
        total = 0
        n_batches = 0

        for pieces, targets in loader:
            pieces = pieces.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(pieces)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

            # Accuracy
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == targets).sum().item()

            # Top-5
            _, top5 = torch.topk(logits, k=5, dim=1)
            for i in range(targets.size(0)):
                if targets[i].item() in top5[i].tolist():
                    epoch_top5_correct += 1
            total += targets.size(0)

            # Gradient check (first few epochs only)
            if step % 200 == 0 and epoch < 3:
                print(f"\n[Step {step}] Gradient norms:")
                print_gradient_norms(model)

        avg_loss = epoch_loss / n_batches
        accuracy = epoch_correct / total
        top5_accuracy = epoch_top5_correct / total

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(accuracy)
        history["train_top5"].append(top5_accuracy)

        # Print epoch summary
        print(
            f"\nEpoch {epoch + 1:3d}/{epochs}: "
            f"loss = {avg_loss:.4f}, "
            f"acc = {accuracy:.1%}, "
            f"top5 = {top5_accuracy:.1%}"
        )

        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            save_visualizations(model, dataset, device, output_dir, epoch + 1)

        # Early stopping
        if accuracy >= 0.99:
            print(f"\nSUCCESS: Reached {accuracy:.1%} accuracy at epoch {epoch + 1}")
            break

    # Final visualizations
    save_visualizations(model, dataset, device, output_dir, epochs, final=True)

    return history


def save_visualizations(
    model: torch.nn.Module,
    dataset: CellClassificationDataset,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    final: bool = False,
) -> None:
    """Save visualization outputs.

    Args:
        model: Trained model.
        dataset: Dataset.
        device: Computation device.
        output_dir: Output directory.
        epoch: Current epoch number.
        final: If True, save as "final" instead of epoch number.
    """
    model.eval()

    # Collect predictions for all pieces
    all_targets = []
    all_preds = []
    all_probs = []
    all_pieces = []

    with torch.no_grad():
        for i in range(len(dataset)):
            piece, target = dataset[i]
            piece_batch = piece.unsqueeze(0).to(device)

            logits = model(piece_batch)
            probs = F.softmax(logits, dim=1)[0]
            pred = torch.argmax(logits, dim=1).item()

            all_targets.append(target.item())
            all_preds.append(pred)
            all_probs.append(probs.cpu())

            if len(all_pieces) < 8:
                all_pieces.append(piece)

    # Calculate accuracy
    correct = sum(t == p for t, p in zip(all_targets, all_preds))
    accuracy = correct / len(all_targets)

    prefix = "final" if final else f"epoch_{epoch:03d}"

    # Save accuracy grid
    puzzle_tensor = dataset.get_puzzle_image()
    save_accuracy_grid(
        puzzle_tensor,
        all_targets,
        all_preds,
        dataset.num_cols,
        dataset.num_rows,
        output_dir / f"{prefix}_accuracy.png",
    )
    print(f"Saved accuracy grid ({accuracy:.1%}) to {output_dir / f'{prefix}_accuracy.png'}")

    # Save heatmap for first sample
    save_heatmap_overlay(
        puzzle_tensor,
        all_probs[0],
        dataset.num_cols,
        dataset.num_rows,
        output_dir / f"{prefix}_heatmap.png",
        target_cell=all_targets[0],
        pred_cell=all_preds[0],
    )
    print(f"Saved heatmap to {output_dir / f'{prefix}_heatmap.png'}")

    # Save grid comparison
    create_grid_comparison(
        all_pieces[:8],
        all_probs[:8],
        all_targets[:8],
        all_preds[:8],
        dataset.num_cols,
        dataset.num_rows,
        output_dir / f"{prefix}_grid.png",
        grid_size=(2, 4),
    )
    print(f"Saved grid comparison to {output_dir / f'{prefix}_grid.png'}")


def main(epochs: int = 200):
    """Run all verification checks.

    Args:
        epochs: Number of training epochs for full training.

    Returns:
        Results dictionary with pass/fail for each test.
    """
    print("=" * 60)
    print("EXPERIMENT 4: LARGER BACKBONE CELL CLASSIFICATION")
    print("=" * 60)
    print("Testing Option A: Increase Backbone Capacity Moderately")
    print("Backbone: 3 -> 32 -> 64 -> 128 -> 256 (256-dim features)")
    print("=" * 60)

    # Setup
    device = get_device()
    print(f"Device: {device}")

    # Create dataset
    dataset = CellClassificationDataset(
        puzzle_id="puzzle_001",
        piece_size=64,
        puzzle_size=256,
        unrotate_pieces=True,
    )

    print(f"Dataset size: {len(dataset)} pieces")
    print(f"Grid: {dataset.num_cols} cols x {dataset.num_rows} rows = {dataset.num_cells} cells")

    # Verify cell indices
    verification = verify_cell_indices(dataset)
    print(f"Cell indices unique: {verification['all_unique']}")
    print(f"All cells covered: {verification['all_covered']}")

    # Create model
    model = CellClassifierLargerBackbone(num_cells=dataset.num_cells).to(device)
    total_params = count_parameters(model)
    backbone_params = count_backbone_parameters(model)
    print("Model: CellClassifierLargerBackbone")
    print(f"Total parameters: {total_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")

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
            batch_size=128,  # Larger batch size for better GPU utilization
            lr=1e-3,
            output_dir=_DEFAULT_OUTPUT_DIR,
        )
        results["final_loss"] = history["train_loss"][-1]
        results["final_accuracy"] = history["train_acc"][-1]
        results["final_top5"] = history["train_top5"][-1]
        results["training_success"] = history["train_acc"][-1] > 0.95
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
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Final accuracy: {results['final_accuracy']:.1%}")
        print(f"Final top-5 accuracy: {results['final_top5']:.1%}")
    print(f"Overall success: {'PASS' if results.get('training_success', False) else 'FAIL'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Larger backbone cell classification experiment")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs for full training",
    )
    args = parser.parse_args()

    main(epochs=args.epochs)
