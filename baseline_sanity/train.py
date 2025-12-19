"""
Training script for baseline sanity check with verification checks.

Verification checklist:
1. Overfit 1 sample first
2. Overfit 10 samples
3. Print predictions every epoch
4. Gradient check every N steps
5. Visualize every 10 epochs
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from dataset import SquareDataset
from model import TinyLocNet, count_parameters
from visualize import create_grid_visualization


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


def overfit_single_sample(model: torch.nn.Module, dataset: SquareDataset, device: torch.device, max_steps: int = 2000) -> bool:
    """
    Test 1: Overfit on a single sample.
    Should reach near-zero loss quickly. If not, something is fundamentally broken.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Overfit single sample")
    print("=" * 60)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Get single sample
    img, target = dataset[0]
    img = img.unsqueeze(0).to(device)  # Add batch dim
    target = target.unsqueeze(0).to(device)

    print(f"Target: ({target[0, 0].item():.4f}, {target[0, 1].item():.4f})")

    for step in range(max_steps):
        optimizer.zero_grad()
        pred = model(img)
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:4d}: loss = {loss.item():.6f}, pred = ({pred[0, 0].item():.4f}, {pred[0, 1].item():.4f})")

        if loss.item() < 1e-5:
            print(f"\n SUCCESS: Converged at step {step} with loss = {loss.item():.6f}")
            return True

    final_loss = loss.item()
    if final_loss < 0.001:
        print(f"\n SUCCESS: Final loss = {final_loss:.6f}")
        return True
    else:
        print(f"\n FAILURE: Did not converge. Final loss = {final_loss:.6f}")
        print("This indicates a fundamental problem with the model or optimization.")
        return False


def overfit_ten_samples(model: torch.nn.Module, dataset: SquareDataset, device: torch.device, max_epochs: int = 500) -> bool:
    """
    Test 2: Overfit on 10 samples.
    Should reach near-zero loss within a few hundred epochs.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Overfit 10 samples")
    print("=" * 60)

    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create subset of 10 samples
    subset = Subset(dataset, range(10))
    loader = DataLoader(subset, batch_size=10, shuffle=False)

    for epoch in range(max_epochs):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = F.mse_loss(preds, targets)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d}: loss = {loss.item():.6f}")

            # Print a few predictions
            with torch.no_grad():
                for i in range(min(3, len(preds))):
                    print(f"  Sample {i}: pred=({preds[i, 0].item():.3f}, {preds[i, 1].item():.3f}) "
                          f"target=({targets[i, 0].item():.3f}, {targets[i, 1].item():.3f})")

        if loss.item() < 0.001:
            print(f"\n SUCCESS: Converged at epoch {epoch} with loss = {loss.item():.6f}")
            return True

    final_loss = loss.item()
    if final_loss < 0.01:
        print(f"\n SUCCESS: Final loss = {final_loss:.6f}")
        return True
    else:
        print(f"\n FAILURE: Did not converge. Final loss = {final_loss:.6f}")
        return False


def full_training(
    model: torch.nn.Module,
    train_dataset: SquareDataset,
    val_dataset: SquareDataset,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_dir: Path = Path("outputs"),
) -> dict:
    """Full training with all verification checks.

    Args:
        model: Neural network model to train.
        train_dataset: Dataset used for training.
        val_dataset: Dataset used for validation.
        device: Device on which to run the training (e.g. CPU or CUDA).
        epochs: Number of training epochs to run.
        batch_size: Batch size for training and validation data loaders.
        lr: Learning rate for the optimizer.
        output_dir: Directory where training outputs such as visualizations
            are saved.

    Returns:
        dict: History dictionary containing per-epoch losses with keys
        ``"train_loss"`` and ``"val_loss"``.
    """
    print("\n" + "=" * 60)
    print("FULL TRAINING")
    print("=" * 60)

    # Reset model weights
    model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
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

        avg_train_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_targets = []
        all_images = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                preds = model(images)
                loss = F.mse_loss(preds, targets)
                val_loss += loss.item()
                val_batches += 1

                # Store for visualization
                if len(all_images) < 8:
                    for i in range(min(8 - len(all_images), len(images))):
                        all_images.append(images[i].cpu())
                        all_preds.append((preds[i, 0].item(), preds[i, 1].item()))
                        all_targets.append((targets[i, 0].item(), targets[i, 1].item()))

        avg_val_loss = val_loss / val_batches
        history["val_loss"].append(avg_val_loss)

        # Print epoch summary with predictions
        print(f"\nEpoch {epoch + 1:3d}/{epochs}: train_loss = {avg_train_loss:.6f}, val_loss = {avg_val_loss:.6f}")

        # Print a few predictions
        print("Sample predictions:")
        for i in range(min(4, len(all_preds))):
            pred = all_preds[i]
            tgt = all_targets[i]
            error = ((pred[0] - tgt[0]) ** 2 + (pred[1] - tgt[1]) ** 2) ** 0.5
            print(f"  pred=({pred[0]:.3f}, {pred[1]:.3f}) target=({tgt[0]:.3f}, {tgt[1]:.3f}) error={error:.4f}")

        # Visualization every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            vis_path = output_dir / f"epoch_{epoch + 1:03d}.png"
            create_grid_visualization(all_images, all_preds, all_targets, vis_path)
            print(f"Saved visualization to {vis_path}")

        # Early stopping check
        if avg_val_loss < 0.001:
            print(f"\n SUCCESS: Reached target loss at epoch {epoch + 1}")
            break

    # Final visualization
    vis_path = output_dir / "final.png"
    create_grid_visualization(all_images, all_preds, all_targets, vis_path)
    print(f"Saved final visualization to {vis_path}")

    return history


def main():
    """Run all verification checks.

    Returns:
        dict[str, bool | float]: Mapping of verification check names to
            their outcomes and metrics.
    """
    print("=" * 60)
    print("BASELINE SANITY CHECK")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = TinyLocNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Create datasets
    train_dataset = SquareDataset(size=1000, seed=42)
    val_dataset = SquareDataset(size=100, seed=123)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Run verification checks
    results = {}

    # Test 1: Overfit single sample
    results["overfit_1"] = overfit_single_sample(model, train_dataset, device)

    # Test 2: Overfit 10 samples
    results["overfit_10"] = overfit_ten_samples(model, train_dataset, device)

    # Test 3: Full training
    if results["overfit_1"] and results["overfit_10"]:
        print("\nBasic tests passed. Starting full training...")
        history = full_training(
            model, train_dataset, val_dataset, device,
            epochs=50, batch_size=32, lr=1e-3,
            output_dir=Path("outputs"),
        )
        results["final_val_loss"] = history["val_loss"][-1]
        results["training_success"] = history["val_loss"][-1] < 0.01
    else:
        print("\nBasic tests FAILED. Skipping full training.")
        print("There is a fundamental problem that needs to be fixed first.")
        results["training_success"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Overfit 1 sample: {'PASS' if results['overfit_1'] else 'FAIL'}")
    print(f"Overfit 10 samples: {'PASS' if results['overfit_10'] else 'FAIL'}")
    if "final_val_loss" in results:
        print(f"Final validation loss: {results['final_val_loss']:.6f}")
    print(f"Overall success: {'PASS' if results.get('training_success', False) else 'FAIL'}")

    return results


if __name__ == "__main__":
    main()
