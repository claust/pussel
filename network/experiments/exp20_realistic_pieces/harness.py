"""Methodology-correct training harness for the realistic 4x4 benchmark.

Fixes the harness problems identified in CRITICAL_REVIEW.md:

- **Frozen split**: datasets come from the checked-in split JSON (see
  ``splits.py``); nothing here re-shuffles or re-partitions.
- **Checkpoint selection on val, not test**: the best checkpoint is the
  one with the highest *validation* both-correct accuracy. The test set
  is never touched during training.
- **Train metrics in eval mode**: the optimization pass accumulates
  losses only. Accuracy-style train metrics are measured after each
  epoch with ``model.eval()`` on the frozen ``train_eval`` subset, using
  the same deterministic all-4-rotations protocol as val/test, so the
  train/val gap is apples-to-apples.
- **Test touched once**: ``evaluate`` is called on the test set a single
  time per experiment, on the val-selected checkpoint (see ``train.py``).
"""

import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset

from .model import FastBackboneModel

# Optional replacement for the position MSE loss. Called with the model's
# (position, attention_map) outputs and the (cx, cy) targets; returns the
# position loss. Lets experiments with heatmap-style heads (exp27) supervise
# the attention map directly while the default stays exp20's plain MSE.
PositionLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

# The metric on which the best checkpoint is selected (computed on val).
SELECTION_METRIC = "both_accuracy"

HISTORY_KEYS = (
    "train_pos_loss",
    "train_rot_loss",
    "train_cell_acc",
    "train_rot_acc",
    "train_both_acc",
    "val_cell_acc",
    "val_rot_acc",
    "val_both_acc",
)


def train_epoch(
    model: FastBackboneModel,
    loader: DataLoader,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    position_weight: float = 1.0,
    rotation_weight: float = 1.0,
    position_loss_fn: Optional[PositionLossFn] = None,
) -> dict[str, float]:
    """Run one optimization epoch and return running losses.

    Deliberately does NOT compute accuracies: metrics measured while the
    model is in train mode (BatchNorm batch statistics, augmentation,
    weights changing mid-epoch) are not comparable to val/test metrics.
    Use ``evaluate`` on the ``train_eval`` subset for train metrics.

    Args:
        model: Model to train.
        loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        scaler: Gradient scaler for mixed precision (CUDA only); if None,
            trains in full precision.
        position_weight: Weight of the position MSE loss.
        rotation_weight: Weight of the rotation cross-entropy loss.
        position_loss_fn: Optional replacement position loss, called as
            ``fn(position, attention_map, targets)``. Default: MSE on the
            position output (exp20 behaviour).

    Returns:
        Dictionary with sample-weighted average position and rotation losses.
    """
    model.train()
    total_position_loss = 0.0
    total_rotation_loss = 0.0
    total_samples = 0

    for pieces, puzzles, targets, _cells, rotations in loader:
        pieces = pieces.to(device)
        puzzles = puzzles.to(device)
        targets = targets.to(device)
        rotations = rotations.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                preds, rotation_logits, attention = model(pieces, puzzles)
                if position_loss_fn is not None:
                    position_loss = position_loss_fn(preds, attention, targets)
                else:
                    position_loss = F.mse_loss(preds, targets)
                rotation_loss = F.cross_entropy(rotation_logits, rotations)
                loss = position_weight * position_loss + rotation_weight * rotation_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds, rotation_logits, attention = model(pieces, puzzles)
            if position_loss_fn is not None:
                position_loss = position_loss_fn(preds, attention, targets)
            else:
                position_loss = F.mse_loss(preds, targets)
            rotation_loss = F.cross_entropy(rotation_logits, rotations)
            loss = position_weight * position_loss + rotation_weight * rotation_loss
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_position_loss += position_loss.item() * batch_size
        total_rotation_loss += rotation_loss.item() * batch_size
        total_samples += batch_size

    return {
        "position_loss": total_position_loss / total_samples,
        "rotation_loss": total_rotation_loss / total_samples,
    }


def evaluate(
    model: FastBackboneModel,
    dataset: Dataset,  # type: ignore[type-arg]
    device: torch.device,
    grid_size: int,
    batch_size: int = 128,
    num_workers: int = 0,
    collect: bool = False,
) -> dict[str, Any]:
    """Compute eval-mode metrics on a deterministic dataset.

    Args:
        model: Model to evaluate.
        dataset: Deterministic dataset (e.g. RealisticPieceTestDataset).
        device: Device to run on.
        grid_size: Grid size for cell prediction.
        batch_size: Evaluation batch size.
        num_workers: Data loader workers.
        collect: If True, also return per-sample predictions and targets
            (for confusion matrices and scatter plots).

    Returns:
        Dictionary with mse_loss, cell_accuracy, rotation_accuracy and
        both_accuracy (cell AND rotation correct); with ``collect=True``
        also predictions/targets/pred_cells/true_cells/pred_rotations/
        true_rotations lists.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    total_mse = 0.0
    total_cell_correct = 0
    total_rotation_correct = 0
    total_both_correct = 0
    n_samples = 0

    all_predictions: list[tuple[float, float]] = []
    all_targets: list[tuple[float, float]] = []
    all_pred_cells: list[int] = []
    all_true_cells: list[int] = []
    all_pred_rotations: list[int] = []
    all_true_rotations: list[int] = []

    with torch.no_grad():
        for pieces, puzzles, targets, cells, rotations in loader:
            pieces = pieces.to(device)
            puzzles = puzzles.to(device)
            targets_dev = targets.to(device)
            cells_dev = cells.to(device)
            rotations_dev = rotations.to(device)

            preds, rotation_logits, _ = model(pieces, puzzles)

            total_mse += F.mse_loss(preds, targets_dev, reduction="sum").item()
            n_samples += targets.size(0)

            # Derive cells from the positions already computed above
            # (model.predict_cell would run a second forward pass).
            pred_cells = model.positions_to_cells(preds, grid_size=grid_size)
            pred_rotations = rotation_logits.argmax(dim=1)

            cell_correct = pred_cells == cells_dev
            rotation_correct = pred_rotations == rotations_dev
            total_cell_correct += int(cell_correct.sum().item())
            total_rotation_correct += int(rotation_correct.sum().item())
            total_both_correct += int((cell_correct & rotation_correct).sum().item())

            if collect:
                for i in range(preds.size(0)):
                    all_predictions.append((preds[i, 0].item(), preds[i, 1].item()))
                    all_targets.append((targets[i, 0].item(), targets[i, 1].item()))
                all_pred_cells.extend(pred_cells.cpu().tolist())
                all_true_cells.extend(cells.tolist())
                all_pred_rotations.extend(pred_rotations.cpu().tolist())
                all_true_rotations.extend(rotations.tolist())

    metrics: dict[str, Any] = {
        "mse_loss": total_mse / n_samples,
        "cell_accuracy": total_cell_correct / n_samples,
        "rotation_accuracy": total_rotation_correct / n_samples,
        "both_accuracy": total_both_correct / n_samples,
        "n_samples": n_samples,
    }
    if collect:
        metrics.update(
            {
                "predictions": all_predictions,
                "targets": all_targets,
                "pred_cells": all_pred_cells,
                "true_cells": all_true_cells,
                "pred_rotations": all_pred_rotations,
                "true_rotations": all_true_rotations,
            }
        )
    return metrics


def _save_checkpoint(
    path: Path,
    model: FastBackboneModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: dict[str, Any],
) -> None:
    """Save a training checkpoint with its validation metrics."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "selection_metric": SELECTION_METRIC,
            "val_cell_acc": val_metrics["cell_accuracy"],
            "val_rot_acc": val_metrics["rotation_accuracy"],
            "val_both_acc": val_metrics["both_accuracy"],
        },
        path,
    )


def fit(
    model: FastBackboneModel,
    train_dataset: Dataset,  # type: ignore[type-arg]
    train_eval_dataset: Dataset,  # type: ignore[type-arg]
    val_dataset: Dataset,  # type: ignore[type-arg]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grid_size: int,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    num_workers: int = 0,
    use_amp: bool = False,
    eval_batch_size: int | None = None,
    position_loss_fn: Optional[PositionLossFn] = None,
) -> tuple[dict[str, list[float]], int, dict[str, Any]]:
    """Train with per-epoch val evaluation and val-based checkpoint selection.

    Each epoch: one optimization pass (losses only), then eval-mode
    metrics on the train_eval subset and on val. The checkpoint with the
    best validation ``SELECTION_METRIC`` is saved to ``checkpoint_best.pt``
    (the latest epoch always to ``checkpoint_last.pt``). The test set is
    NOT touched here.

    Args:
        model: Model to train.
        train_dataset: Augmented training dataset (random rotations).
        train_eval_dataset: Deterministic dataset over the frozen
            train_eval subset (for eval-mode train metrics).
        val_dataset: Deterministic validation dataset.
        optimizer: Optimizer.
        device: Device to train on.
        grid_size: Grid size for cell prediction.
        epochs: Number of epochs.
        batch_size: Training batch size.
        output_dir: Directory for checkpoints.
        num_workers: Data loader workers.
        use_amp: Use automatic mixed precision (CUDA only).
        eval_batch_size: Batch size for evaluation passes (default 2x train).
        position_loss_fn: Optional replacement position loss (see
            ``train_epoch``); default is exp20's position MSE.

    Returns:
        Tuple of (history, best_epoch, best val metrics).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_batch_size = eval_batch_size or batch_size * 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    scaler = GradScaler("cuda") if use_amp and device.type == "cuda" else None

    history: dict[str, list[float]] = {key: [] for key in HISTORY_KEYS}
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {SELECTION_METRIC: -1.0}

    print(f"\nTraining for {epochs} epochs (checkpoint selection: val {SELECTION_METRIC})...")
    print("-" * 100)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        losses = train_epoch(model, train_loader, optimizer, device, scaler=scaler, position_loss_fn=position_loss_fn)
        train_metrics = evaluate(
            model, train_eval_dataset, device, grid_size, batch_size=eval_batch_size, num_workers=num_workers
        )
        val_metrics = evaluate(
            model, val_dataset, device, grid_size, batch_size=eval_batch_size, num_workers=num_workers
        )

        epoch_time = time.time() - epoch_start

        history["train_pos_loss"].append(losses["position_loss"])
        history["train_rot_loss"].append(losses["rotation_loss"])
        history["train_cell_acc"].append(train_metrics["cell_accuracy"])
        history["train_rot_acc"].append(train_metrics["rotation_accuracy"])
        history["train_both_acc"].append(train_metrics["both_accuracy"])
        history["val_cell_acc"].append(val_metrics["cell_accuracy"])
        history["val_rot_acc"].append(val_metrics["rotation_accuracy"])
        history["val_both_acc"].append(val_metrics["both_accuracy"])

        is_best = val_metrics[SELECTION_METRIC] > best_val_metrics[SELECTION_METRIC]
        if is_best:
            best_epoch = epoch
            best_val_metrics = val_metrics
            _save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, epoch, val_metrics)
        _save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, epoch, val_metrics)

        print(
            f"Epoch {epoch:3d}/{epochs}: "
            f"pos_loss={losses['position_loss']:.4f}, "
            f"rot_loss={losses['rotation_loss']:.4f}, "
            f"train_cell={train_metrics['cell_accuracy']:.1%}, "
            f"train_rot={train_metrics['rotation_accuracy']:.1%}, "
            f"val_cell={val_metrics['cell_accuracy']:.1%}, "
            f"val_rot={val_metrics['rotation_accuracy']:.1%}, "
            f"val_both={val_metrics['both_accuracy']:.1%}, "
            f"time={epoch_time:.1f}s" + (" *" if is_best else "")
        )

    print("-" * 100)
    print(f"Best checkpoint: epoch {best_epoch} with val {SELECTION_METRIC}={best_val_metrics[SELECTION_METRIC]:.1%}")
    return history, best_epoch, best_val_metrics


def load_best_checkpoint(model: FastBackboneModel, output_dir: Path, device: torch.device) -> int:
    """Load checkpoint_best.pt weights into the model.

    Args:
        model: Model to load the weights into.
        output_dir: Directory containing checkpoint_best.pt.
        device: Device to map the checkpoint to.

    Returns:
        The epoch the checkpoint was saved at.
    """
    checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint["epoch"])
