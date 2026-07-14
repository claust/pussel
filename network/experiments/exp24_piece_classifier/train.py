"""Train the binary piece classifier.

Selection is on validation balanced accuracy; the test split is evaluated
once at the end on the best-val checkpoint, with a confusion matrix and a
per-category breakdown (faces and household objects are the false-positive
cases the classifier exists to kill).

Run from the network/ directory (the default --data-root is
network/datasets/piece_classifier):
    uv run python -m experiments.exp24_piece_classifier.train --epochs 12
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    DEFAULT_DATA_ROOT,
    INPUT_SIZE,
    PieceClassifierDataset,
    Sample,
    collect_samples,
    make_splits,
    summarize,
)
from .model import PieceClassifier, count_parameters

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Returns:
        The selected torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model: PieceClassifier, dataset: PieceClassifierDataset, device: torch.device, batch_size: int) -> tuple:
    """Compute probabilities and labels over a dataset in order.

    Args:
        model: The classifier.
        dataset: Dataset to evaluate (must not shuffle-augment labels).
        device: Device to run on.
        batch_size: Evaluation batch size.

    Returns:
        Tuple of (probabilities, labels) as float numpy arrays.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, targets in loader:
        p = torch.sigmoid(model(images.to(device)))
        probs.append(p.cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def compute_metrics(probs: "np.ndarray", labels: "np.ndarray", threshold: float = 0.5) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        probs: Predicted piece probabilities.
        labels: Ground-truth labels (1 = piece).
        threshold: Decision threshold.

    Returns:
        Dict with accuracy, balanced accuracy, precision, recall, AUC and
        confusion-matrix counts (tn, fp, fn, tp).
    """
    from sklearn.metrics import roc_auc_score

    preds = (probs >= threshold).astype(np.int64)
    y = labels.astype(np.int64)
    tp = int(((preds == 1) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    pos_recall = tp / max(1, tp + fn)
    neg_recall = tn / max(1, tn + fp)
    auc = float(roc_auc_score(y, probs)) if len(set(y.tolist())) > 1 else float("nan")
    return {
        "accuracy": (tp + tn) / max(1, len(y)),
        "balanced_accuracy": (pos_recall + neg_recall) / 2,
        "precision": tp / max(1, tp + fp),
        "recall": pos_recall,
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def category_breakdown(probs: "np.ndarray", samples: list[Sample], threshold: float = 0.5) -> dict[str, dict]:
    """Per-category accuracy breakdown for a test evaluation.

    Args:
        probs: Predicted probabilities aligned with samples.
        samples: The evaluated samples, in prediction order.
        threshold: Decision threshold.

    Returns:
        Mapping "source/category" -> {n, correct, accuracy, mean_prob}.
    """
    breakdown: dict[str, dict] = {}
    for prob, sample in zip(probs, samples):
        key = f"{sample.source}/{sample.category}"
        entry = breakdown.setdefault(key, {"n": 0, "correct": 0, "prob_sum": 0.0, "label": sample.label})
        predicted = int(prob >= threshold)
        entry["n"] += 1
        entry["correct"] += int(predicted == sample.label)
        entry["prob_sum"] += float(prob)
    for entry in breakdown.values():
        entry["accuracy"] = entry["correct"] / entry["n"]
        entry["mean_prob"] = entry.pop("prob_sum") / entry["n"]
    return breakdown


def benchmark_cpu_latency(model: PieceClassifier, runs: int = 20) -> float:
    """Measure single-image CPU inference latency.

    Args:
        model: The classifier (will be moved to CPU).
        runs: Number of timed runs.

    Returns:
        Median latency in milliseconds.
    """
    model = model.cpu().eval()
    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def main(
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the classifier and evaluate the test split once.

    Args:
        epochs: Training epochs.
        batch_size: Batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        data_root: Dataset root (positives/ + negatives/).
        output_dir: Where checkpoints and results land.
        num_workers: DataLoader workers.
        seed: Split/shuffle seed.

    Returns:
        Results dict (also written to results.json).
    """
    # Seed every RNG in play: torch (model init, torchvision transforms),
    # random (RandomDownscale, split shuffling) and numpy.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    samples = collect_samples(data_root)
    splits = make_splits(samples, seed=seed)
    for name in ("train", "val", "test"):
        print(summarize(splits[name], name))

    train_ds = PieceClassifierDataset(splits["train"], train=True)
    val_ds = PieceClassifierDataset(splits["val"])
    test_ds = PieceClassifierDataset(splits["test"])

    n_pos = sum(1 for s in splits["train"] if s.label == 1)
    n_neg = len(splits["train"]) - n_pos
    pos_weight = torch.tensor(n_neg / max(1, n_pos), device=device)
    print(f"Device: {device}; pos_weight={pos_weight.item():.3f}")

    model = PieceClassifier(pretrained=True).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    best_metric = -1.0
    best_epoch = -1
    history: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "checkpoint_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        for images, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.shape[0]
        epoch_loss /= max(1, len(loader) * batch_size)

        val_probs, val_labels = predict(model, val_ds, device, batch_size * 2)
        val_metrics = compute_metrics(val_probs, val_labels)
        history.append({"epoch": epoch, "train_loss": epoch_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(
            f"Epoch {epoch:2d}/{epochs} loss={epoch_loss:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f} val_auc={val_metrics['auc']:.4f} "
            f"({time.time() - start:.1f}s)"
        )

        if val_metrics["balanced_accuracy"] > best_metric:
            best_metric = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"architecture": "mobilenet_v3_small", "input_size": INPUT_SIZE},
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_path,
            )

    print(f"\nBest epoch: {best_epoch} (val balanced accuracy {best_metric:.3f})")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_probs, test_labels = predict(model, test_ds, device, batch_size * 2)
    test_metrics = compute_metrics(test_probs, test_labels)
    breakdown = category_breakdown(test_probs, splits["test"])
    print("\nTEST metrics:")
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "auc"):
        print(f"  {key}: {test_metrics[key]:.4f}")
    cm = {k: test_metrics[k] for k in ("tn", "fp", "fn", "tp")}
    print(f"  confusion matrix: tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")
    print("\nPer-category test breakdown (accuracy, mean piece-probability):")
    for key in sorted(breakdown):
        entry = breakdown[key]
        print(f"  {key:32s} n={entry['n']:4d} acc={entry['accuracy']:.3f} mean_prob={entry['mean_prob']:.3f}")

    latency_ms = benchmark_cpu_latency(model)
    print(f"\nCPU latency (single image, median): {latency_ms:.1f} ms")
    model.to(device)

    results: dict[str, Any] = {
        "experiment": "exp24_piece_classifier",
        "architecture": "mobilenet_v3_small",
        "input_size": INPUT_SIZE,
        "parameters": count_parameters(model),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_metric,
        "test_metrics": test_metrics,
        "test_breakdown": breakdown,
        "cpu_latency_ms": latency_ms,
        "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")
    print(f"Checkpoint: {checkpoint_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the binary piece classifier")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        seed=args.seed,
    )
