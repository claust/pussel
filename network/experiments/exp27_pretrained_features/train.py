"""Training entry point for exp27: frozen pretrained features + trainable heads.

Reuses the exp20 methodology harness unchanged (``harness.fit`` /
``harness.evaluate``, val-based checkpoint selection, test touched once via
``--eval-test``) and the exp26 domain-randomization data path unchanged (same
frozen split, same augmentations — here they regularize the adapters/heads
rather than trying to close sim-to-real by themselves). The differences from
exp26 are the model (frozen DINOv2 + adapters + heads, ~1.9M trainable) and
the input sizes (piece 224, puzzle 448 — ViT patch-14 grids that divide the
4x4 cells exactly).

North-star discipline is inherited: val selects the checkpoint, the synthetic
test set is touched once with ``--eval-test``, and the north_star real-photo
benchmark is never seen here (evaluate it once, separately).

Run from the network/ directory:
    uv run python -m experiments.exp27_pretrained_features.train --epochs 25
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import torch

from ..exp20_realistic_pieces.harness import SELECTION_METRIC, evaluate, fit, load_best_checkpoint
from ..exp20_realistic_pieces.model import FastBackboneModel
from ..exp20_realistic_pieces.visualize import save_prediction_grid, save_training_curves
from ..exp26_domain_randomization.aug_dataset import GRID_SIZE, NUM_CELLS, create_datasets_from_split
from ..exp26_domain_randomization.augment import AUG_PRESETS, AugmentConfig, config_to_dict, seed_everything
from .model import DEFAULT_ENCODER, FrozenFeatureModel, PositionHeadType, heatmap_ce_loss

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "frozen_features"


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def export_state_dict_checkpoint(output_dir: Path, device: torch.device) -> Path:
    """Write a raw-state_dict copy of checkpoint_best.pt for standalone evaluators.

    Args:
        output_dir: Directory containing ``checkpoint_best.pt``.
        device: Device to map the checkpoint to while loading.

    Returns:
        Path to the written raw-state_dict checkpoint.
    """
    ckpt_path = output_dir / "checkpoint_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"'model_state_dict' missing from {ckpt_path} (keys: {sorted(ckpt)})")
    raw_path = output_dir / "checkpoint_best_state_dict.pt"
    torch.save(ckpt["model_state_dict"], raw_path)
    return raw_path


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Train exp27 (frozen features) with val-based selection.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Results dictionary (also written to results.json).
    """
    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AUG_PRESETS[args.aug_preset]
    config = AugmentConfig(**config_to_dict(config))  # type: ignore[arg-type]
    if args.disable_aug:
        config.enabled = False

    print("\n" + "=" * 70)
    print("EXP27 FROZEN PRETRAINED FEATURES (frozen split + val selection)")
    print("=" * 70)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells | epochs={args.epochs} | batch={args.batch_size}")
    print(f"Encoder: {args.encoder} (frozen{', last block unfrozen' if args.unfreeze_last_block else ''})")
    print(f"Position head: {args.position_head} | piece {args.piece_size} | puzzle {args.puzzle_size}")
    print(f"Aug preset: {args.aug_preset} | flags: {config.ablation_flags()}")

    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device} (AMP: {use_amp})")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset_kwargs: dict[str, Any] = {
        "augment_config": config,
        "piece_size": args.piece_size,
        "puzzle_size": args.puzzle_size,
        "allow_missing": args.allow_missing_puzzles,
    }
    if args.dataset_root is not None:
        dataset_kwargs["dataset_root"] = args.dataset_root
    if args.puzzle_root is not None:
        dataset_kwargs["puzzle_root"] = args.puzzle_root
    if args.split_path is not None:
        dataset_kwargs["split_path"] = args.split_path
    datasets = create_datasets_from_split(**dataset_kwargs)

    print(f"\nCreating model ({args.encoder})...")
    position_head = cast(PositionHeadType, args.position_head)
    model = FrozenFeatureModel(
        encoder_name=args.encoder,
        adapter_dim=args.adapter_dim,
        position_head=position_head,
        grid_size=GRID_SIZE,
        unfreeze_last_block=args.unfreeze_last_block,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total_params:,} (trainable {trainable_params:,}) | random cell baseline {1.0 / NUM_CELLS:.1%}")

    optimizer = torch.optim.AdamW(
        model.get_parameter_groups(
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    )

    # The harness annotates FastBackboneModel; FrozenFeatureModel implements
    # the same interface (forward signature, positions_to_cells).
    harness_model = cast(FastBackboneModel, model)

    if args.position_loss == "heatmap_ce" and args.position_head != "dense":
        raise ValueError("--position-loss heatmap_ce requires --position-head dense")
    position_loss_fn = heatmap_ce_loss if args.position_loss == "heatmap_ce" else None

    train_start = time.time()
    history, best_epoch, best_val_metrics = fit(
        harness_model,
        train_dataset=datasets["train"],
        train_eval_dataset=datasets["train_eval"],
        val_dataset=datasets["val"],
        optimizer=optimizer,
        device=device,
        grid_size=GRID_SIZE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=output_dir,
        num_workers=args.num_workers,
        use_amp=use_amp,
        position_loss_fn=position_loss_fn,
    )
    total_time = time.time() - train_start

    results: dict[str, Any] = {
        "experiment": "exp27_pretrained_features",
        "encoder": args.encoder,
        "position_head": args.position_head,
        "position_loss": args.position_loss,
        "adapter_dim": args.adapter_dim,
        "unfreeze_last_block": args.unfreeze_last_block,
        "grid_size": GRID_SIZE,
        "num_cells": NUM_CELLS,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "piece_size": args.piece_size,
        "puzzle_size": args.puzzle_size,
        "seed": args.seed,
        "device": str(device),
        "amp": use_amp,
        "total_training_time": total_time,
        "selection_metric": f"val_{SELECTION_METRIC}",
        "aug_preset": args.aug_preset,
        "aug_config": config_to_dict(config),
        "aug_flags": config.ablation_flags(),
        "best_epoch": best_epoch,
        "best_val_cell_acc": best_val_metrics["cell_accuracy"],
        "best_val_rot_acc": best_val_metrics["rotation_accuracy"],
        "best_val_both_acc": best_val_metrics["both_accuracy"],
        "history": history,
    }

    save_training_curves(history, output_dir / "training_curves.png")
    print("Saved training_curves.png")

    raw_ckpt = export_state_dict_checkpoint(output_dir, device)
    print(f"Exported raw state_dict checkpoint for north-star eval: {raw_ckpt.name}")

    if args.eval_test:
        print("\nEvaluating TEST set once on the best-val checkpoint...")
        checkpoint_epoch = load_best_checkpoint(harness_model, output_dir, device)
        test_metrics = evaluate(
            harness_model,
            datasets["test"],
            device,
            grid_size=GRID_SIZE,
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            collect=True,
        )
        print(
            f"TEST (epoch {checkpoint_epoch}): cell={test_metrics['cell_accuracy']:.1%}, "
            f"rot={test_metrics['rotation_accuracy']:.1%}, both={test_metrics['both_accuracy']:.1%}"
        )
        results["test_cell_acc"] = test_metrics["cell_accuracy"]
        results["test_rot_acc"] = test_metrics["rotation_accuracy"]
        results["test_both_acc"] = test_metrics["both_accuracy"]
        results["test_n_samples"] = test_metrics["n_samples"]

        save_prediction_grid(
            predictions=test_metrics["predictions"],
            targets=test_metrics["targets"],
            pred_cells=test_metrics["pred_cells"],
            true_cells=test_metrics["true_cells"],
            pred_rotations=test_metrics["pred_rotations"],
            true_rotations=test_metrics["true_rotations"],
            output_path=output_dir / "test_predictions.png",
        )
        print("Saved test_predictions.png")
    else:
        print("\nTest set NOT evaluated (pass --eval-test for the one-shot final evaluation).")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="exp27 frozen-feature training (frozen split, val selection)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (frozen heads converge fast)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, help="timm name of the frozen patch-14 ViT")
    parser.add_argument("--adapter-dim", type=int, default=256, help="Adapter output dimension")
    parser.add_argument(
        "--position-head", choices=("dense", "pooled"), default="dense", help="Position head (pooled = exp20 ablation)"
    )
    parser.add_argument(
        "--position-loss",
        choices=("heatmap_ce", "mse"),
        default="heatmap_ce",
        help="Position loss: window cross-entropy on the dense heatmap (default) or exp20-style MSE (ablation)",
    )
    parser.add_argument(
        "--unfreeze-last-block", action="store_true", help="Phase 2: train the encoder's last block too"
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="LR for unfrozen encoder params")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="LR for adapters and heads")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--piece-size", type=int, default=224, help="Piece image size (multiple of 14)")
    parser.add_argument("--puzzle-size", type=int, default=448, help="Puzzle image size (multiple of 14)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--dataset-root", type=Path, default=None, help="RGBA pieces dataset root")
    parser.add_argument("--puzzle-root", type=Path, default=None, help="Source puzzle images root")
    parser.add_argument("--split-path", type=Path, default=None, help="Frozen split JSON (default: exp20 v1)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers")
    parser.add_argument("--eval-test", action="store_true", help="Evaluate the test set ONCE after training")
    parser.add_argument(
        "--allow-missing-puzzles",
        action="store_true",
        help="Tolerate missing puzzle dirs (smoke tests only; results NOT comparable)",
    )
    parser.add_argument(
        "--aug-preset",
        choices=sorted(AUG_PRESETS.keys()),
        default="full",
        help="Augmentation preset (exp26 domain randomization)",
    )
    parser.add_argument("--disable-aug", action="store_true", help="Master off switch (black-composite like exp20)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
