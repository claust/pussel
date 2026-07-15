"""Training entry point for exp26: domain-randomization on realistic pieces.

Reuses the exp20 methodology harness unchanged (``harness.fit`` /
``harness.evaluate``, val-based checkpoint selection, test touched once via
``--eval-test``) and the exp20 model. The ONLY difference from exp20 is the
training data path: pieces are RGBA and pushed through
``augment.augment_piece`` with an independently jittered puzzle
(``augment.augment_puzzle``), so the model can no longer win by
pixel-identical matching (the exp25 failure).

North-star discipline is inherited from the frozen split: val selects the
checkpoint, the synthetic test set is touched once with ``--eval-test``,
and the north-star real-photo benchmark is never seen here (evaluate it
once, separately, with ``exp25_north_star_eval/evaluate.py``).

Run from the network/ directory:
    uv run python -m experiments.exp26_domain_randomization.train --epochs 50
    uv run python -m experiments.exp26_domain_randomization.train --aug-preset no_photometric --epochs 50
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..exp20_realistic_pieces.harness import SELECTION_METRIC, evaluate, fit, load_best_checkpoint
from ..exp20_realistic_pieces.model import FastBackboneModel, count_parameters
from ..exp20_realistic_pieces.visualize import save_prediction_grid, save_training_curves
from .aug_dataset import GRID_SIZE, NUM_CELLS, create_datasets_from_split
from .augment import AUG_PRESETS, AugmentConfig, config_to_dict, seed_everything

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "domain_randomization"


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(args: argparse.Namespace) -> AugmentConfig:
    """Build the augmentation config from a preset plus CLI overrides.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The resolved ``AugmentConfig``.
    """
    preset = AUG_PRESETS[args.aug_preset]
    # Copy so we never mutate the shared preset instance.
    config = AugmentConfig(**config_to_dict(preset))  # type: ignore[arg-type]

    if args.disable_aug:
        config.enabled = False
    if args.no_photometric:
        config.photometric = False
    if args.no_scale:
        config.scale_jitter = False
    if args.no_perspective:
        config.perspective = False
    if args.no_rotation_jitter:
        config.rotation_jitter = False
    if args.no_background:
        config.background = False
    if args.no_noise:
        config.noise = False
    if args.no_jpeg:
        config.jpeg = False
    return config


def export_state_dict_checkpoint(output_dir: Path, device: torch.device) -> Path:
    """Write a raw-state_dict checkpoint for the north-star evaluator.

    ``harness`` saves ``checkpoint_best.pt`` as a dict
    (``{"model_state_dict": ...}``), but ``exp25_north_star_eval/evaluate.py``
    loads a *raw* state_dict. Export a raw copy so north-star evaluation
    needs no format juggling.

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
    """Train exp26 with domain randomization and val-based selection.

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
    config = build_config(args)

    print("\n" + "=" * 70)
    print("EXP26 DOMAIN RANDOMIZATION (frozen split + val selection)")
    print("=" * 70)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells | epochs={args.epochs} | batch={args.batch_size}")
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

    backbone_name = "shufflenet_v2_x0_5"
    print(f"\nCreating model ({backbone_name})...")
    model = FastBackboneModel(backbone_name=backbone_name, pretrained=True, freeze_backbone=False).to(device)
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Params: {total_params:,} (trainable {trainable_params:,}) | random cell baseline {1.0 / NUM_CELLS:.1%}")

    optimizer = torch.optim.AdamW(
        model.get_parameter_groups(
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )
    )

    train_start = time.time()
    history, best_epoch, best_val_metrics = fit(
        model,
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
    )
    total_time = time.time() - train_start

    results: dict[str, Any] = {
        "experiment": "exp26_domain_randomization",
        "backbone": backbone_name,
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
        checkpoint_epoch = load_best_checkpoint(model, output_dir, device)
        test_metrics = evaluate(
            model,
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
    parser = argparse.ArgumentParser(description="exp26 domain-randomization training (frozen split, val selection)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--piece-size", type=int, default=128, help="Piece image size")
    parser.add_argument("--puzzle-size", type=int, default=256, help="Puzzle image size")
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
        help="Augmentation preset (ablation entry point)",
    )
    parser.add_argument("--disable-aug", action="store_true", help="Master off switch (black-composite like exp20)")
    parser.add_argument("--no-photometric", action="store_true", help="Disable independent photometric jitter")
    parser.add_argument("--no-scale", action="store_true", help="Disable piece scale jitter")
    parser.add_argument("--no-perspective", action="store_true", help="Disable piece perspective warp")
    parser.add_argument("--no-rotation-jitter", action="store_true", help="Disable sub-90-degree rotation jitter")
    parser.add_argument("--no-background", action="store_true", help="Disable realistic backgrounds (always black)")
    parser.add_argument("--no-noise", action="store_true", help="Disable sensor noise")
    parser.add_argument("--no-jpeg", action="store_true", help="Disable JPEG recompression artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
