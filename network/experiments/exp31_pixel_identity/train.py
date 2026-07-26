"""Training entry point for exp31: exp26's recipe on two-capture data.

The training recipe is **exp26/exp30, unchanged** — same model
(``FastBackboneModel`` / ShuffleNetV2_x0.5), same optimizer and LRs, same
128 px piece / 256 px overview, same frozen exp20 split, same exp20 harness
(val selects the checkpoint, the synthetic test set is touched once with
``--eval-test``), and the same output artifact names, so
``exp25_north_star_eval/evaluate.py`` consumes
``checkpoint_best_state_dict.pt`` with no changes.

The only difference is what the two views *are*. exp30 handed the model a
piece and an overview that came out of one JPEG through one pipeline; exp31
hands it two independent captures of two different physical prints — separate
resample kernels and subpixel phases, separate blur/sharpen, separate sensor
noise, separate non-linear tone curves, separate JPEG block grids, plus
Asymmetric Random Patching into exactly one view, rembg-calibrated
segmentation slop, a bright die-cut rim and a photographed-box overview.

Prediction under test: the section 8 NCC-headroom probe passes first (median
masked NCC at the ground-truth location falls from 0.99 toward the real 0.73
while the wrong-cell decoy stays near 0.41), and then north_star
**both-correct moves off the 12.7-13.2% floor**. If the probe passes and
transfer still does not move, pixel identity is *not* the remaining blocker
and the ranking in section 7 needs rethinking again (Test 3's corpus swap and
Test 4's FDA become the live hypotheses).

Run from the network/ directory:
    uv run python -m experiments.exp31_pixel_identity.train --epochs 50 --eval-test
    uv run python -m experiments.exp31_pixel_identity.train --capture-preset no_arp --epochs 50
"""

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..exp20_realistic_pieces.harness import SELECTION_METRIC, evaluate, fit, load_best_checkpoint
from ..exp20_realistic_pieces.model import FastBackboneModel, count_parameters
from ..exp20_realistic_pieces.visualize import save_prediction_grid, save_training_curves
from ..exp26_domain_randomization.augment import seed_everything
from .capture import CAPTURE_PRESETS, CaptureConfig, capture_config_to_dict
from .capture_dataset import GRID_SIZE, NUM_CELLS, create_datasets_from_split

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs" / "pixel_identity"


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# CLI switch -> the config field(s) it turns off. Adding an ablation lever
# means one entry here plus one ``parser.add_argument`` in
# ``_add_ablation_args`` — the two lists are the whole contract.
DISABLE_SWITCHES: dict[str, tuple[str, ...]] = {
    "disable_aug": ("enabled",),
    "no_capture": ("capture",),
    "no_photometric": ("photometric",),
    "no_scale": ("scale_jitter",),
    "no_perspective": ("perspective",),
    "no_rotation_jitter": ("rotation_jitter",),
    "no_background": ("background",),
    "no_independent_chains": ("independent_chains",),
    "no_resolution_asymmetry": ("resolution_asymmetry",),
    "no_arp": ("arp",),
    "no_scene_surface": ("scene_surface",),
    "no_seg_slop": ("seg_slop",),
    "no_rim": ("rim", "shadow"),
    "no_box_photo": ("box_photo",),
    "no_crop_jitter": ("crop_jitter",),
    "no_piece_lighting": ("piece_lighting",),
}

# Switches whose field lives on the nested per-view ViewChainConfig rather
# than flat on CaptureConfig; applied to both chains.
CHAIN_DISABLE_SWITCHES: dict[str, tuple[str, ...]] = {
    "no_substrate": ("substrate",),
}


def build_config(args: argparse.Namespace) -> CaptureConfig:
    """Build the capture config from a preset plus CLI overrides.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The resolved :class:`CaptureConfig`.
    """
    # Deep copy: the presets hold nested ViewChainConfig instances that must
    # never be mutated in place (they are module-level singletons).
    config = copy.deepcopy(CAPTURE_PRESETS[args.capture_preset])

    for switch, fields in DISABLE_SWITCHES.items():
        if getattr(args, switch):
            for name in fields:
                setattr(config, name, False)

    for switch, fields in CHAIN_DISABLE_SWITCHES.items():
        if getattr(args, switch):
            for chain in (config.piece_chain, config.overview_chain):
                for name in fields:
                    setattr(chain, name, False)

    if args.overview_target_px is not None:
        config.overview_chain.target_px = args.overview_target_px
    if args.piece_target_px is not None:
        config.piece_chain.target_px = args.piece_target_px
    return config


def export_state_dict_checkpoint(output_dir: Path, device: torch.device) -> Path:
    """Write a raw-state_dict checkpoint for the north-star evaluator.

    ``harness`` saves ``checkpoint_best.pt`` as a dict
    (``{"model_state_dict": ...}``), but ``exp25_north_star_eval/evaluate.py``
    loads a *raw* state_dict via ``load_state_dict(torch.load(...,
    weights_only=True))``. Export a raw copy under exp26/exp30's filename so
    north-star evaluation needs no format juggling.

    Args:
        output_dir: Directory containing ``checkpoint_best.pt``.
        device: Device to map the checkpoint to while loading.

    Returns:
        Path to the written raw-state_dict checkpoint.

    Raises:
        KeyError: If the harness checkpoint has no ``model_state_dict``.
    """
    ckpt_path = output_dir / "checkpoint_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"'model_state_dict' missing from {ckpt_path} (keys: {sorted(ckpt)})")
    raw_path = output_dir / "checkpoint_best_state_dict.pt"
    torch.save(ckpt["model_state_dict"], raw_path)
    return raw_path


def _build_datasets(
    args: argparse.Namespace,
    config: CaptureConfig,
) -> dict[str, torch.utils.data.Dataset]:  # type: ignore[type-arg]
    """Build the four datasets from the CLI arguments.

    Args:
        args: Parsed CLI namespace.
        config: Resolved capture config.

    Returns:
        Mapping of split name to dataset.
    """
    dataset_kwargs: dict[str, Any] = {
        "capture_config": config,
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
    return create_datasets_from_split(**dataset_kwargs)


def _evaluate_test(
    model: FastBackboneModel,
    dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    results: dict[str, Any],
) -> None:
    """Run the one-shot synthetic test evaluation and record it.

    Args:
        model: Trained model (reloaded from the best-val checkpoint here).
        dataset: The test dataset.
        device: Compute device.
        args: Parsed CLI namespace.
        output_dir: Output directory.
        results: Results dict to update in place.
    """
    print("\nEvaluating TEST set once on the best-val checkpoint...")
    checkpoint_epoch = load_best_checkpoint(model, output_dir, device)
    metrics = evaluate(
        model,
        dataset,
        device,
        grid_size=GRID_SIZE,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        collect=True,
    )
    print(
        f"TEST (epoch {checkpoint_epoch}): cell={metrics['cell_accuracy']:.1%}, "
        f"rot={metrics['rotation_accuracy']:.1%}, both={metrics['both_accuracy']:.1%}"
    )
    results["test_cell_acc"] = metrics["cell_accuracy"]
    results["test_rot_acc"] = metrics["rotation_accuracy"]
    results["test_both_acc"] = metrics["both_accuracy"]
    results["test_n_samples"] = metrics["n_samples"]

    save_prediction_grid(
        predictions=metrics["predictions"],
        targets=metrics["targets"],
        pred_cells=metrics["pred_cells"],
        true_cells=metrics["true_cells"],
        pred_rotations=metrics["pred_rotations"],
        true_rotations=metrics["true_rotations"],
        output_path=output_dir / "test_predictions.png",
    )
    print("Saved test_predictions.png")


def main(args: argparse.Namespace) -> dict[str, Any]:
    """Train exp31 on two-independent-captures data with val-based selection.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Results dictionary (also written to results.json).

    Raises:
        ValueError: If ``--epochs`` is below 1.
    """
    if args.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {args.epochs}")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args)

    print("\n" + "=" * 70)
    print("EXP31 PIXEL IDENTITY (exp26 recipe, two independent captures)")
    print("=" * 70)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {NUM_CELLS} cells | epochs={args.epochs} | batch={args.batch_size}")
    print(f"Capture preset: {args.capture_preset} | flags: {config.ablation_flags()}")
    print(
        f"Chain scales: piece {config.piece_chain.scale_min}-{config.piece_chain.scale_max}, "
        f"overview {config.overview_chain.scale_min}-{config.overview_chain.scale_max} "
        f"(target_px piece={config.piece_chain.target_px}, overview={config.overview_chain.target_px})"
    )

    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Device: {device} (AMP: {use_amp})")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    datasets = _build_datasets(args, config)

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
        "experiment": "exp31_pixel_identity",
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
        "capture_preset": args.capture_preset,
        "capture_config": capture_config_to_dict(config),
        "capture_flags": config.ablation_flags(),
        "framing": "lossless_transpose + jittered pad_to_square(margin=0.05-0.12)",
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
        _evaluate_test(model, datasets["test"], device, args, output_dir, results)
    else:
        print("\nTest set NOT evaluated (pass --eval-test for the one-shot final evaluation).")

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_dir / 'results.json'}")
    return results


def _add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add the per-component ablation switches.

    Args:
        parser: Parser to extend.
    """
    parser.add_argument(
        "--capture-preset",
        choices=sorted(CAPTURE_PRESETS.keys()),
        default="full",
        help="Capture preset (ablation entry point; 'exp30' reproduces exp30)",
    )
    parser.add_argument("--disable-aug", action="store_true", help="Master off switch (black-composite like exp20)")
    parser.add_argument("--no-capture", action="store_true", help="Disable all exp31 additions (exp26/exp30 path)")
    parser.add_argument("--no-photometric", action="store_true", help="Disable exp26 photometric jitter")
    parser.add_argument("--no-scale", action="store_true", help="Disable exp26 piece scale jitter")
    parser.add_argument("--no-perspective", action="store_true", help="Disable exp26 piece perspective warp")
    parser.add_argument("--no-rotation-jitter", action="store_true", help="Disable sub-90-degree rotation jitter")
    parser.add_argument("--no-background", action="store_true", help="Disable realistic backgrounds (always black)")
    parser.add_argument(
        "--no-independent-chains",
        action="store_true",
        help="Run ONE chain over both views from one RNG state (the shared-pass diagnostic that should fail)",
    )
    parser.add_argument(
        "--no-resolution-asymmetry",
        action="store_true",
        help="Give the overview the piece's sampling budget (independent draws, matched MTF)",
    )
    parser.add_argument("--no-arp", action="store_true", help="Disable Asymmetric Random Patching")
    parser.add_argument("--no-scene-surface", action="store_true", help="Disable the surface the piece bleeds from")
    parser.add_argument("--no-seg-slop", action="store_true", help="Disable the rembg-calibrated mask slop")
    parser.add_argument("--no-rim", action="store_true", help="Disable the bright die-cut rim and cast shadow")
    parser.add_argument("--no-substrate", action="store_true", help="Disable the substrate/illumination field")
    parser.add_argument("--no-piece-lighting", action="store_true", help="Disable room lighting on the piece view")
    parser.add_argument(
        "--no-box-photo", action="store_true", help="Disable glare/perspective/vignette on the overview"
    )
    parser.add_argument("--no-crop-jitter", action="store_true", help="Disable crop/bbox jitter on the piece")
    parser.add_argument(
        "--overview-target-px",
        type=int,
        default=None,
        help="Absolute overview sampling budget in px (higher-res corpus knob; ~500 for 125 px/cell at 4x4)",
    )
    parser.add_argument(
        "--piece-target-px",
        type=int,
        default=None,
        help="Absolute piece sampling budget in px (leave unset so the piece keeps its native detail)",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="exp31 pixel-identity training (frozen split, val selection)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--backbone-lr", type=float, default=1e-4, help="Backbone LR")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Head LR")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--piece-size", type=int, default=128, help="Piece image size")
    parser.add_argument("--puzzle-size", type=int, default=256, help="Overview image size")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--dataset-root", type=Path, default=None, help="RGBA pieces dataset root (exp30's v2)")
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
    _add_ablation_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
