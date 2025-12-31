#!/usr/bin/env python3
"""Run full exp20 experiment: generate 5000 puzzles + train 50 epochs."""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add network to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main() -> None:
    """Run full experiment."""
    print("=" * 70)
    print("EXP20: REALISTIC PIECES - FULL RUN (5000 PUZZLES)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Phase 1: Dataset Generation
    print("=" * 70)
    print("PHASE 1: GENERATING 5000 PUZZLE DATASET")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print()

    from experiments.exp20_realistic_pieces.generate_dataset import generate_dataset

    output_dir = Path(__file__).parent.parent.parent / "datasets" / "realistic_4x4_5k"

    gen_start = time.time()
    generate_dataset(
        source_dir=Path(__file__).parent.parent.parent / "datasets" / "puzzles",
        output_dir=output_dir,
        n_puzzles=5000,
        seed=42,
        padding=20,
        points_per_curve=20,
    )
    gen_time = time.time() - gen_start

    print()
    print(f"Generation completed in {gen_time / 60:.1f} minutes")
    print()

    # Phase 2: Training
    print("=" * 70)
    print("PHASE 2: TRAINING FOR 50 EPOCHS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
    print()

    from experiments.exp20_realistic_pieces.train import main as train_main

    train_start = time.time()
    train_main(
        epochs=50,
        n_train=4500,
        n_test=500,
        batch_size=64,
        backbone_lr=1e-4,
        head_lr=1e-3,
        dataset_root=output_dir,
    )
    train_time = time.time() - train_start

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Generation time: {gen_time / 60:.1f} minutes")
    print(f"Training time: {train_time / 60:.1f} minutes ({train_time / 50:.1f}s/epoch)")
    print(f"Total time: {(gen_time + train_time) / 60:.1f} minutes")
    print()
    print("Results saved to: experiments/exp20_realistic_pieces/outputs/")


if __name__ == "__main__":
    main()
