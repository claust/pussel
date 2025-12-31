#!/bin/bash
# Exp20: Full experiment with 5000 puzzles
# Runs dataset generation followed by 50-epoch training

set -e

cd /Users/claus/Repos/pussel/network
source ../venv/bin/activate

echo "========================================"
echo "EXP20: REALISTIC PIECES - FULL RUN"
echo "========================================"
echo "Started at: $(date)"
echo ""

# Phase 1: Dataset Generation
echo "========================================"
echo "PHASE 1: GENERATING 5000 PUZZLE DATASET"
echo "========================================"
echo "Start time: $(date)"
echo ""

python -m experiments.exp20_realistic_pieces.generate_dataset \
    --n-puzzles 5000 \
    --output-dir datasets/realistic_4x4_5k \
    --seed 42

echo ""
echo "Generation completed at: $(date)"
echo ""

# Phase 2: Training
echo "========================================"
echo "PHASE 2: TRAINING FOR 50 EPOCHS"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Update dataset paths for training
python -c "
import sys
sys.path.insert(0, '.')
from experiments.exp20_realistic_pieces.train import main

# Run training with 5000 puzzles dataset
import experiments.exp20_realistic_pieces.dataset as ds
ds.DEFAULT_DATASET_ROOT = ds.Path('datasets/realistic_4x4_5k')

main(
    epochs=50,
    n_train=4500,
    n_test=500,
    batch_size=64,
    backbone_lr=1e-4,
    head_lr=1e-3,
)
"

echo ""
echo "========================================"
echo "EXPERIMENT COMPLETE"
echo "========================================"
echo "Finished at: $(date)"
echo ""
echo "Results saved to: experiments/exp20_realistic_pieces/outputs/"
