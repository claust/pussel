# Experiment 20: Realistic Puzzle Pieces with 4x4 Grid

## Objective

Test whether the model can generalize to realistically-shaped puzzle pieces
(Bezier curve tabs/blanks) instead of the square-cut rectangles used in
previous experiments (exp1-19).

## Hypothesis

Training with realistic piece shapes will:
1. Improve generalization to real-world puzzles
2. Potentially help rotation prediction (tab/blank positions provide cues)
3. Test robustness to irregular piece boundaries

## Key Changes from Exp18

| Aspect | Exp 18 | Exp 20 |
|--------|--------|--------|
| Piece shape | Square rectangles | Bezier curve tabs/blanks |
| Grid size | 3x3 (9 cells) | 4x4 (16 cells) |
| Piece source | Extracted at runtime | Pre-generated to disk |
| Background | Crop from puzzle | Black (0,0,0) |
| Training puzzles | 20,000 | 500 (pilot) |
| Random baseline | 11.1% | 6.25% |

## Architecture

Same dual-backbone architecture as exp18:
- **Backbone**: ShuffleNetV2_x0.5 (dual - one for piece, one for puzzle)
- **Position output**: Continuous (x, y) coordinates via spatial correlation
- **Rotation output**: 4-class classification via rotation correlation

The model uses **coordinate regression** (MSE loss on x, y), not cell
classification. Cell accuracy is an evaluation metric derived from positions.

## Dataset

### Generation

Run the dataset generator to create realistic pieces:

```bash
cd network
source ../venv/bin/activate
python -m experiments.exp20_realistic_pieces.generate_dataset --n-puzzles 500
```

This will:
1. Load puzzle images from `datasets/puzzles/`
2. Generate 4x4 edge grids with Bezier curves using `puzzle_shapes` library
3. Cut 16 pieces per puzzle with realistic interlocking edges
4. Fill transparency with black background
5. Apply random rotation
6. Save to `datasets/realistic_4x4/`

### File Format

Pieces are named with center coordinates:
```
puzzle_00001_x0.125_y0.125_rot90.png
```
- `x0.125`, `y0.125` = normalized center coordinates
- `rot90` = rotation applied (0, 90, 180, 270)

### Disk Usage

~400MB for 500 puzzles (16 pieces x ~50KB each)

## Training

### Pilot Configuration

```bash
cd network
source ../venv/bin/activate
python -m experiments.exp20_realistic_pieces.train --epochs 50 --n-train 500 --n-test 50
```

### Success Criteria (Pilot)

| Metric | Random Baseline | Target |
|--------|-----------------|--------|
| Cell accuracy | 6.25% (1/16) | > 50% |
| Rotation accuracy | 25% | > 70% |
| Train-test gap | - | < 15% |

## Files

```
exp20_realistic_pieces/
├── __init__.py           # Package exports
├── README.md             # This file
├── generate_dataset.py   # Create realistic pieces from source puzzles
├── dataset.py            # DataLoader for pre-generated pieces
├── model.py              # FastBackboneModel (adapted for 4x4)
├── train.py              # Training script
├── visualize.py          # Visualization utilities
└── outputs/              # Results (created during training)
    ├── checkpoint_best.pt
    ├── checkpoint_last.pt
    ├── results.json
    ├── test_predictions.png
    └── training_curves.png
```

## Running the Experiment

1. **Generate dataset** (one-time):
   ```bash
   python -m experiments.exp20_realistic_pieces.generate_dataset --n-puzzles 500
   ```

2. **Train model**:
   ```bash
   python -m experiments.exp20_realistic_pieces.train
   ```

3. **View results**: Check `outputs/` directory for visualizations and metrics.

## Dependencies

- `puzzle_shapes` library (in `shared/puzzle_shapes/`)
- PyTorch with MPS/CUDA support
- torchvision for backbone models
- PIL for image processing
- matplotlib for visualizations
