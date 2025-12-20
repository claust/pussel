# Experiment 8: High Resolution Coarse Regression (512x512 puzzle)

## Objective

Test whether increasing the puzzle resolution from 256x256 to 512x512 improves cross-puzzle generalization for 2x2 quadrant prediction.

**Key Question**: Does providing more visual detail in the puzzle image help the spatial correlation mechanism make better template matching decisions?

## Background

### Experiment 7 Results

Exp7 established that cross-puzzle generalization IS achievable with the right architecture:

| Metric | Exp7 (256x256) | Target |
|--------|----------------|--------|
| Test Accuracy | **67.0%** | >70% |
| Test MSE | 0.0712 | <0.10 |
| vs Random | 2.68x | - |

**Key breakthrough in exp7**: The `DualInputRegressorWithCorrelation` architecture preserves spatial information from the puzzle, enabling proper template matching. This was a massive improvement over the original architecture which discarded spatial information via global average pooling.

### Hypothesis for Exp8

With 256x256 puzzle images, MobileNetV3-Small produces an 8x8 spatial feature map. Each spatial location covers:
- 256/8 = 32 pixels in the original puzzle image
- For a 2x2 quadrant grid, each quadrant spans 4x4 = 16 feature locations

With 512x512 puzzle images, we get a 16x16 spatial feature map:
- 512/16 = 32 pixels per location (same ratio)
- For a 2x2 quadrant grid, each quadrant spans 8x8 = 64 feature locations

**Hypothesis**: More spatial locations in the feature map should provide finer-grained correlation, potentially improving accuracy.

## Experiment Design

### Key Change from Exp7

| Aspect | Exp7 | Exp8 (This Experiment) |
|--------|------|------------------------|
| Puzzle resolution | 256x256 | **512x512** |
| Piece resolution | 128x128 | 128x128 (unchanged) |
| Spatial feature map | 8x8 = 64 locations | **16x16 = 256 locations** |
| Pixels per quadrant | 128x128 | 256x256 |
| Training puzzles | 800 | 800 |
| Test puzzles | 200 | 200 |

### Architecture

Same `DualInputRegressorWithCorrelation` from exp7:

```
DualInputRegressorWithCorrelation
├── Piece Encoder: MobileNetV3-Small (pretrained, frozen)
│   └── Input: 128x128 → Output: 576-dim vector (after pooling)
├── Puzzle Encoder: MobileNetV3-Small (pretrained, frozen)
│   └── Input: 512x512 → Output: 576-dim × 16×16 spatial map
├── Spatial Correlation Module
│   ├── Projects piece (576→128) and puzzle (576→128)
│   ├── Computes dot product at each 16×16 location
│   ├── Softmax attention over spatial locations
│   └── Weighted average → expected position
└── Refinement Head
    └── Small MLP to adjust correlation-based prediction
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training puzzles | 800 |
| Test puzzles | 200 (held out) |
| Piece size | 128x128 |
| Puzzle size | **512x512** |
| Epochs | 50 |
| Batch size | 32 (smaller than exp7 due to memory) |
| Learning rate | 1e-3 (AdamW with cosine annealing) |
| Backbone | Frozen (Phase 1) |

### Success Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| Training MSE | < 0.02 | Can fit training puzzles |
| Test MSE | < 0.10 | Generalizes to new puzzles |
| Test Quadrant Accuracy | > 70% | Better than exp7's 67% |

## Why This Might Help

1. **Finer spatial resolution**: 16x16 vs 8x8 feature map means 4x more locations for correlation
2. **More detail per quadrant**: Each quadrant covers 256x256 pixels vs 128x128
3. **Better texture discrimination**: Higher resolution preserves more visual details

## Why This Might NOT Help

1. **Same feature ratio**: Each feature location still covers 32x32 pixels
2. **More noise**: More spatial locations could spread attention over irrelevant areas
3. **Memory overhead**: Larger tensors may require smaller batch sizes
4. **Frozen backbone**: Features are still from ImageNet, not task-specific

## File Structure

```
experiments/exp8_high_res_correlation/
├── README.md           # This file
├── __init__.py         # Package marker
├── dataset.py          # QuadrantDataset with 512x512 puzzle support
├── model.py            # DualInputRegressorWithCorrelation
├── train.py            # Training script (puzzle_size=512 default)
├── visualize.py        # Visualization utilities
├── generate_pieces.py  # Piece generation script
└── outputs/            # Saved models and visualizations
```

## Usage

```bash
cd network
source ../venv/bin/activate

# Run the experiment with default settings (512x512 puzzle)
python -m experiments.exp8_high_res_correlation.train

# Custom settings
python -m experiments.exp8_high_res_correlation.train --epochs 100 --puzzle-size 512

# Compare with exp7 settings (256x256)
python -m experiments.exp8_high_res_correlation.train --puzzle-size 256
```

## Results

### Configuration

| Parameter | Value |
|-----------|-------|
| Training puzzles | 800 (3,200 samples) |
| Test puzzles | 200 (800 samples, held out) |
| Piece size | 128x128 |
| Puzzle size | **512x512** |
| Epochs | 50 |
| Batch size | 32 |
| Learning rate | 1e-3 (AdamW with cosine annealing) |
| Backbone | Frozen (Phase 1) |
| Trainable parameters | 147,875 / 2,001,891 total |
| Training time | 1754 seconds (~29 minutes) |

### Training Metrics

| Epoch | Train MSE | Train Acc | Test MSE | Test Acc | Test Dist |
|-------|-----------|-----------|----------|----------|-----------|
| 1     | 0.0586    | 36.3%     | 0.1107   | 39.9%    | 0.327     |
| 10    | 0.0381    | 63.4%     | 0.0872   | 54.8%    | 0.280     |
| 20    | 0.0317    | 70.1%     | 0.0839   | 55.9%    | 0.270     |
| 30    | 0.0273    | 74.7%     | 0.0806   | 57.9%    | 0.262     |
| 40    | 0.0250    | 76.8%     | 0.0793   | 58.2%    | 0.260     |
| 50    | 0.0247    | 77.3%     | 0.0789   | 59.0%    | 0.259     |

### Final Results

| Metric | Exp7 (256px) | Exp8 (512px) | Change |
|--------|--------------|--------------|--------|
| Test Accuracy | **67.0%** | 59.0% | **-8.0%** |
| Test MSE | 0.0712 | 0.0789 | +10.8% |
| vs Random | 2.68x | 2.36x | -12% |
| Train Accuracy | 84.5% | 81.0% | -3.5% |

### Success Criteria

- Training MSE < 0.02: **FAIL** (0.0438)
- Test MSE < 0.10: **PASS** (0.0789)
- Test Accuracy > 70%: **FAIL** (59.0%)

## Conclusions

### Experiment Outcome: NEGATIVE RESULT

**Higher resolution (512x512) actually DECREASED test accuracy by 8 percentage points** compared to exp7's 256x256 resolution (59% vs 67%).

### Analysis: Why Did Higher Resolution Hurt?

Several factors likely contributed to the performance decrease:

1. **Attention diffusion**: With 16x16 = 256 spatial locations (vs 8x8 = 64), the softmax attention is spread over 4x more positions. This makes it harder for the model to focus on the correct quadrant, leading to more diffuse and less confident predictions.

2. **Feature resolution mismatch**: The piece encoder still produces a single 576-dim vector (via global pooling), but this is now correlated with 256 puzzle locations instead of 64. The piece representation may lack the capacity to discriminate among so many locations.

3. **Frozen ImageNet features at wrong scale**: The MobileNetV3-Small backbone was trained on 224x224 ImageNet images. At 512x512, the network processes the puzzle at a very different scale than its pretraining, potentially producing less discriminative features.

4. **Overfitting gap increased**: The train/test accuracy gap widened from ~17% (exp7) to ~22% (exp8), suggesting the model is memorizing training puzzles more without better generalization.

5. **Quadrant-level task doesn't need fine detail**: For a 2x2 quadrant task, each quadrant covers 25% of the image. The coarse 8x8 feature map (4x4 per quadrant) may actually be sufficient, and more detail adds noise rather than signal.

### Key Insight

**More resolution is not always better.** For the coarse quadrant prediction task:
- 256x256 (8x8 features) provides the right level of abstraction
- 512x512 (16x16 features) provides too much detail, making the correlation noisier

This suggests the optimal resolution depends on the task granularity. Finer grids (3x3, 4x4) might benefit from higher resolution, but for 2x2 quadrants, 256x256 is sufficient.

## Next Steps

Given that higher resolution hurt performance:

1. **Return to 256x256 resolution** - Exp7's configuration remains the best baseline

2. **Focus on fine-tuning backbone (Phase 2)** - Instead of input resolution, improve the feature quality:
   - Unfreeze MobileNetV3 with lower learning rate
   - Learn task-specific features instead of using frozen ImageNet features

3. **Test finer grids with appropriate resolution**:
   - 3x3 grid: May need 256x256 or 384x384
   - 4x4 grid: May need 384x384 or 512x512
   - Match resolution to task granularity

4. **Consider attention mechanisms** - Multi-head attention might help focus correlation more effectively

5. **Piece encoder improvements** - The piece representation may be the bottleneck; consider spatial features from piece too

## Relationship to Previous Experiments

```
exp1 (baseline sanity)     → Verified training works
        ↓
exp2 (single puzzle reg)   → Verified regression works for memorization
        ↓
exp3-4 (classification)    → Classification needs more capacity
        ↓
exp5-6 (generalization)    → 950 classes: generalization fails completely
        ↓
exp7 (coarse + correlation)→ 67% test accuracy (spatial correlation breakthrough!)
        ↓
exp8 (THIS EXPERIMENT)     → 512x512 resolution: 59% (WORSE - resolution not the answer)
        ↓
(future: exp9...)          → Fine-tune backbone at 256x256 (most promising direction)
```

## Summary

| Experiment | Resolution | Test Accuracy | Key Finding |
|------------|------------|---------------|-------------|
| exp7       | 256x256    | **67.0%**     | Spatial correlation works! |
| exp8       | 512x512    | 59.0%         | Higher resolution hurts (attention diffusion) |

**Recommendation**: For exp9, return to 256x256 and focus on fine-tuning the backbone to learn task-specific features.
