# Experiment 13: Rotation Correlation with 5K Puzzles

## Objective

Test whether **scaling up the training data** from 800 to 4499 puzzles improves the rotation correlation architecture from exp12. Specifically:

1. Does more diverse training data fix the **position regression** observed in exp12?
2. Does rotation accuracy continue to improve with more data?

**Key Question**: Can more training data alone fix the position regression without architectural changes?

## Background

### Exp12 Results

Exp12 introduced the rotation correlation approach, achieving:
- **86.9% rotation accuracy** (major breakthrough, +27% over exp11)
- **66.5% quadrant accuracy** (regressed from exp11's 72.9%)

The rotation correlation solved the rotation problem, but position regressed. One hypothesis: the smaller dataset (800 puzzles) didn't provide enough diversity for joint position-rotation learning.

### This Experiment

Rather than making architectural changes, we test whether simply **scaling up the data** can fix the position regression:
- Training puzzles: 800 → **4499** (5.6x increase)
- Test puzzles: 200 → **500** (2.5x increase)
- Architecture: **Identical** to exp12

## Experiment Design

### Model Architecture

Same as exp12:

| Component | Details |
|-----------|---------|
| Backbone | MobileNetV3-Small (unfrozen, LR=1e-4) |
| Position Head | Spatial Correlation |
| Rotation Head | Rotation Correlation |
| Total Parameters | 2,521,444 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training puzzles | **4499** (was 800 in exp12) |
| Test puzzles | **500** (was 200 in exp12) |
| Samples per epoch | 17,996 (4 per puzzle) |
| Test samples | 8,000 (16 per puzzle) |
| Input Size | Piece: 128x128, Puzzle: 256x256 |
| Loss | MSE (position) + CrossEntropy (rotation) |
| Loss weights | Position: 1.0, Rotation: 1.0 |
| Optimizer | AdamW |
| Batch Size | 64 |
| Epochs | 100 |
| Position dropout | 0.1 |
| Rotation dropout | 0.2 |

### Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Test Position Acc** | **> 80%** | Fix the regression from exp12 (66.5%) |
| **Test Rotation Acc** | **> 90%** | Improve beyond exp12 (86.9%) |
| 0° vs 180° Confusion | < 5% | Maintain low confusion from exp12 |

## Results

### Training Metrics Over Time

| Epoch | Train Pos Loss | Train Rot Loss | Train Quad | Train Rot | Test Quad | Test Rot |
|-------|----------------|----------------|------------|-----------|-----------|----------|
| 1     | 0.0532         | 1.1219         | 44.6%      | 46.0%     | 53.4%     | 63.0%    |
| 10    | 0.0307         | 0.3149         | 70.4%      | 88.1%     | 70.0%     | 83.5%    |
| 20    | 0.0229         | 0.1818         | 78.2%      | 93.3%     | 77.0%     | 87.7%    |
| 30    | 0.0181         | 0.1223         | 83.0%      | 95.5%     | 79.7%     | 88.3%    |
| 40    | 0.0152         | 0.0937         | 85.8%      | 96.7%     | 81.7%     | 90.7%    |
| 50    | 0.0127         | 0.0691         | 88.0%      | 97.5%     | 83.9%     | 90.5%    |
| 60    | 0.0108         | 0.0497         | 89.8%      | 98.3%     | 85.1%     | 92.4%    |
| 70    | 0.0095         | 0.0399         | 91.1%      | 98.6%     | 85.7%     | 92.2%    |
| 80    | 0.0086         | 0.0332         | 91.8%      | 98.8%     | **86.3%** | **93.2%** |
| 90    | 0.0083         | 0.0303         | 92.4%      | 98.9%     | 86.5%     | 92.6%    |
| 100   | 0.0082         | 0.0257         | 92.3%      | 99.0%     | 86.3%     | 92.8%    |

### Training Curves

![Training Curves](outputs/training_curves.png)

Key observations:
- Both position and rotation improve steadily throughout training
- Test accuracy continues improving until epoch ~80
- Train-test gap is much smaller than exp12 (better generalization)

### Final Results

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| Position MSE | 0.0083 | 0.0274 | 3.3x |
| Quadrant Accuracy | 96.9% | **86.3%** | -10.6% |
| Rotation Accuracy | 97.7% | **92.8%** | -4.9% |
| Mean Distance | 0.058 | 0.102 | 1.8x |

### Comparison with Previous Experiments

| Metric | Exp10 | Exp11 | Exp12 | **Exp13** | Change vs Exp12 |
|--------|-------|-------|-------|-----------|-----------------|
| Test Quadrant Acc | 78.4% | 72.9% | 66.5% | **86.3%** | **+19.8%** |
| Test Rotation Acc | 61.4% | 59.8% | 86.9% | **92.8%** | **+5.9%** |
| Test MSE | 0.046 | 0.054 | 0.069 | **0.027** | **2.5x better** |
| Train-Test Rot Gap | 36.6% | 37.9% | 12.4% | **4.9%** | **Much smaller** |

### Success Criteria Check

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Position target (>80%) | 80% | 86.3% | **PASS** |
| Rotation target (>90%) | 90% | 92.8% | **PASS** |
| 0°↔180° confusion (<5%) | 5% | 0.8-1.6% | **PASS** |

### Test Predictions Visualization

![Test Predictions](outputs/test_predictions.png)

### Rotation Confusion Matrix

```
         Pred: 0°   90°  180°  270°
True   0°: 1893   44   16   47
True  90°:   53 1827   73   47
True 180°:   31   48 1865   56
True 270°:   64   45   51 1840
```

**Confusion rates (dramatically improved from exp12):**
- 0° confused as 180°: **0.8%** (was 7.1% in exp12)
- 180° confused as 0°: **1.6%** (was 9.4% in exp12)
- 90° confused as 270°: **2.4%** (was 9.1% in exp12)
- 270° confused as 90°: **2.2%** (was 6.6% in exp12)

## Analysis

### Hypothesis Validation

1. **Did more data fix the position regression?** **YES!**
   - Position accuracy jumped from 66.5% to 86.3% (+19.8%)
   - This is now higher than exp9 (93%) minus the expected cost of joint prediction
   - The position regression was primarily a data diversity issue

2. **Did rotation accuracy improve?** **YES!**
   - Rotation accuracy improved from 86.9% to 92.8% (+5.9%)
   - Exceeded the 90% stretch target
   - Train-test gap reduced from 12.4% to 4.9%

3. **Is generalization improved?** **YES!**
   - Train-test gaps are much smaller across all metrics
   - The model is less likely to memorize puzzle-specific patterns
   - More diverse training examples = better feature learning

### Why More Data Helped

1. **Position Learning**: With 800 puzzles, the model saw limited texture diversity. Position prediction requires learning generalizable visual features, not puzzle-specific patterns. 4499 puzzles provide 5.6x more texture variety.

2. **Rotation Correlation**: The rotation head compares piece to puzzle features. More diverse puzzles teach the model to focus on structural matches (edges, gradients) rather than memorizing specific texture combinations.

3. **Reduced Overfitting**: Train-test gaps shrunk dramatically:
   - Position: 25.9% gap (exp12) → 10.6% gap (exp13)
   - Rotation: 12.4% gap (exp12) → 4.9% gap (exp13)

### Key Insight

**Data quantity matters as much as architecture.** The rotation correlation architecture (exp12) was correct, but 800 puzzles weren't enough to train it properly. Scaling to 4499 puzzles unlocked the architecture's full potential.

## Conclusion

**The experiment is a SUCCESS.** All targets met or exceeded.

**Key achievements:**
- **+19.8% position accuracy** (66.5% → 86.3%) - regression completely fixed
- **+5.9% rotation accuracy** (86.9% → 92.8%) - exceeded 90% target
- **Rotation confusion nearly eliminated** (0.8-1.6% vs 7-9% in exp12)
- **Much better generalization** (smaller train-test gaps)

**Key insight:**
The rotation correlation architecture from exp12 was correct. The position regression was caused by **insufficient training data**, not an architectural flaw. Scaling from 800 to 4499 puzzles fixed the regression without any architectural changes.

**Verdict: SUCCESS**
- Position: **FIXED** (86.3% vs exp12's 66.5%)
- Rotation: **EXCELLENT** (92.8% vs exp12's 86.9%)
- Architecture: **VALIDATED** (rotation correlation + more data = success)

## Next Steps

### Priority 1: Fine-tune Further
1. **Increase epochs**: Training was still improving at epoch 100
2. **Learning rate tuning**: Try lower LR for longer training
3. **Data augmentation**: Add more aggressive augmentations

### Priority 2: Finer Grid
1. **3x3 grid**: Move from 4-class to 9-class position prediction
2. **4x4 grid**: 16-class position prediction
3. **Continuous regression**: Direct (x, y) coordinate prediction

### Priority 3: Real Puzzle Pieces
1. **Actual piece shapes**: Test on real puzzle piece images (not quadrant crops)
2. **Edge detection**: Incorporate piece edge patterns
3. **Shape matching**: Add piece contour features

## Relationship to Previous Experiments

```
exp1-4 (baseline/classification)  -> Verified training pipeline
        |
exp5-6 (generalization)           -> 950 classes: fails completely
        |
exp7 (coarse + correlation)       -> 67% test (spatial correlation breakthrough)
        |
exp8 (high resolution)            -> 59% (resolution not the answer)
        |
exp9 (fine-tuning)                -> 93% position-only (FINE-TUNING SUCCESS)
        |
exp10 (add rotation)              -> 78% quad, 61% rot (rotation ignores puzzle)
        |
exp11 (spatial rotation)          -> 73% quad, 60% rot (still ignores puzzle)
        |
exp12 (rotation correlation)      -> 67% quad, 87% rot (ROTATION SOLVED, position regressed)
        |
exp13 (THIS EXPERIMENT)           -> 86% quad, 93% rot (MORE DATA = BOTH SOLVED!)
```

## File Structure

```
experiments/exp13_rotation_correlation_5k/
├── README.md           # This file
├── __init__.py         # Package marker
├── dataset.py          # Same as exp12 (random rotation sampling)
├── model.py            # DualInputRegressorWithRotationCorrelation (same as exp12)
├── train.py            # Training script (updated defaults for 5K puzzles)
├── visualize.py        # Visualization utilities
└── outputs/            # Saved models and visualizations
    ├── model.pt                   # Final model
    ├── model_best.pt              # Best checkpoint (epoch 80)
    ├── checkpoint_latest.pt       # Latest checkpoint
    ├── test_predictions.png       # Confusion matrices
    ├── training_curves.png        # Loss and accuracy curves
    └── experiment_results.json    # Full results
```

## Usage

```bash
cd network
source ../venv/bin/activate

# Run with default settings (4499 train, 500 test)
python -m experiments.exp13_rotation_correlation_5k.train

# Custom parameters
python -m experiments.exp13_rotation_correlation_5k.train \
    --epochs 100 \
    --batch-size 64 \
    --n-train 4499 \
    --n-test 500
```
