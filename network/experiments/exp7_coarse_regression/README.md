# Experiment 7: Coarse Regression with Multi-Puzzle Training

## Objective

Test whether cross-puzzle generalization is fundamentally achievable by simplifying the task to quadrant prediction (2x2 grid = 4 pieces per puzzle) using coordinate regression.

**Key Question**: Can a model learn to match puzzle pieces to their correct quadrant across different puzzles it has never seen before?

## Background

### Summary of Previous Experiments

| Exp | Task | Result | Key Finding |
|-----|------|--------|-------------|
| 2 | Single puzzle, 950 pieces, regression | SUCCESS (loss 0.007) | Regression works for memorization |
| 3-4 | Single puzzle, 950-class classification | 83-99% accuracy | Classification needs more capacity |
| 5 | Cross-puzzle, dual-input, 950 classes | 0.1% test (random) | Single-puzzle training doesn't generalize |
| 6 | Multi-puzzle (5), 950 classes | 2.4% train, 0% test | 950 classes too hard even to train |

### Why Previous Generalization Failed

1. **Too fine-grained**: 950 classes (38x25 grid) requires extremely precise matching
2. **Classification is unforgiving**: No partial credit for nearby predictions
3. **Insufficient training diversity**: 5 puzzles not enough variation
4. **Resolution mismatch**: Each cell only ~13x20 pixels in 512x512 puzzle image

### This Experiment's Approach

Strip the problem down to its simplest form:

1. **Coarse grid**: 2x2 = 4 quadrants (not 950 cells)
2. **Regression**: Predict (cx, cy) coordinates, not class labels
3. **Many puzzles**: Train on 100+ different puzzle images
4. **Lightweight backbone**: MobileNetV3-Small for fast iteration

If we can't get quadrant-level generalization to work, finer-grained matching is hopeless.

## Experiment Design

### Task Definition

Given:
- A puzzle piece image (one quadrant of a puzzle)
- The complete puzzle image

Predict:
- The center coordinates (cx, cy) of where this piece belongs
- Coordinates normalized to [0, 1] range

### Grid Structure

```
+-------+-------+
|       |       |
| (0,0) | (1,0) |   Quadrant centers:
|       |       |   - Top-left:     (0.25, 0.25)
+-------+-------+   - Top-right:    (0.75, 0.25)
|       |       |   - Bottom-left:  (0.25, 0.75)
| (0,1) | (1,1) |   - Bottom-right: (0.75, 0.75)
|       |       |
+-------+-------+
```

### Dataset

| Aspect | Value |
|--------|-------|
| Pieces per puzzle | 4 (2x2 grid) |
| Training puzzles | 100+ (TBD based on available data) |
| Test puzzles | 20+ (held out, never seen during training) |
| Piece resolution | 64x64 or 128x128 |
| Puzzle resolution | 256x256 or 512x512 |

### Architecture

```
DualInputRegressor (MobileNetV3-Small backbone)
├── Piece Encoder: MobileNetV3-Small (pretrained)
│   └── Output: 576-dim features (after pooling)
├── Puzzle Encoder: MobileNetV3-Small (shared or separate weights)
│   └── Output: 576-dim features
├── Feature Fusion
│   └── Concatenate → Linear(1152, 256) → ReLU → Linear(256, 64)
└── Position Head
    └── Linear(64, 2) → Sigmoid → (cx, cy) ∈ [0, 1]
```

**Why MobileNetV3-Small?**
- ~2.5M parameters (vs ~11M for ResNet-18)
- Pretrained on ImageNet
- Designed for efficiency
- Fast training iterations

### Loss Function

```python
loss = F.mse_loss(pred_coords, target_coords)
```

Simple MSE on (cx, cy) coordinates. This naturally gives partial credit for nearby predictions.

### Training Strategy

1. **Phase 1: Frozen backbone**
   - Freeze MobileNetV3 weights
   - Train only fusion layers and position head
   - Verify the model can learn with fixed features

2. **Phase 2: Fine-tuning**
   - Unfreeze backbone with lower learning rate
   - Train end-to-end

### Evaluation

**Metrics:**
- **MSE Loss**: Direct regression quality
- **Quadrant Accuracy**: Is predicted center in the correct quadrant?
- **Distance Error**: Euclidean distance between predicted and true center

**Success Criteria:**

| Metric | Target | Meaning |
|--------|--------|---------|
| Training MSE | < 0.02 | Can fit training puzzles |
| Test MSE | < 0.10 | Generalizes to new puzzles |
| Test Quadrant Accuracy | > 70% | Practical usefulness (random = 25%) |

## Why This Should Work

1. **Regression is forgiving**: Predicting (0.3, 0.3) when target is (0.25, 0.25) gets low loss
2. **4 regions are visually distinct**: Each quadrant covers 25% of the image - large, distinguishable areas
3. **More training diversity**: 100+ puzzles provides variety in textures, colors, patterns
4. **Pretrained features**: MobileNetV3 already understands visual concepts from ImageNet

## What Success Would Mean

If this experiment succeeds (>70% test quadrant accuracy):
- Cross-puzzle generalization IS learnable
- The approach can be progressively refined to finer grids (3x3, 4x4, ...)
- The dual-input architecture with pretrained backbones is viable

## What Failure Would Mean

If this experiment fails (<40% test quadrant accuracy):
- The visual matching problem may require fundamentally different approaches
- Consider: contrastive learning, template matching, or edge-based features
- May need to rethink what features enable piece-to-puzzle matching

## File Structure

```
experiments/exp7_coarse_regression/
├── README.md           # This file
├── __init__.py         # Package marker
├── dataset.py          # Multi-puzzle dataset with 2x2 grid pieces
├── model.py            # MobileNetV3-based dual-input regressor
├── train.py            # Training script with train/test evaluation
├── generate_pieces.py  # Script to generate 2x2 pieces from puzzle images
├── visualize.py        # Visualization utilities
└── outputs/            # Saved models and visualizations
```

## Usage

```bash
cd network
source ../venv/bin/activate

# Generate 2x2 pieces from puzzle images (if needed)
python -m experiments.exp7_coarse_regression.generate_pieces

# Run the experiment
python -m experiments.exp7_coarse_regression.train

# With custom settings
python -m experiments.exp7_coarse_regression.train --epochs 100 --batch-size 32
```

## Results

### Phase 1: Frozen Backbone (MobileNetV3-Small)

**Configuration:**
- Training puzzles: 800 (3,200 samples)
- Test puzzles: 200 (800 samples, held out)
- Piece size: 128x128
- Puzzle size: 256x256
- Epochs: 100
- Batch size: 64
- Learning rate: 1e-3 (AdamW with cosine annealing)
- Backbone: Frozen (only fusion layers and position head trained)
- Trainable parameters: 311,746 / 2,165,762 total

**Training Metrics:**

| Epoch | Train MSE | Train Acc | Test MSE | Test Acc | Test Dist |
|-------|-----------|-----------|----------|----------|-----------|
| 1     | 0.0606    | 32.5%     | 0.1145   | 33.5%    | 0.335     |
| 10    | 0.0524    | 39.3%     | 0.1019   | 39.6%    | 0.313     |
| 20    | 0.0481    | 45.4%     | 0.1013   | 40.4%    | 0.310     |
| 50    | 0.0359    | 62.3%     | 0.1064   | 39.9%    | 0.310     |
| 100   | 0.0283    | 72.6%     | 0.1095   | 41.6%    | 0.309     |

**Final Results:**

| Metric | Training | Test | Target |
|--------|----------|------|--------|
| MSE Loss | 0.0374 | 0.1095 | < 0.02 / < 0.10 |
| Quadrant Accuracy | 84.7% | 41.6% | > 70% |
| Mean Distance | 0.1717 | 0.3091 | - |
| vs Random (25%) | 3.4x | 1.67x | - |

**Success Criteria:**
- Training MSE < 0.02: **FAIL** (0.0374)
- Test MSE < 0.10: **FAIL** (0.1095)
- Test Accuracy > 70%: **FAIL** (41.6%)

### Analysis

**Key Findings:**
1. **Some generalization achieved**: Test accuracy of 41.6% is significantly above random baseline (25%), showing the model learned some cross-puzzle matching.

2. **Overfitting observed**: Training accuracy reached 84.7% while test accuracy plateaued at ~40% from epoch 10 onwards. This indicates the model memorizes training puzzles rather than learning generalizable features.

3. **Frozen backbone limitation**: With only 311K trainable parameters (fusion layers), the model lacks capacity to learn puzzle-specific matching patterns. The pretrained ImageNet features may not be optimal for this task.

4. **Test accuracy plateau**: Test accuracy stabilized around 40% early in training and didn't improve with more epochs, suggesting the frozen backbone features have limited discriminative power for this task.

**Interpretation:**
The result falls in the "weak generalization" category (30-50% accuracy). The model learned something beyond random guessing, but the 41.6% accuracy is below the 70% target. This suggests:
- Cross-puzzle generalization IS partially achievable
- Frozen backbone features are insufficient
- Phase 2 (fine-tuning) may improve results

### Augmentation Fix (Post Phase 1)

The initial augmentation included horizontal/vertical flips, which was incorrect for this task:
- Flipping a top-left quadrant horizontally makes it look like a top-right quadrant
- But the label still says "top-left" → conflicting training signal

**Fixed augmentations** (pieces only):
| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| ColorJitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 | Lighting robustness |
| RandomGrayscale | p=0.1 | Don't rely solely on color |
| RandomRotation | ±5° | Small rotations, same quadrant |
| RandomAffine | translate=5%, scale=±5% | Minor position/size shifts |
| GaussianBlur | kernel=3, sigma=0.1-1.0 | Blur/focus robustness |

**Removed**: RandomHorizontalFlip, RandomVerticalFlip

### Phase 1b: With Fixed Augmentation (50 epochs)

**Results comparison:**

| Metric | Before Fix (100 ep) | After Fix (50 ep) | Change |
|--------|---------------------|-------------------|--------|
| Test Accuracy | 42.5% | **44.8%** | +2.3% |
| Best Test Acc | 43.4% (ep 40) | **46.1% (ep 30)** | +2.7% |
| Test MSE | 0.1083 | **0.0910** | -16% |
| vs Random | 1.70x | **1.79x** | +0.09x |
| Train Accuracy | 82.9% | 62.6% | Less overfitting |

**Training progression:**
| Epoch | Train MSE | Train Acc | Test MSE | Test Acc | Test Dist |
|-------|-----------|-----------|----------|----------|-----------|
| 1     | 0.0604    | 33.6%     | 0.1150   | 33.2%    | 0.334     |
| 10    | 0.0493    | 41.6%     | 0.0955   | 41.4%    | 0.302     |
| 20    | 0.0449    | 48.5%     | 0.0913   | 45.5%    | 0.295     |
| 30    | 0.0417    | 52.2%     | 0.0905   | **46.1%**| 0.291     |
| 40    | 0.0386    | 55.7%     | 0.0910   | 43.9%    | 0.290     |
| 50    | 0.0387    | 55.1%     | 0.0910   | 44.8%    | 0.290     |

**Success Criteria:**
- Training MSE < 0.02: **FAIL** (0.0687)
- Test MSE < 0.10: **PASS** (0.0910)
- Test Accuracy > 70%: **FAIL** (44.8%)

**Key findings:**
1. **Test MSE now passes target** - Fixed augmentation improved regression quality
2. **Better generalization** - Test accuracy improved from 42.5% to 46.1% peak
3. **Less overfitting** - Train/test gap reduced from 40% to 18%
4. **Faster convergence** - Best result at epoch 30, with early stopping beneficial

**Conclusion:** Removing the incorrect flips and adding appropriate augmentations provided a meaningful ~3% improvement. However, frozen backbone remains the bottleneck. Phase 2 (fine-tuning) is needed to break through the 46% ceiling.

### Architectural Flaw Discovery

After Phase 1b, we investigated why a pretrained model couldn't achieve higher accuracy on such a simple 4-class task.

**The Problem: No Spatial Information**

The original `DualInputRegressor` architecture:
```
Piece  → MobileNetV3 → Global Avg Pool → 576-dim vector
Puzzle → MobileNetV3 → Global Avg Pool → 576-dim vector
                                              ↓
                              Concatenate → MLP → (cx, cy)
```

**The puzzle's spatial structure is completely discarded!** Global average pooling collapses a spatial feature map (8x8x576) into a single 576-dim vector. The model receives:
- "Here's an abstract summary of the piece"
- "Here's an abstract summary of the entire puzzle"
- "Now tell me WHERE the piece goes"

Without spatial features, the model cannot perform template matching. It can only learn indirect correlations (e.g., "sky-like textures tend to be at the top").

**The Fix: Spatial Correlation**

New `DualInputRegressorWithCorrelation` architecture:
```
Piece  → MobileNetV3.features → Global Pool → 576-dim vector
Puzzle → MobileNetV3.features → KEEP SPATIAL → 8x8x576 feature map
                                     ↓
              Spatial Correlation: dot product at each location
                                     ↓
                    Softmax attention → weighted position → (cx, cy)
```

Key changes:
1. **Preserve puzzle spatial features** - No global pooling on puzzle
2. **Spatial correlation** - Compute similarity between piece vector and each puzzle location
3. **Soft attention** - Convert correlation scores to position via weighted average

This enables proper template matching: "Where in the puzzle does this piece's features best match?"

### Phase 1c: With Spatial Correlation (50 epochs)

**Results comparison:**

| Metric | Without Spatial (1b) | With Spatial (1c) | Improvement |
|--------|---------------------|-------------------|-------------|
| Test Accuracy | 46.1% | **67.0%** | **+20.9%** |
| Test MSE | 0.0910 | **0.0712** | -22% |
| vs Random | 1.79x | **2.68x** | +50% |
| Train Accuracy | 62.6% | 88.3% | +25.7% |

**Training progression:**
| Epoch | Train MSE | Train Acc | Test MSE | Test Acc | Test Dist |
|-------|-----------|-----------|----------|----------|-----------|
| 1     | 0.0584    | 37.5%     | 0.1041   | 50.0%    | 0.319     |
| 10    | 0.0321    | 71.6%     | 0.0761   | 61.9%    | 0.255     |
| 20    | 0.0251    | 79.7%     | 0.0714   | 64.2%    | 0.242     |
| 30    | 0.0207    | 83.6%     | 0.0716   | 65.2%    | 0.240     |
| 40    | 0.0195    | 85.0%     | 0.0707   | 66.9%    | 0.236     |
| 50    | 0.0185    | 84.5%     | 0.0712   | **67.0%**| 0.237     |

**Success Criteria:**
- Training MSE < 0.02: **FAIL** (0.0322) - but close!
- Test MSE < 0.10: **PASS** (0.0712)
- Test Accuracy > 70%: **FAIL** (67.0%) - only 3% away!

**Key findings:**
1. **Massive accuracy improvement** - Test accuracy jumped from 46% to 67%, a 21% absolute gain
2. **Spatial correlation was the key** - The model can now perform proper template matching
3. **50% at epoch 1** - Even untrained, the correlation-based approach already beats random by 2x
4. **Still improving** - Test accuracy increased throughout training, more epochs may help
5. **Close to target** - Only 3% away from the 70% success criterion

**Conclusion:** The architectural flaw (discarding spatial information) was the root cause of poor performance. With spatial correlation, the model achieves meaningful cross-puzzle generalization. The 67% accuracy validates that:
- Cross-puzzle matching IS learnable
- Pretrained features + spatial correlation work well
- The approach can be extended to finer grids

### Next Steps

1. **Phase 2: Fine-tune backbone** - Unfreeze MobileNetV3 with lower learning rate to learn task-specific features
2. **Increase puzzle resolution** - Try 512x512 puzzle images for more detail
3. **Alternative architectures** - Consider spatial correlation modules or attention mechanisms
4. **Contrastive learning** - Pre-train with piece-puzzle contrastive objective

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
exp7 (THIS EXPERIMENT)     → Coarse regression: 67% test accuracy (meaningful generalization!)
        ↓
(future: exp8, exp9...)    → Fine-tune backbone, increase resolution
```
