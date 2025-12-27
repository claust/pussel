# Experiment 9: Fine-tune Backbone (Phase 2)

## Objective

Fine-tune the MobileNetV3-Small backbone to learn task-specific features for cross-puzzle generalization. This is Phase 2 of the coarse regression approach, building on exp7's frozen backbone results.

**Key Question**: Can learning task-specific features push test accuracy beyond the 67% achieved with frozen ImageNet features?

## Background

### Previous Results Summary

| Experiment | Configuration | Test Accuracy | Key Finding |
|------------|---------------|---------------|-------------|
| exp7 (Phase 1) | 256x256, frozen backbone | **67.0%** | Spatial correlation works |
| exp8 | 512x512, frozen backbone | 59.0% | Higher resolution hurts (attention diffusion) |

### Why Frozen Backbone Hit a Ceiling

Exp7 demonstrated that:
1. Spatial correlation is essential for template matching
2. Frozen ImageNet features provide useful but limited representations
3. Test accuracy plateaued at 67%, just short of the 70% target

**Hypothesis**: The frozen backbone was the bottleneck, not the resolution. ImageNet features aren't optimized for puzzle piece matching. By fine-tuning, the backbone can learn:
- Texture matching patterns specific to puzzles
- Edge and boundary features important for piece localization
- Color/pattern correlations between pieces and full puzzles

## Experiment Design

### Key Changes from Exp7

| Aspect | Exp7 (Phase 1) | Exp9 (Phase 2) |
|--------|----------------|----------------|
| Backbone | **Frozen** | **Unfrozen** (fine-tuned) |
| Backbone LR | N/A | 1e-4 (10x lower than heads) |
| Head LR | 1e-3 | 1e-3 (same) |
| Trainable params | ~150K | ~2M (full model) |
| Regularization | Minimal | Dropout + weight decay |
| Puzzle size | 256x256 | 256x256 (same) |

### Architecture

Same `DualInputRegressorWithCorrelation` from exp7, but with backbone unfrozen:

```
DualInputRegressorWithCorrelation
├── Piece Encoder: MobileNetV3-Small (UNFROZEN, LR=1e-4)
│   └── Input: 128x128 → Output: 576-dim vector
├── Puzzle Encoder: MobileNetV3-Small (UNFROZEN, LR=1e-4)
│   └── Input: 256x256 → Output: 576-dim × 8×8 spatial map
├── Spatial Correlation Module (LR=1e-3)
│   ├── Piece projection with dropout
│   ├── Puzzle projection with dropout
│   ├── Correlation + softmax attention
│   └── Weighted position output
└── Refinement Head (LR=1e-3)
    └── Small MLP with dropout
```

### Differential Learning Rates

To prevent catastrophic forgetting of pretrained features while still allowing task-specific adaptation:

| Parameter Group | Learning Rate | Rationale |
|-----------------|---------------|-----------|
| Backbone (both encoders) | 1e-4 | Slow adaptation, preserve pretrained features |
| Correlation module | 1e-3 | Fast learning for task-specific matching |
| Refinement head | 1e-3 | Fast learning for position refinement |

### Gradual Unfreezing (Optional)

An alternative strategy that unfreezes layers progressively:

| Phase | Epoch | Layers Unfrozen | Rationale |
|-------|-------|-----------------|-----------|
| Start | 1 | [10, 11, 12] | Final conv blocks (most task-specific) |
| Phase 2 | 20 | [7, 8, 9] | Middle blocks |
| (optional) | 40 | [4, 5, 6] | Earlier blocks |
| Never | - | [0, 1, 2, 3] | Keep frozen (low-level features) |

This allows the model to first adapt high-level features before touching lower-level ones.

### Regularization

To prevent overfitting with more trainable parameters:

| Technique | Value | Purpose |
|-----------|-------|---------|
| Dropout | 0.1 | In correlation and refinement heads |
| Weight decay | 1e-4 | L2 regularization on all parameters |
| Data augmentation | Same as exp7 | Color jitter, small rotations, blur |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Training puzzles | 800 (3,200 samples) |
| Test puzzles | 200 (800 samples) |
| Piece size | 128x128 |
| Puzzle size | 256x256 |
| Epochs | 100 |
| Batch size | 64 |
| Backbone LR | 1e-4 |
| Head LR | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| LR schedule | Cosine annealing |

### Success Criteria

| Target | Value | Meaning |
|--------|-------|---------|
| Beat exp7 | > 67% | Fine-tuning helps |
| Primary target | > 70% | Practical usefulness |
| Stretch goal | > 75% | Strong generalization |

## Why This Should Work

1. **Task-specific features**: ImageNet features are general-purpose. Fine-tuning allows learning puzzle-specific texture and pattern matching.

2. **Differential LRs prevent catastrophic forgetting**: Lower backbone LR means we adapt pretrained features rather than overwriting them.

3. **More capacity**: With ~2M trainable parameters instead of ~150K, the model can learn more complex matching patterns.

4. **Same proven architecture**: Exp7 validated that spatial correlation works. We're only improving the feature quality.

## Why This Might NOT Work

1. **Overfitting risk**: More trainable parameters could lead to memorizing training puzzles instead of learning general matching.

2. **Pretrained features might be optimal**: ImageNet pretraining might already capture the best visual features, and fine-tuning could hurt generalization.

3. **Need more data**: Fine-tuning typically requires more data than training from scratch. 800 puzzles might not be enough.

## File Structure

```
experiments/exp9_finetune_backbone/
├── README.md           # This file
├── __init__.py         # Package marker
├── dataset.py          # QuadrantDataset (same as exp7)
├── model.py            # DualInputRegressorWithCorrelation + fine-tuning support
├── train.py            # Training script with differential LRs
├── visualize.py        # Visualization utilities (same as exp7)
├── generate_pieces.py  # Piece generation script (same as exp7)
└── outputs/            # Saved models and visualizations
```

## Usage

```bash
cd network
source ../venv/bin/activate

# Run with default settings (full fine-tuning)
python -m experiments.exp9_finetune_backbone.train

# Custom backbone learning rate
python -m experiments.exp9_finetune_backbone.train --backbone-lr 1e-5

# Use gradual unfreezing
python -m experiments.exp9_finetune_backbone.train --gradual-unfreeze

# More regularization
python -m experiments.exp9_finetune_backbone.train --dropout 0.2 --weight-decay 1e-3

# All options
python -m experiments.exp9_finetune_backbone.train \
    --epochs 100 \
    --batch-size 64 \
    --backbone-lr 1e-4 \
    --head-lr 1e-3 \
    --dropout 0.1 \
    --weight-decay 1e-4 \
    --gradual-unfreeze
```

## Results

### Configuration

| Parameter | Value |
|-----------|-------|
| Training puzzles | 800 (3,200 samples) |
| Test puzzles | 200 (800 samples) |
| Piece size | 128x128 |
| Puzzle size | 256x256 |
| Epochs | 100 |
| Batch size | 64 |
| Backbone LR | 1e-4 |
| Head LR | 1e-3 |
| Weight decay | 1e-4 |
| Dropout | 0.1 |
| Total parameters | 2,001,891 |
| Trainable parameters | 2,001,891 (all unfrozen) |
| Training time | 3,095 seconds (~52 minutes) |

### Training Metrics

| Epoch | Train MSE | Train Acc | Test MSE | Test Acc | Test Dist |
|-------|-----------|-----------|----------|----------|-----------|
| 1     | 0.0585    | 37.6%     | 0.1017   | 46.6%    | 0.312     |
| 10    | 0.0197    | 83.8%     | 0.0533   | 72.6%    | 0.195     |
| 20    | 0.0094    | 93.7%     | 0.0346   | 83.4%    | 0.138     |
| 30    | 0.0057    | 96.2%     | 0.0264   | 86.4%    | 0.108     |
| 40    | 0.0038    | 97.2%     | 0.0222   | 89.8%    | 0.089     |
| 50    | 0.0029    | 98.2%     | 0.0194   | 91.5%    | 0.073     |
| 60    | 0.0020    | 98.2%     | 0.0181   | 91.4%    | 0.066     |
| 70    | 0.0016    | 98.8%     | 0.0178   | 91.9%    | 0.061     |
| 80    | 0.0016    | 98.2%     | 0.0163   | 92.6%    | 0.057     |
| 90    | 0.0012    | 99.0%     | 0.0167   | 92.9%    | 0.057     |
| 100   | 0.0012    | 99.0%     | 0.0167   | **93.0%**| 0.057     |

### Final Results

| Metric | Exp7 (frozen) | Exp9 (fine-tuned) | Change |
|--------|---------------|-------------------|--------|
| Test Accuracy | 67.0% | **93.0%** | **+26.0%** |
| Test MSE | 0.0712 | 0.0167 | -77% |
| Mean Distance | 0.237 | 0.057 | -76% |
| vs Random | 2.68x | **3.72x** | +39% |
| Train Accuracy | 84.5% | 99.3% | +14.8% |

### Success Criteria

- Beat exp7 (>67%): **PASS** (93.0% vs 67.0%)
- Target (>70%): **PASS** (93.0%)
- Stretch (>75%): **PASS** (93.0%)

## Analysis

### Key Findings

1. **Massive improvement**: Test accuracy jumped from 67% (frozen backbone) to **93%** (fine-tuned), a **+26 percentage point** improvement. This conclusively demonstrates that task-specific features are essential for cross-puzzle matching.

2. **Rapid learning**: The model exceeded exp7's 67% test accuracy by epoch 10 (72.6%), showing that fine-tuning quickly adapts the pretrained features. The 70% target was surpassed within the first 10 epochs.

3. **Excellent generalization**: Despite having ~13x more trainable parameters (2M vs 150K), the model generalized well. The train/test accuracy gap is ~6% (99.3% vs 93.0%), which is reasonable for this task.

4. **Consistent improvement**: Test accuracy improved steadily throughout training:
   - Epoch 10: 72.6%
   - Epoch 30: 86.4%
   - Epoch 50: 91.5%
   - Epoch 100: 93.0%

5. **Low error magnitude**: Mean distance error dropped from 0.237 (exp7) to 0.057 (exp9). Even when wrong, predictions are much closer to the correct location.

### Why Fine-tuning Worked So Well

1. **Task-specific feature learning**: The backbone learned to extract features optimized for puzzle piece matching rather than ImageNet classification. This includes texture matching, edge detection at piece boundaries, and color/pattern correlation.

2. **Differential learning rates helped**: The 10x lower backbone LR (1e-4 vs 1e-3) prevented catastrophic forgetting of useful pretrained features while still allowing adaptation.

3. **Spatial correlation + fine-tuning synergy**: The spatial correlation architecture from exp7 provided the right inductive bias, and fine-tuning provided features that work better with this architecture.

4. **Sufficient regularization**: Dropout (0.1) and weight decay (1e-4) prevented overfitting despite having 2M trainable parameters.

### Comparison with Previous Experiments

| Experiment | Test Accuracy | Key Insight |
|------------|---------------|-------------|
| exp5-6 | 0-2% | 950 classes too hard |
| exp7 (frozen) | 67% | Spatial correlation breakthrough |
| exp8 (512px) | 59% | Higher resolution hurts |
| **exp9 (fine-tuned)** | **93%** | Task-specific features essential |

## Next Steps

Since the experiment exceeded all success criteria:

1. **Increase task difficulty** - Move to finer grids:
   - 3x3 grid (9 positions) - target 80%+ accuracy
   - 4x4 grid (16 positions) - target 70%+ accuracy
   - Gradually increase complexity toward the original 950-cell grid

2. **Add rotation prediction** - Predict both position AND rotation (0°, 90°, 180°, 270°)

3. **Test on real puzzles** - Use actual puzzle images instead of synthetic data

4. **Try alternative backbones** - Compare MobileNetV3 with:
   - EfficientNet-B0 (more capacity)
   - ConvNeXt-Tiny (modern architecture)
   - ResNet-18 (classic baseline)

5. **Optimize for inference** - The current model is small (2M params) but could be further optimized for mobile deployment

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
exp8 (high resolution)     → 512x512: 59% (resolution not the answer)
        ↓
exp9 (THIS EXPERIMENT)     → 93% test accuracy (FINE-TUNING SUCCESS!)
        ↓
(future: exp10...)         → Finer grids (3x3, 4x4), rotation prediction
```

## Summary

| Aspect | Exp7 (Phase 1) | Exp9 (Phase 2) |
|--------|----------------|----------------|
| Backbone | Frozen | Fine-tuned |
| Test Accuracy | 67.0% | **93.0%** |
| Improvement | - | **+26.0%** |
| Conclusion | Spatial correlation works | Task-specific features essential |

**Bottom line**: Fine-tuning the backbone is highly effective. The combination of spatial correlation architecture (exp7) + task-specific features (exp9) achieves strong cross-puzzle generalization. Ready to proceed to finer-grained tasks.
