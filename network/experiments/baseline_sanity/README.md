# Baseline Sanity Check

A minimal experiment to verify that our neural network training pipeline works at the most fundamental level.

## Why This Exists

We were training a bounding box regression model for puzzle piece localization, but the loss was stuck at 1.0 and wouldn't budge. We tried:
- Vanilla IoU loss
- CIoU loss
- Combined L1 + CIoU

Nothing helped. This told us the problem wasn't the loss function — **the model wasn't learning anything at all**. The gradient signal wasn't reaching the weights.

## Diagnosis Approach

Before debugging complex architecture, we need to verify the fundamentals work. This is a "puzzle for 3-year-olds" — the absolute simplest possible setup that should trivially succeed. If this fails, we know exactly where to look.

## The Experiment

### Dataset: Single Square Localization
- 64x64 RGB images
- Solid gray background (random shade 100-180)
- One colored square, fixed size 16x16, random color
- Square placed randomly (fully within bounds)
- Target: normalized center coordinates (cx, cy) in [0, 1]
- No classification, no varying sizes, no multiple objects
- Generated on-the-fly, no files needed

### Architecture (~15K params)
```
Input: 64x64x3
Conv2D(16, 3x3, stride=2, ReLU, padding=1)  → 32x32x16
Conv2D(32, 3x3, stride=2, ReLU, padding=1)  → 16x16x32
Conv2D(64, 3x3, stride=2, ReLU, padding=1)  → 8x8x64
GlobalAveragePooling                         → 64
Linear(64, 2)                                → (x, y)
Sigmoid                                      → output in [0, 1]
```

No BatchNorm. No Dropout. No skip connections. Just the bare minimum.

### Loss Function
```python
loss = F.mse_loss(pred_xy, target_xy)
```

No IoU variants. Just MSE on coordinates. IoU-based losses are for refinement, not for getting off the ground.

### Training Config
- Adam optimizer, lr=1e-3
- Batch size 32
- Train for 50 epochs or until loss < 0.001

## Verification Checks

The training script includes these built-in verification checks:

1. **Overfit 1 sample**: Loop on a single image until loss → ~0. If this doesn't converge, something is fundamentally broken.

2. **Overfit 10 samples**: Should reach near-zero loss quickly.

3. **Print predictions**: Every epoch, print (pred, target) pairs to verify predictions are actually changing.

4. **Gradient check**: Every N steps, print gradient norms per layer to watch for zeros or exploding values.

5. **Visualize**: Save sample images with predicted box (red) vs ground truth (green) overlay every 10 epochs.

## Success Criteria

- Loss drops to < 0.01 within 20 epochs
- Predicted centers visually align with squares
- Gradients flow through all layers (no zeros)

## Usage

Run from the `network/` directory:

```bash
cd network
python -m experiments.baseline_sanity.train
```

Or run individual components:
```bash
python -m experiments.baseline_sanity.dataset    # Test dataset generation
python -m experiments.baseline_sanity.model      # Test model architecture
python -m experiments.baseline_sanity.visualize  # Test visualization
```

## File Structure

```
experiments/baseline_sanity/
├── README.md       # This file
├── dataset.py      # On-the-fly square image generator
├── model.py        # Tiny conv net (~15K params)
├── train.py        # Training loop with all verification checks
├── visualize.py    # Helper to draw boxes on images
└── outputs/        # Saved visualizations (generated during training)
```

## Next Steps

Once this trivial case succeeds, progressively add complexity:
1. Variable square sizes
2. Full bounding boxes (x1, y1, x2, y2)
3. IoU-based loss functions
4. Multiple objects
5. Real puzzle piece images

But not until this baseline succeeds.
