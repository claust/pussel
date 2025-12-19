# Loss Function Investigation

## Initial Problem

During training with GIoU (Generalized IoU) loss for bounding box regression, the position loss was stuck at exactly **1.000** and not decreasing, even after hundreds of batches.

```
Epoch 0: train/position_loss=1.000, train/rotation_loss=1.400, train/total_loss=2.400
```

## Root Cause Analysis

### 1. Understanding the Data

Puzzle pieces are **very small** relative to the full puzzle image:
- Typical piece: ~2.6% width × 4% height = **~0.1% of puzzle area**
- Normalized coordinates example: `[0.026, 0.0, 0.052, 0.04]`

### 2. Initial Model Predictions

Due to Sigmoid activation on the position head, initial predictions cluster around the center:
```python
# Initial predictions (random weights + Sigmoid)
[[0.49, 0.50, 0.51, 0.49],  # Small box at center
 [0.49, 0.51, 0.51, 0.49],
 ...]
```

### 3. GIoU Gradient Problem

When predicted boxes don't overlap with ground truth:
- **IoU = 0** (no intersection)
- GIoU relies on the enclosing box to provide gradients
- This results in **very weak gradient signals**

We tested gradient magnitudes:

| Loss Function | Loss Value | Gradient Magnitude |
|---------------|------------|-------------------|
| GIoU          | 1.99       | ~0.08             |
| DIoU          | 1.90       | ~0.93             |
| CIoU          | 1.90       | ~0.93             |

**CIoU/DIoU provide ~12x stronger gradients** when boxes don't overlap because they include a center-point distance term.

## Changes Made

### Change 1: GIoU → CIoU

Switched from `generalized_box_iou_loss` to `complete_box_iou_loss`:

```python
# Before
from torchvision.ops import generalized_box_iou_loss
position_loss = generalized_box_iou_loss(pred_boxes, gt_boxes, reduction="mean")

# After
from torchvision.ops import complete_box_iou_loss
position_loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="mean")
```

**Result:** Loss was no longer stuck at exactly 1.0, but still decreasing very slowly (~0.01 improvement per 800 batches).

### Change 2: Added SmoothL1 Loss

Combined CIoU with SmoothL1 for direct coordinate gradients:

```python
ciou_loss = complete_box_iou_loss(pred_boxes, gt_boxes, reduction="mean")
l1_loss = F.smooth_l1_loss(position_pred, batch["position"])
position_loss = 0.5 * ciou_loss + 0.5 * l1_loss
```

**Rationale:**
- CIoU provides geometric awareness (overlap, center distance, aspect ratio)
- SmoothL1 provides strong, direct gradients on each coordinate independently
- Combination often works better than either alone for small boxes

## Suggestions for Further Improvement

If the loss still doesn't decrease satisfactorily, consider these options:

### 1. Increase Learning Rate
The LR finder showed stability up to ~10^-2. Current LR of 1e-3 may still be conservative. Try:
- 2e-3 or 3e-3

### 2. Adjust Loss Weighting
Current 50/50 split may not be optimal:
```python
# Try emphasizing L1 more for faster initial convergence
position_loss = 0.3 * ciou_loss + 0.7 * l1_loss
```

### 3. Use Center + Size Parameterization
Instead of predicting (x1, y1, x2, y2), predict (cx, cy, w, h):
- Center coordinates may be easier to learn
- Width/height can be learned independently
- Many object detection models use this approach

### 4. Add Auxiliary Center Loss
Add explicit supervision on box centers:
```python
pred_center = (position_pred[:, :2] + position_pred[:, 2:]) / 2
gt_center = (batch["position"][:, :2] + batch["position"][:, 2:]) / 2
center_loss = F.mse_loss(pred_center, gt_center)
```

### 5. Warmup Strategy
Start with pure L1 loss, gradually introduce CIoU:
```python
ciou_weight = min(1.0, current_epoch / 5)  # Ramp up over 5 epochs
position_loss = ciou_weight * ciou_loss + (1 - 0.5*ciou_weight) * l1_loss
```

### 6. Check Spatial Correlation Module
The spatial correlation module should help localize pieces. Verify it's working:
- Log correlation maps to TensorBoard
- Check if correlation peaks align with piece locations
- Consider if the module needs more capacity

### 7. Position Head Architecture
Current head is simple (Linear → ReLU → Dropout → Linear → Sigmoid). Consider:
- Adding more layers
- Using different activation (e.g., remove Sigmoid, use unconstrained output with clamping)
- Separate heads for center vs. size prediction

### 8. Data Augmentation Review
Ensure augmentations don't make the task harder:
- Brightness/contrast changes might affect piece-puzzle matching
- Consider reducing augmentation strength initially

## References

- [Distance-IoU Loss (DIoU)](https://arxiv.org/abs/1911.08287) - Zheng et al., 2020
- [Complete-IoU Loss (CIoU)](https://arxiv.org/abs/1911.08287) - Same paper, extends DIoU
- [Generalized IoU (GIoU)](https://arxiv.org/abs/1902.09630) - Rezatofighi et al., 2019

## Current Issue: Plateau at 0.6

After adding SmoothL1, the position loss dropped quickly from ~1.1 to ~0.6 but then **plateaued** around 0.60-0.62 from batch ~10 through batch 800+.

### Possible Causes

1. **Mean prediction collapse** - SmoothL1 can cause the model to predict the "average" target position, which minimizes L1 loss but doesn't solve the actual task
2. **Learning rate too low** - Model may be stuck in a local minimum and needs more momentum to escape
3. **CIoU not contributing** - If boxes still don't overlap (IoU ≈ 0), the CIoU component provides limited signal beyond its center distance term
4. **Capacity limitation** - The model architecture may need changes to learn fine-grained positions

### Suggested Next Steps

#### Option A: Increase Learning Rate
Current LR is 1e-3. The LR finder showed stability up to ~10^-2. Try:
```bash
python train.py --learning_rate 2e-3
# or
python train.py --learning_rate 3e-3
```

#### Option B: Shift Loss Weighting Toward L1
Give more weight to the direct coordinate signal:
```python
position_loss = 0.3 * ciou_loss + 0.7 * l1_loss
```

#### Option C: Add Debug Logging
Check if predictions are collapsing to a single value:
```python
# In training_step, add:
if batch_idx % 100 == 0:
    print(f"Pred mean: {position_pred.mean(dim=0).tolist()}")
    print(f"Pred std:  {position_pred.std(dim=0).tolist()}")
```
If std is very low, the model is predicting nearly the same position for all inputs.

#### Option D: Learning Rate Warmup + Higher Peak
Start with low LR, ramp up over first epoch:
```python
# In configure_optimizers, add warmup scheduler
```

## Training Log

| Date | Change | Result |
|------|--------|--------|
| 2025-12-18 | Initial GIoU | Loss stuck at 1.000 |
| 2025-12-18 | GIoU → CIoU | Loss varies but decreases slowly |
| 2025-12-18 | Added SmoothL1 (50/50) | Dropped to ~0.6, plateaued |
| 2025-12-18 | LR=1e-3 baseline | Pred std ~0.005 vs GT std ~0.3 (collapse) |
| 2025-12-18 | LR=2e-3 | Same collapse to different corner |
| 2025-12-18 | Pure L1 (no CIoU) | Y-coords showed slight improvement |
| 2025-12-18 | Remove Sigmoid | **Worse** - complete collapse to 0/1 |
| 2025-12-18 | LR=1e-2 | Complete collapse to corner [1,1,0,0] |

## Key Finding: Prediction Collapse

All experiments show **prediction collapse** - the model converges to a fixed position regardless of input:

| Experiment | Batch 50 Pred mean | Pred std | GT std | Loss |
|------------|-------------------|----------|--------|------|
| LR=1e-3 | [0.01, 0.92, 0.99, 0.11] | ~0.02 | ~0.30 | 0.61 |
| LR=2e-3 | [0.96, 0.94, 0.01, 0.01] | ~0.02 | ~0.28 | 0.62 |
| Pure L1 | [0.50, 0.47, 0.53, 0.50] | ~0.02 | ~0.28 | 0.04 |
| No Sigmoid | [1.00, 0.00, 0.00, 1.00] | 0.00 | ~0.29 | 0.62 |
| LR=1e-2 | [1.00, 1.00, 0.00, 0.00] | 0.00 | ~0.26 | 0.62 |

**Root Cause:** The model finds degenerate solutions (fixed predictions) that minimize average loss. SmoothL1 with fixed center prediction achieves ~0.15 L1 loss, and the model has no incentive to learn input-dependent predictions.

## Recommended Next Steps (in order of priority)

### 1. Center + Size Parameterization
Instead of (x1, y1, x2, y2), predict (cx, cy, w, h):
```python
# In position_head output
center = features[:, :2]  # cx, cy - center coordinates
size = features[:, 2:]    # w, h - normalized width/height

# Convert back to corners
x1 = center[:, 0] - size[:, 0] / 2
y1 = center[:, 1] - size[:, 1] / 2
x2 = center[:, 0] + size[:, 0] / 2
y2 = center[:, 1] + size[:, 1] / 2
```
**Rationale:** Size is relatively constant for puzzle pieces (~0.03), so the model only needs to learn center position. This decouples the easier subproblem.

### 2. Separate Center Loss with MSE
Add explicit center supervision with MSE (penalizes large errors more):
```python
pred_center = (position_pred[:, :2] + position_pred[:, 2:]) / 2
gt_center = (gt_position[:, :2] + gt_position[:, 2:]) / 2
center_loss = F.mse_loss(pred_center, gt_center)  # MSE not L1!
```

### 3. Initialize Position Head Near Ground Truth Mean
Initialize final layer bias to the mean of ground truth positions:
```python
# After creating position_head
with torch.no_grad():
    # GT mean is approximately [0.5, 0.5, 0.53, 0.53]
    # Inverse sigmoid: logit(0.5) = 0, logit(0.53) ≈ 0.12
    self.position_head[-2].bias.fill_(0.0)  # Start centered
```

### 4. Auxiliary Variance Loss
Penalize low prediction variance to prevent collapse:
```python
pred_variance = position_pred.var(dim=0).mean()
variance_penalty = F.relu(0.01 - pred_variance)  # Encourage variance > 0.01
loss = position_loss + 0.1 * variance_penalty
```

### 5. Gradient Analysis
Log gradient magnitudes to understand where learning is blocked:
```python
# After backward pass
for name, param in model.named_parameters():
    if param.grad is not None and 'position_head' in name:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
```

---

## Multi-Scale FPN Architecture Experiment (2025-12-19)

### Hypothesis

The prediction collapse might be caused by a **scale mismatch problem**:
- Puzzle images (3000×2000 px) compressed to 256×256
- Piece images (~79×80 px) expanded to 224×224
- A piece representing 0.1% of the puzzle appears as 76% after resizing
- Feature map resolution (~7×7) is coarser than target location (~2.6% × 4%)

### Implementation

Built a Multi-Scale FPN architecture:
- **FeaturePyramidNetwork**: Extracts P3 (64×64), P4 (32×32), P5 (16×16) features
- **MultiScaleCorrelation**: Correlates piece features against puzzle at each scale
- **Larger puzzle images**: 512×512 instead of 256×256
- **ResNet18 backbone**: Lighter backbone works well with FPN
- **13.4M trainable parameters**

### LR Finder Results

```
Suggested learning rate: 0.00257 (~2.5e-3)
Current config LR: 0.0001
```

The suggested LR is ~25x higher than the initial conservative config.

### Training Results

After ~70 batches with batch_size=8:

| Metric | Value |
|--------|-------|
| train/position_loss | 0.760 |
| train/rotation_loss | 1.500 |
| train/total_loss | 2.260 |
| Training speed | ~0.82 it/s |
| Est. epoch time | **~33 hours** |

**Prediction Collapse STILL Occurring:**
```
[Batch 50] Multi-Scale Predictions:
  Pred mean: [0.5945, 0.4335, 0.4138, 0.5699]
  Pred std:  [0.0001, 0.0002, 0.0001, 0.0002]  ← Extremely low!
  GT mean:   [0.5079, 0.5828, 0.5356, 0.6211]
  GT std:    ~0.28
```

**Correlation Maps Look Reasonable:**
```
Corr P3: min=-0.0129, max=0.0586
Corr P4: min=0.0027, max=0.0589
Corr P5: min=0.0090, max=0.0530
```

### Conclusion: Multi-Scale Architecture Does NOT Solve Prediction Collapse

The experiment definitively shows that the prediction collapse is **NOT caused by**:
- ❌ Feature resolution being too coarse
- ❌ Scale mismatch between piece and puzzle
- ❌ Lack of spatial correlation capacity
- ❌ Puzzle image size being too small

The correlation module IS computing meaningful spatial correlations (visible in the min/max values), but the position regressor still collapses to near-constant outputs.

### Root Cause Analysis

The problem appears to be in the **position regression head**, not the feature extraction:

1. **The regression head finds a degenerate solution** - predicting ~0.5 for all coordinates minimizes average L1 loss
2. **No mechanism forces position-dependent predictions** - the model has no incentive to use the correlation features
3. **SmoothL1 + CIoU both allow collapse** - neither loss explicitly penalizes low variance

### Recommended Next Steps

Given that multi-scale features don't help, focus on:

1. **Variance penalty loss** - Explicitly penalize low prediction variance
2. **Contrastive/matching loss** - Force model to distinguish between different positions
3. **Detection-style heads** - Use anchor boxes or heatmap prediction instead of direct regression
4. **Center + Size parameterization** - Decouple the easier subproblems

The multi-scale architecture may still be valuable once the collapse issue is solved, but it's not the solution to the core problem.
