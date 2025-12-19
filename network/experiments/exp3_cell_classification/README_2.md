# Cell Classification Experiment - Attempt 2

## Hypothesis

The first attempt (`README_1.md`) achieved 83.6% accuracy with a direct 64 → 950 classification head. The hypothesis for attempt 2 is:

> The bottleneck is the abrupt mapping from 64 features to 950 classes. Adding intermediate FC layers will give the network more capacity to transform and organize the feature space, leading to higher accuracy.

## What Changed from Attempt 1

### Architecture Comparison

| Component | Attempt 1 (CellClassifier) | Attempt 2 (CellClassifierDeep) |
|-----------|---------------------------|-------------------------------|
| Backbone | Conv 3→16→32→64, GAP | Same |
| Features | 64-dim | 64-dim |
| FC Head | Linear(64, 950) | Linear(64, 256) → ReLU → Linear(256, 512) → ReLU → Linear(512, 950) |
| Parameters | 85,334 | ~660,000 |

### Design Rationale

**Problem with Attempt 1:**
- 64 features trying to distinguish 950 classes
- Only ~0.067 features per class on average
- Direct linear mapping may not have enough capacity

**Solution in Attempt 2:**
```
64 features (from backbone)
    ↓
Linear(64, 256) + ReLU      # Expand feature space 4x
    ↓
Linear(256, 512) + ReLU     # Further expansion
    ↓
Linear(512, 950)            # Final classification
```

**Why this should help:**
1. **Progressive expansion:** 64 → 256 → 512 → 950 is a gradual increase
2. **Non-linear transformations:** ReLU between layers allows complex feature combinations
3. **More capacity:** 660K params (8x more than attempt 1) provides room for memorization
4. **Feature organization:** Intermediate layers can learn to cluster similar pieces together

### What Stays the Same

- Same backbone (Conv layers)
- Same dataset (puzzle_001, 950 pieces)
- Same training procedure (overfit tests, then full training)
- Same success criteria (>95% top-1, >99% top-5)

## Expected Outcomes

### If This Works (accuracy > 95%)

The bottleneck was indeed the FC head capacity. Implications:
- Classification is viable but needs more capacity than regression
- The 64-dim feature space from the backbone is sufficient
- May want to try intermediate sizes (64 → 256 → 950)

### If This Doesn't Work (accuracy similar to attempt 1)

The bottleneck is in the backbone, not the FC head. Next steps:
- Increase backbone capacity (more channels)
- Try a different approach (hierarchical classification, regression + discretization)

## Usage

```bash
cd network
source ../venv/bin/activate

# Run with the new deep model
python -m experiments.3_cell_classification.train --model cell_classifier_deep --epochs 200
```

## Results

### Test 1: Overfit Single Piece - PASSED

```
Step    0: loss = 6.870633, pred = 577, target_prob = 0.0010
Step   21: loss = 0.000979

SUCCESS: Converged at step 21 with loss = 0.000979
```

**Finding:** Single piece overfit works, even faster than attempt 1 (21 vs 33 steps).

### Test 2: Overfit 10 Pieces - PASSED

```
Epoch   0: loss = 6.854482, accuracy = 0.0%
Epoch 100: loss = 0.845998, accuracy = 80.0%
Epoch 274: loss = 0.097864, accuracy = 100.0%

SUCCESS: 100% accuracy at epoch 274
```

**Finding:** 10-piece overfit works, similar to attempt 1.

### Test 3: Full Training (950 pieces) - FAILED COMPLETELY

```
Epoch   1: loss = 6.8817, acc = 0.0%, top5 = 0.0%
Epoch  50: loss = 6.8582, acc = 0.0%, top5 = 0.4%
Epoch 100: loss = 6.8581, acc = 0.1%, top5 = 0.3%
Epoch 200: loss = 6.8580, acc = 0.0%, top5 = 0.2%
```

**Final Results:**
- **Top-1 accuracy: 0.0%** (attempt 1 achieved 83.6%)
- **Top-5 accuracy: 0.2%** (attempt 1 achieved 95.4%)
- **Loss: 6.858** (stuck at ln(950) = random chance)

### Comparison: Attempt 1 vs Attempt 2

| Metric | Attempt 1 (CellClassifier) | Attempt 2 (CellClassifierDeep) |
|--------|---------------------------|-------------------------------|
| Parameters | 85,334 | 659,158 |
| 1-piece overfit | PASS (33 steps) | PASS (21 steps) |
| 10-piece overfit | PASS (266 epochs) | PASS (274 epochs) |
| Full training loss | 0.6427 | 6.858 (no learning) |
| Full training accuracy | **83.6%** | **0.0%** |
| Top-5 accuracy | 95.4% | 0.2% |

## Conclusions

### Experiment Result: FAILED

Adding deeper FC layers made performance **dramatically worse**, not better.

### Analysis: Why Did This Happen?

1. **The loss is stuck at ln(950) ≈ 6.857**, which is the cross-entropy loss for uniform random predictions over 950 classes. The model never learned anything useful.

2. **Overfit tests passed**, meaning the architecture CAN learn when:
   - Training on very few samples (1 or 10)
   - Gradients are strong and clear

3. **Full training failed**, indicating problems with:
   - **Gradient flow** through 3 FC layers with ReLU
   - **Optimization landscape** - deeper networks are harder to optimize
   - **Learning rate** may be too low for the larger model

### Root Cause Hypothesis

The deeper FC head (64 → 256 → 512 → 950) creates a more complex optimization landscape. With:
- ReLU activations potentially causing "dead neurons"
- No batch normalization to stabilize training
- Same learning rate (1e-3) as the simpler model
- 8x more parameters to optimize

The gradients either vanish or the optimization gets stuck in a bad local minimum.

### Key Insight

**More parameters ≠ better learning.** The simpler architecture (64 → 950) actually worked better because:
1. Fewer layers = better gradient flow
2. Direct mapping = simpler optimization landscape
3. The bottleneck was NOT the FC head capacity

### Implications

1. **The bottleneck is in the backbone**, not the classification head
2. The 64-dim features from the backbone may not contain enough discriminative information
3. Adding FC layers between backbone and output doesn't help if the input features are insufficient

### Recommended Next Steps

1. **Improve the backbone** - More conv layers, more channels, or pretrained features
2. **Try batch normalization** - May help stabilize training of deeper FC heads
3. **Increase learning rate** - The deeper model may need a higher LR
4. **Consider regression** - The single_puzzle_overfit showed regression works well with this backbone

---

## Discussion: What Should Attempt 3 Be?

After attempts 1 and 2, we considered two options for improving classification performance.

### What We Know So Far

| Experiment | Backbone | Task | Result |
|------------|----------|------|--------|
| single_puzzle_overfit | Simple 3-conv (64-dim) | Regression | **SUCCESS** (loss < 0.01) |
| cell_classification #1 | Simple 3-conv (64-dim) | Classification (950 classes) | 83.6% accuracy |
| cell_classification #2 | Simple 3-conv (64-dim) + deep FC | Classification (950 classes) | 0% (optimization failure) |

### Key Insight from Regression Success

The regression experiment achieved near-perfect results with the simple backbone. This tells us:

> **The 64-dim features from the simple backbone contain enough information to distinguish all 950 pieces.**

If the features were insufficient, regression would have failed too.

### Why Classification Struggled

The classification task (950 classes) is harder than regression (2 continuous values) because:
1. Must learn 950 decision boundaries vs. 2 smooth functions
2. No notion of "close" - neighbor cells are penalized the same as far cells
3. Cross-entropy is less forgiving than MSE

### Option A: Increase Backbone Capacity Moderately

Instead of the simple 3-conv backbone, use a larger one:

```
Current:  3→16→32→64 (23K params, 64-dim features)
Proposed: 3→32→64→128→256 (150K params, 256-dim features)
```

**Arguments FOR:**
- Tests whether feature dimensionality is the bottleneck
- Stays close to current architecture (controlled experiment)
- Lower complexity than ResNet-18
- Faster training

**Arguments AGAINST:**
- Regression already works with 64-dim features, so dimension may not be the issue
- Still a custom architecture without pretrained weights
- May not be enough capacity for 950-class classification

### Option B: Move to ResNet-18

Replace the simple backbone with ResNet-18 (pretrained on ImageNet).

| Aspect | Simple Backbone | ResNet-18 |
|--------|----------------|-----------|
| Parameters | ~23K | ~11M |
| Feature dim | 64 | 512 |
| Pretrained | No | Yes (ImageNet) |
| Architecture | 3 conv layers | 18 layers + skip connections |

**Arguments FOR:**
1. **Richer features**: 512-dim vs 64-dim gives more room for 950-class separation
2. **Pretrained knowledge**: Already knows useful visual features from ImageNet
3. **Better gradient flow**: Skip connections prevent vanishing gradients (which may have caused attempt 2's failure)
4. **Standard approach**: Industry-proven for classification tasks
5. **Establishes upper bound**: Shows what's achievable with a strong backbone

**Arguments AGAINST:**
1. **Regression already works**: The simple backbone's 64-dim features ARE sufficient for the task (proven by single_puzzle_overfit)
2. **Overkill for overfit test**: 11M params to memorize 950 samples is excessive
3. **Different learning dynamics**: May need different LR, longer training, careful tuning
4. **Obscures the core question**: Are we testing classification vs. regression, or architecture?
5. **Slower iteration**: Larger model = slower training cycles

### Comparison Summary

| Path | What it tests | Complexity | Risk |
|------|---------------|------------|------|
| Option A: Bigger backbone (256-dim) | Is feature dimension the issue? | Low | May not be enough |
| Option B: ResNet-18 frozen | Can rich features solve classification? | Medium | Overkill, but answers the question |
| Option B: ResNet-18 fine-tuned | Can we achieve perfect classification? | High | Slow, complex |

### If Choosing ResNet-18

Recommended approach:
1. **Freeze backbone initially** - Only train the classification head first
2. **Use pretrained weights** - ImageNet pretrained
3. **Lower learning rate** - 1e-4 instead of 1e-3 for the backbone
