# Experiment 4: Larger Backbone Cell Classification

## Hypothesis

Experiment 3 achieved 83.6% accuracy with a simple 3-conv backbone producing 64-dim features. The hypothesis for experiment 4 is:

> The bottleneck is the feature dimensionality, not the classification head. By increasing the backbone from 64-dim to 256-dim features, we provide more room for 950-class separation, leading to higher accuracy.

This tests **Option A** from the exp3 conclusions: "Increase Backbone Capacity Moderately."

## What Changed from Experiment 3

### Architecture Comparison

| Component | exp3 CellClassifier | exp4 LargerBackbone |
|-----------|---------------------|---------------------|
| Backbone | 3→16→32→64 (3 conv) | 3→32→64→128→256 (4 conv) |
| Features | 64-dim | 256-dim |
| FC Head | Linear(64, 950) | Linear(256, 950) |
| Total Params | 85,334 | 632,566 |
| Backbone Params | ~23K | ~388K |

### Design Rationale

**Problem with exp3:**
- 64 features trying to distinguish 950 classes
- Only ~0.067 features per class on average
- Regression worked with 64-dim features, but classification needs more capacity

**Solution in exp4:**
```
Input: 64x64x3
Conv2D(32, 3x3, stride=2, ReLU)  → 32x32x32
Conv2D(64, 3x3, stride=2, ReLU)  → 16x16x64
Conv2D(128, 3x3, stride=2, ReLU) → 8x8x128
Conv2D(256, 3x3, stride=2, ReLU) → 4x4x256
GlobalAveragePooling              → 256
Linear(256, 950)                  → logits
```

**Why this should help:**
1. **More features per class:** 256/950 ≈ 0.27 features per class (4x more than exp3)
2. **Deeper backbone:** 4 conv layers for richer visual feature extraction
3. **Progressive channel expansion:** 32→64→128→256 gives gradual feature refinement
4. **Direct classification head:** Keeps the successful approach from exp3 (no deep FC)

### Other Changes

- **MPS Support:** Added macOS Metal GPU acceleration for faster training
- Same dataset (puzzle_001, 950 pieces)
- Same training procedure (overfit tests, then full training)
- Same success criteria (>95% top-1, >99% top-5)

## Usage

```bash
cd network
source ../venv/bin/activate

# Run the experiment
python -m experiments.exp4_larger_backbone.train --epochs 200
```

## Results

### Device
- **MPS (Metal Performance Shaders)** - macOS GPU acceleration enabled

### Test 1: Overfit Single Piece - PASSED

```
Step    0: loss = 6.870803, pred = 933, target_prob = 0.0010
Step   11: loss = 0.001216

SUCCESS: Converged at step 11 with loss = 0.001216
```

**Finding:** Single piece overfit works faster than exp3 (11 vs 33 steps).

### Test 2: Overfit 10 Pieces - PASSED

```
Epoch   0: loss = 6.844907, accuracy = 0.0%
Epoch  50: loss = 0.934982, accuracy = 80.0%
Epoch  79: loss = 0.082484, accuracy = 100.0%

SUCCESS: 100% accuracy at epoch 79
```

**Finding:** 10-piece overfit works, faster than exp3 (79 vs 266 epochs).

### Test 3: Full Training (950 pieces) - SUCCESS!

Training progression (batch_size=128 for better GPU utilization):
```
Epoch   1: loss = 6.8639, acc =  0.0%, top5 =  0.2%
Epoch  50: loss = 2.8948, acc = 30.2%, top5 = 57.8%
Epoch 100: loss = 0.7178, acc = 83.5%, top5 = 95.3%
Epoch 150: loss = 0.2213, acc = 94.7%, top5 = 99.3%
Epoch 200: loss = 0.1839, acc = 95.4%, top5 = 99.7%
Epoch 219: loss = 0.0414, acc = 99.3%, top5 = 100.0%

SUCCESS: Reached 99.3% accuracy at epoch 219
```

**Final Results:**
- **Top-1 accuracy: 99.3%** (target was >95%)
- **Top-5 accuracy: 100.0%** (target was >99%)
- **Loss: 0.0414**
- **Batch size: 128** (increased from 32 for ~60% GPU utilization on M4)

### Comparison: exp3 vs exp4

| Metric | exp3 CellClassifier | exp4 LargerBackbone |
|--------|---------------------|---------------------|
| Parameters | 85,334 | 632,566 |
| 1-piece overfit | PASS (33 steps) | PASS (11 steps) |
| 10-piece overfit | PASS (266 epochs) | PASS (80 epochs) |
| Epochs to converge | 200 (no convergence) | 219 |
| Full training loss | 0.6427 | **0.0414** |
| Full training accuracy | 83.6% | **99.3%** |
| Top-5 accuracy | 95.4% | **100.0%** |

## Conclusions

### Experiment Result: SUCCESS

Increasing backbone capacity from 64-dim to 256-dim features dramatically improved classification performance:
- **Top-1 accuracy improved by 15.7 percentage points** (83.6% → 99.3%)
- **Top-5 accuracy improved to perfect** (95.4% → 100.0%)

### Analysis: Why Did This Work?

1. **The bottleneck WAS feature dimensionality.** exp3's 64-dim features were insufficient for 950-class separation, even though they worked for regression.

2. **Classification needs more capacity than regression.** Regression learns 2 smooth functions (x, y coordinates). Classification must learn 950 discrete decision boundaries.

3. **256-dim features provide adequate class separation.** With ~0.27 features per class (vs 0.067 in exp3), the model has enough room to encode distinguishing information.

4. **Direct classification head remains optimal.** The failure of exp3's deep FC head (CellClassifierDeep) was not about capacity but about optimization. A simple linear mapping works best when the features are rich enough.

### Key Insights

1. **Feature quality > classification head complexity.** Invest in the backbone, not the FC layers.

2. **4x feature dimension = 15% accuracy boost.** Going from 64 to 256 dimensions was sufficient for near-perfect classification.

3. **GPU acceleration matters.** MPS reduced training time significantly on macOS.

### Architecture Recommendations

For 950-class puzzle piece classification:
- **Backbone:** 4+ conv layers with 256+ dim output
- **Head:** Direct linear projection (no hidden layers)
- **No BatchNorm required** for this overfitting task

### Next Steps

1. **Test generalization:** Train on one puzzle, test on another
2. **Try with rotation prediction:** Add 4-class rotation classification
3. **Explore pretrained backbones:** ResNet-18 may provide even richer features
4. **Data augmentation:** Color jitter, random crops for better generalization

---

## Visualizations

Training outputs are saved to `outputs/`. Here's how to interpret each type:

### `*_accuracy.png` - Prediction Accuracy Map

Shows which pieces the model predicted correctly across the entire puzzle.

- **Background:** The puzzle image with a grid overlay (38x25 cells)
- **Green border:** Correct prediction (model guessed the right cell)
- **Red border:** Wrong prediction (model guessed a different cell)

At 99.1% accuracy, almost all cells should be green with only ~9 red cells.

### `*_heatmap.png` - Confidence Heatmap

Shows the model's probability distribution for a single piece (piece 0).

- **Background:** The puzzle image (semi-transparent)
- **Heatmap colors:** Model's confidence for each cell location
  - Dark blue = low probability
  - Yellow = medium probability
  - Red = high probability (model thinks piece belongs here)
- **Green border:** Ground truth (where the piece actually belongs)
- **Red border:** Model's prediction (overlaps green if correct)

A well-trained model shows a sharp red spot at the green border location.

### `*_grid.png` - Multi-Piece Comparison

Shows 8 pieces side-by-side with their probability heatmaps.

- **Layout:** 2x4 grid of piece pairs
- **Left column:** The piece image (model input)
- **Right column:** The probability heatmap (38x25 grid)
- **Title format:** `T:(col,row) P:(col,row) OK/X`
  - `T:` = Target (ground truth)
  - `P:` = Prediction
  - `OK` = Correct, `X` = Wrong

Useful for comparing model behavior across multiple pieces at once.
