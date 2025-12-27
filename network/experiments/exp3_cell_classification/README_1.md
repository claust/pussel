# Cell Classification Experiment

A controlled experiment to test whether framing puzzle piece localization as
**cell classification** (instead of coordinate regression) improves learning.
This is a small incremental step from the single puzzle overfit experiment.

## Background: Why This Experiment?

### Previous Experiment Results

The single puzzle overfit experiment (`experiments/single_puzzle_overfit/`)
successfully demonstrated that:

1. A simple CNN can memorize piece-to-position mappings for real texture data
2. MSE loss on center coordinates (cx, cy) works for learning
3. Final loss of 0.007 was achieved after 500 epochs

### The Question

The previous experiment used **coordinate regression**: output two continuous
values (cx, cy) in [0, 1]. But puzzles have a natural **discrete structure** -
pieces belong to specific cells in a grid.

> Can framing this as classification over cells improve learning or provide
> better interpretability?

### Potential Benefits of Cell Classification

1. **Discrete output space**: 950 cells instead of continuous [0,1] x [0,1]
2. **Cross-entropy loss**: Well-studied, often easier to optimize than MSE
3. **Interpretable output**: Probability distribution shows model's confidence
   across all cells
4. **Heatmap visualization**: Can overlay probabilities on the puzzle image
5. **Natural for puzzles**: Pieces genuinely belong to discrete cells, not
   arbitrary coordinates

## Experiment Design

### Core Change from Previous Experiment

| Aspect      | Single Puzzle Overfit   | Cell Classification        |
| ----------- | ----------------------- | -------------------------- |
| Output      | (cx, cy) in [0, 1]      | Probability over 950 cells |
| Loss        | MSE                     | Cross-entropy              |
| Final layer | Linear(64, 2) + Sigmoid | Linear(64, 950) + Softmax  |
| Target      | Normalized coordinates  | Cell index (0-949)         |
| Evaluation  | Coordinate error        | Top-1 accuracy             |

### What Stays the Same

- **Same puzzle**: puzzle_001 (38 columns x 25 rows = 950 pieces)
- **Same piece encoder**: Conv layers -> GAP -> features
- **Same piece size**: 64x64 pixels
- **Same approach**: Overfit test (train and evaluate on all pieces)
- **Same philosophy**: Start simple, verify it works

### Grid Structure

Puzzle_001 has a fixed grid:

- **Columns**: 38
- **Rows**: 25
- **Total cells**: 950 (one piece per cell)

Each piece maps to exactly one cell index: `cell_index = row * 38 + col`

### Target Format

Instead of normalized (cx, cy) coordinates, we compute a cell index:

```python
# From normalized coordinates
col = int(cx * num_cols)  # 0-37
row = int(cy * num_rows)  # 0-24
cell_index = row * num_cols + col  # 0-949
```

### Architecture

Same backbone, different head:

```
Piece (64x64x3)
    |
Conv2D(16, 3x3, stride=2, ReLU, padding=1)  -> 32x32x16
Conv2D(32, 3x3, stride=2, ReLU, padding=1)  -> 16x16x32
Conv2D(64, 3x3, stride=2, ReLU, padding=1)  -> 8x8x64
GlobalAveragePooling                         -> 64
    |
Linear(64, 950)                              -> 950 logits
    |
Softmax (during inference)                   -> probabilities
```

### Loss Function

```python
loss = F.cross_entropy(logits, target_cell_index)
```

### Visualization: Heatmaps

The key benefit is interpretable output. For each piece, we can:

1. Get the 950-dimensional probability vector
2. Reshape to (25, 38) grid
3. Overlay as a heatmap on the puzzle image

This shows:

- Where the model thinks the piece belongs (peak)
- Alternative candidates (secondary peaks)
- Overall confidence distribution

## Success Criteria

| Metric         | Target | Meaning                           |
| -------------- | ------ | --------------------------------- |
| Top-1 Accuracy | > 95%  | Correct cell predicted            |
| Top-5 Accuracy | > 99%  | Correct cell in top 5 predictions |
| Loss           | < 0.5  | Cross-entropy converging          |

### Comparison with Coordinate Regression

We can compare results:

- Regression MSE 0.007 corresponds to ~8% average error in each coordinate
- For a 38x25 grid, that's ~3 columns and ~2 rows of error
- Classification should achieve near-perfect accuracy if it can memorize

## File Structure

```
experiments/3_cell_classification/
├── README.md           # This file
├── __init__.py         # Package marker
├── dataset.py          # Load pieces with cell index targets
├── model.py            # Classifier network (950 outputs)
├── train.py            # Training with accuracy metrics
├── visualize.py        # Heatmap visualization on puzzle
└── outputs/            # Saved visualizations
```

## Usage

```bash
cd network
source ../venv/bin/activate

# Run full experiment
python -m experiments.cell_classification.train

# Test individual components
python -m experiments.cell_classification.dataset    # Verify data loading
python -m experiments.cell_classification.model      # Test architecture
python -m experiments.cell_classification.visualize  # Test heatmap visualization
```

## Expected Outcomes

### If Classification Works Better

- Faster convergence than regression
- Higher effective accuracy (correct cell vs close coordinates)
- Clearer failure modes visible in heatmaps

### If Classification Works Similarly

- Validates that both formulations are learnable
- Heatmap visualization still provides value
- May prefer regression for sub-cell precision

### If Classification Fails

- The 950-class problem may be too hard for the small network
- May need larger capacity (more features before classification head)
- Could try coarser grid first (e.g., 4x4 regions)

## Relationship to Previous Experiments

```
baseline_sanity          -> Verified training pipeline works
        |
        v
single_puzzle_overfit    -> Verified real texture learning works (regression)
        |
        v
cell_classification      -> Test classification formulation (THIS EXPERIMENT)
        |
        v
(future)                 -> Add rotation, multi-puzzle, etc.
```

## Results

### Test 1: Overfit Single Piece - PASSED

Training on a single piece converged quickly:

```
Step    0: loss = 6.983494, pred = 890, target_prob = 0.0009
Step   33: loss = 0.007856, pred = 0, target_prob = 0.9922

SUCCESS: Converged at step 33 with loss = 0.007856
```

**Finding:** The network can perfectly classify a single piece to its correct
cell.

### Test 2: Overfit 10 Pieces - PASSED

Training on 10 pieces converged, though slower than regression:

```
Epoch   0: loss = 6.878081, accuracy = 0.0%
Epoch 150: loss = 0.818187, accuracy = 100.0%
Epoch 266: loss = 0.098456, accuracy = 100.0%

SUCCESS: 100% accuracy at epoch 266
```

**Finding:** The network can memorize a small set of pieces, though 10-piece
classification (266 epochs) took longer than 10-piece regression (~100 epochs in
single_puzzle_overfit).

### Test 3: Full Training (950 pieces) - DID NOT REACH TARGET

Training on all 950 pieces achieved good but not target accuracy:

```
Epoch   1: loss = 6.8718, acc =  0.0%, top5 =  0.0%
Epoch  50: loss = 4.1965, acc =  6.4%, top5 = 24.0%
Epoch 100: loss = 2.4651, acc = 34.8%, top5 = 67.2%
Epoch 150: loss = 1.2422, acc = 65.8%, top5 = 88.7%
Epoch 200: loss = 0.6427, acc = 83.6%, top5 = 95.4%
```

**Final Results:**

- **Top-1 accuracy: 83.6%** (target was >95%)
- **Top-5 accuracy: 95.4%** (target was >99%)
- **Loss: 0.6427** (target was <0.5)

### Comparison with Coordinate Regression

| Metric             | Regression (single_puzzle_overfit) | Classification (this experiment) |
| ------------------ | ---------------------------------- | -------------------------------- |
| Parameters         | 23,234                             | 85,334                           |
| Final loss         | 0.0069                             | 0.6427                           |
| Epochs to converge | ~500                               | >200 (not converged)             |
| Success criterion  | MSE < 0.01                         | Accuracy > 95%                   |
| Result             | **PASSED**                         | Did not reach target             |

### Visualization Analysis

The accuracy grid visualizations show:

1. **Epoch 1:** All cells red (wrong) - network predicting randomly
2. **Epoch 100:** ~54% green cells scattered across puzzle
3. **Epoch 200:** ~90% green cells, errors scattered (not clustered)

The heatmap visualizations show:

- Model learns to focus probability mass in the correct region
- Some probability "leakage" to neighboring cells
- High confidence on many predictions, but not all

## Conclusions

### Experiment Result: PARTIAL SUCCESS

The cell classification experiment demonstrated that the approach is **learnable
but harder** than coordinate regression for the same backbone architecture.

### Key Findings

1. **Classification is harder than regression for this task:** With the same
   64-dimensional feature space before the output layer, regression (2 outputs)
   significantly outperformed classification (950 outputs). This suggests the
   increased output dimensionality makes the learning problem more difficult.

2. **High top-5 accuracy indicates the model is "close":** At 95.4% top-5
   accuracy, the model usually has the correct cell in its top predictions. The
   errors are not random - they tend to be nearby cells.

3. **More parameters didn't automatically help:** The CellClassifier (85K
   params) has 3.7x more parameters than PieceLocNet (23K params), yet performed
   worse. The additional parameters are in the output layer (64→950), which may
   not be the bottleneck.

4. **The heatmap visualization is valuable:** Even though accuracy wasn't
   perfect, the probability distribution provides insight into model confidence
   and failure modes.

5. **Single/10-piece overfit still works:** The model can memorize small sets,
   confirming the learning pipeline works. The difficulty is in scaling to 950
   classes.

### Why Classification is Harder

1. **Output dimensionality:** Predicting 950 logits requires the 64-dim feature
   vector to encode enough information to distinguish all 950 classes.
   Regression only needs 2 outputs.

2. **Mutual exclusivity:** Cross-entropy enforces that probabilities sum to 1. A
   small error in one logit affects the entire distribution. MSE on coordinates
   treats x and y independently.

3. **No notion of "close":** Classification treats predicting cell 100 vs cell
   101 (neighbors) the same as predicting cell 100 vs cell 500 (far apart).
   Regression naturally captures proximity.

### Implications

1. **Regression is more efficient for this task:** For memorization/overfitting,
   coordinate regression works better with a small network.

2. **Classification may need more capacity:** Either more features before the
   classification head, or a hierarchical approach (predict region, then cell
   within region).

3. **Alternative: Soft labels or ordinal classification:** Could use soft
   targets that give partial credit to nearby cells, combining benefits of both
   approaches.

### Recommended Next Steps

1. **Try larger feature dimension:** Increase from 64 to 256 features before
   classification head
2. **Hierarchical classification:** First predict coarse region (e.g., 4x4),
   then fine cell
3. **Hybrid approach:** Use classification for coarse location + regression for
   refinement
4. **Continue with regression:** Given these results, proceed with regression
   for the main model while keeping classification as a potential alternative

## Notes

- Cell indices are deterministic from coordinates - no ambiguity
- Using the same 64-dimensional feature space before the classification head
- Softmax temperature could be tuned if needed (default T=1)
- Class imbalance is not an issue (each cell has exactly one piece)
