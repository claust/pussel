# Experiment 21: Masked Rotation Correlation

## Objective

Fix the rotation prediction failure from exp20 by using **mask-based correlation**.
In exp20, rotation correlation achieved 95% on training but only 25% (random) on test,
indicating severe overfitting. The hypothesis is that irregular Bezier edges introduce
noise that breaks rotation matching.

## Hypothesis

The rotation correlation module compares piece features to puzzle region features.
With realistic pieces, the comparison includes:
- **Actual puzzle texture** (the content we want to match)
- **Black background** (fills where tabs/blanks create gaps)
- **Edge artifacts** (Bezier curve boundaries)

By masking out non-puzzle pixels, we can:
1. Compare only the actual puzzle texture content
2. Ignore black background regions that don't exist in the puzzle
3. Reduce edge artifacts that differ between piece and puzzle

## Key Changes from Exp20

| Aspect | Exp 20 | Exp 21 |
|--------|--------|--------|
| Rotation correlation | Full feature map | Masked feature map |
| Mask source | None | Derived from black background |
| Dataset | exp20's realistic_4x4 | Same (reused) |
| Architecture | Same backbone | Same backbone |

## Mask Generation

Since exp20 pieces were saved with black (0,0,0) background, we derive masks at runtime:

```python
# In dataset.py
def generate_mask(piece_tensor: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Generate mask from piece by detecting non-black pixels.

    Args:
        piece_tensor: Piece image [3, H, W] with values in [0, 1].
        threshold: Pixels with mean RGB < threshold are considered background.

    Returns:
        Binary mask [1, H, W] where 1 = puzzle content, 0 = background.
    """
    # Mean across RGB channels
    mean_rgb = piece_tensor.mean(dim=0, keepdim=True)  # [1, H, W]
    # Black pixels have mean ~0, puzzle content has mean > threshold
    mask = (mean_rgb > threshold).float()
    return mask
```

## Masked Rotation Correlation

The rotation correlation module is modified to:

1. Generate mask from piece features (using the RGB input, not feature maps)
2. Resize mask to feature map spatial dimensions
3. Apply mask when computing correlation between rotated piece and puzzle region

```python
# Pseudo-code for masked rotation correlation
for rotation in [0, 90, 180, 270]:
    rotated_piece_features = rotate(piece_features, rotation)
    rotated_mask = rotate(mask, rotation)

    # Only compare where mask is valid
    masked_piece = rotated_piece_features * rotated_mask
    masked_puzzle = puzzle_region_features * rotated_mask

    score = similarity(masked_piece, masked_puzzle, mask=rotated_mask)
```

## Success Criteria

| Metric | Exp20 Result | Target |
|--------|--------------|--------|
| Cell accuracy | 73% | >= 73% (maintain) |
| Rotation accuracy | 25% (random) | > 50% |
| Train-test gap (rotation) | 70% | < 20% |

## Files

```
exp21_masked_rotation/
├── __init__.py           # Package exports
├── README.md             # This file
├── dataset.py            # Dataset with mask generation
├── model.py              # Model with masked rotation correlation
├── train_cuda.py         # CUDA training script
├── visualize.py          # Visualization utilities
├── outputs/              # Results (created during training)
└── runpod/               # RunPod deployment scripts
    ├── prepare_package.sh
    └── setup_and_train.sh
```

## Running the Experiment

### Local Testing (CPU/MPS)
```bash
cd network
source ../venv/bin/activate
python -m experiments.exp21_masked_rotation.train_cuda --epochs 5 --n-train 100 --n-test 20
```

### RunPod (Full Training)
```bash
# Prepare package
cd network/experiments/exp21_masked_rotation
./runpod/prepare_package.sh

# Upload to RunPod
scp -P <PORT> -i ~/.ssh/runpod_key ../../../network/runpod_package/runpod_training.tar.gz root@<IP>:/workspace/

# Run on RunPod
ssh -p <PORT> -i ~/.ssh/runpod_key root@<IP>
cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh
```

## Results

Training completed on RunPod (RTX 5090) in 4.8 hours (50 epochs).

### Final Metrics (Best Epoch: 48)

| Metric | Exp20 | Exp21 | Target | Status |
|--------|-------|-------|--------|--------|
| Test Cell Accuracy | 73% | **73.7%** | ≥73% | ✅ PASS |
| Test Rotation Accuracy | 25% | **24.7%** | >50% | ❌ FAIL |
| Train Rotation Accuracy | 95% | 93.6% | - | - |
| Rotation Train-Test Gap | 70% | **68.9%** | <20% | ❌ FAIL |

### Training Progression

| Epoch | Train Cell | Test Cell | Train Rot | Test Rot |
|-------|------------|-----------|-----------|----------|
| 1 | 10.1% | 13.2% | 48.9% | 24.8% |
| 10 | 43.2% | 47.3% | 83.8% | 24.3% |
| 25 | 62.9% | 65.6% | 90.5% | 24.5% |
| 40 | 69.5% | 71.8% | 92.8% | 24.5% |
| 50 | 71.9% | 73.8% | 93.6% | 24.5% |

### Visualizations

- `outputs/training_curves.png` - Loss and accuracy curves
- `outputs/test_predictions.png` - Position predictions and confusion matrices

## Conclusion

**The masked rotation correlation hypothesis FAILED.**

Despite masking out black background pixels during rotation correlation, test rotation accuracy remained at random baseline (~25%) while training accuracy reached 93.6%. This indicates:

1. **Masking doesn't solve overfitting**: The model still memorizes training piece rotations instead of learning generalizable rotation features.

2. **The problem isn't background interference**: Even when comparing only actual puzzle content, the model can't generalize rotation predictions to unseen pieces.

3. **Cell accuracy is unaffected**: Position prediction works well (73.7%), matching exp20. The mask doesn't harm position learning.

### Why Masking Failed

The rotation overfitting likely stems from:
- **Piece-specific texture patterns**: Each piece has unique internal textures that the model memorizes rather than learning orientation-invariant features.
- **Edge shape memorization**: Even with masked correlation, the irregular Bezier edge shapes are visible in the feature maps and can be memorized.
- **Fundamental approach issue**: Rotation correlation comparing piece vs puzzle features may be inherently prone to overfitting on unique piece characteristics.

### Recommended Next Steps

1. **Data augmentation for rotation**: Train with all 4 rotations of each piece as separate samples to force rotation invariance.
2. **Rotation-invariant architecture**: Use architectures designed to be rotation-equivariant (e.g., Group Equivariant CNNs).
3. **Classification instead of correlation**: Predict rotation as a classification task from piece features alone, using data augmentation to learn rotation-invariant representations.
4. **Edge removal preprocessing**: More aggressively crop pieces to remove edge regions entirely, keeping only central texture.

## Dependencies

- Same as exp20
- Generates dataset on RunPod from source puzzles
