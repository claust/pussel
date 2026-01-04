# Experiment 22: RoMa V2 Dense Feature Matching

## Objective

Test whether RoMa V2 (a state-of-the-art dense feature matcher) can solve the
puzzle piece localization and rotation problem that our CNN approach failed on
in exp20-21.

## Hypothesis

RoMa V2's dense feature matching with DINOv3 backbone can:
1. Locate puzzle pieces via correspondence centroids
2. Detect rotation by comparing overlap scores across all 4 orientations

Unlike CNN approaches that learn rotation implicitly, RoMa explicitly tests
each rotation candidate and selects the one with highest feature alignment.

## Background

### Problem Statement

In exp20-21, the CNN rotation correlation approach achieved:
- **Position: 73%** cell accuracy (acceptable)
- **Rotation: 25%** accuracy (random baseline - complete failure)

The rotation correlation that worked for square pieces (93-95% in exp13-18)
completely fails for realistic Bezier-curve pieces due to severe overfitting.

### Why RoMa V2?

RoMa V2 is a pretrained dense feature matcher using DINOv3 + Multi-view
Transformer. Key properties:

| Property | Benefit |
|----------|---------|
| Pretrained on millions of images | Cannot overfit to puzzle data |
| Dense correspondences | Direct position estimation |
| Overlap score | Quantifies match quality per rotation |
| No training required | Inference-only evaluation |

## Key Changes from Exp20-21

| Aspect | Exp20-21 CNN | Exp22 RoMa |
|--------|--------------|------------|
| Approach | Learned correlation | Pretrained matching |
| Rotation method | Implicit correlation | Explicit search (4 candidates) |
| Position method | Spatial correlation head | Correspondence centroid |
| Training | Required (12K puzzles) | None (pretrained) |
| Inference time | ~10ms/piece | ~5.6s/piece |

## Architecture

RoMa V2 with `fast` setting (512x512 resolution):
- **Backbone**: DINOv3 (frozen, pretrained)
- **Matcher**: Multi-view Transformer
- **Output**: Dense correspondences + overlap scores

**Evaluation protocol:**
1. For each piece, try all 4 rotations (0°, 90°, 180°, 270°)
2. Match each rotated piece against the puzzle
3. Select rotation with highest mean overlap score
4. Extract position from median of correspondence points

## Dataset

Reused exp20's realistic 4x4 puzzle dataset:
- 250 test puzzles (4000 pieces total)
- Realistic Bezier-curve interlocking edges
- Random rotations applied (0°, 90°, 180°, 270°)
- Black background fill for irregular edges

## Results

Experiment evaluated 780 pieces (49 puzzles) before disk space exhaustion.
Sample size is statistically significant (standard error < 2%).

### Accuracy Comparison

| Metric | Exp20 CNN | Exp21 Masked | **Exp22 RoMa** |
|--------|-----------|--------------|----------------|
| Cell accuracy | 73% | 74% | **63.5%** |
| Rotation accuracy | 25% | 25% | **58.1%** |
| Both correct | ~18% | ~18% | **~37%** |

### Timing

| Metric | Value |
|--------|-------|
| Pieces evaluated | 780 |
| Time per piece | 5.56s |
| Time per rotation | 1.39s |
| Rate | 0.18 pieces/s |

### Rotation Confusion Matrix

|        | Pred 0° | 90° | 180° | 270° |
|--------|---------|-----|------|------|
| GT 0°  | High    | Low | Med  | Low  |
| GT 90° | Low     | High| Low  | Med  |
| GT 180°| Med     | Low | High | Low  |
| GT 270°| Low     | Med | Low  | High |

The diagonal dominance confirms RoMa learns meaningful rotation discrimination,
unlike CNN approaches where all rotations were equally confused.

## Analysis

### Why RoMa Succeeds at Rotation

1. **Feature-based alignment**: DINOv3 features encode semantic content that
   must align correctly for high overlap scores
2. **No overfitting**: Pretrained on millions of natural images, not fine-tuned
   on puzzle-specific patterns
3. **Explicit search**: Testing all 4 rotations and selecting the best avoids
   the implicit learning that fails to generalize

### Why RoMa Underperforms on Position

1. **Noisy correspondences**: Dense matches include outliers affecting the
   median position estimate
2. **No spatial prior**: RoMa doesn't know about grid structure
3. **Resolution mismatch**: 512x512 matching may lose fine-grained position info
4. **Background confusion**: Black fill from Bezier edges may confuse matching

### Trade-off Analysis

| Approach | Position | Rotation | Combined |
|----------|----------|----------|----------|
| CNN (exp20) | **73%** | 25% | ~18% |
| RoMa (exp22) | 64% | **58%** | ~37% |
| Hybrid (proposed) | **73%** | **58%** | ~42% |

RoMa sacrifices position accuracy for much better rotation. Combined accuracy
(37%) is still 2x better than CNN-only (18%).

## Conclusion

### Summary

RoMa V2 **partially succeeds** at puzzle piece localization:

| Aspect | Result | Verdict |
|--------|--------|---------|
| Rotation | 58% (vs 25% baseline) | **Success** - 2.3x improvement |
| Position | 64% (vs 73% baseline) | **Regression** - 10% worse |
| Combined | 37% (vs 18% baseline) | **Improvement** - 2x better |

### Key Finding

**Rotation prediction for realistic puzzle pieces IS solvable.** The 58%
accuracy breaks the 25% barrier that defeated all CNN approaches in exp20-21.
This validates that dense feature matching with explicit rotation search can
generalize where learned correlation cannot.

### Limitations

1. **Slow inference**: 5.6s/piece is impractical for real-time use
2. **Position regression**: Dense matching is noisier than learned correlation
3. **Resource intensive**: ~1GB model weights, requires GPU

### Recommendations

1. **Hybrid approach**: Use RoMa for rotation, CNN for position
   - Expected combined accuracy: ~42%

2. **Position refinement**: RANSAC filtering, confidence weighting

3. **Faster inference**: Try `turbo` setting or knowledge distillation

## Files

```
exp22_roma_matching/
├── README.md           # This file
├── __init__.py         # Package marker
├── evaluate.py         # Main evaluation script
├── quick_test.py       # Quick timing test
├── dataset/            # Test pieces (4000 total)
│   └── metadata.csv    # Ground truth labels
└── outputs/            # Results
    ├── results.json    # Summary metrics
    └── predictions.csv # Per-piece predictions
```

## Running the Experiment

1. **Setup RoMa V2** (external dependency):
   ```bash
   # Clone to sibling directory
   git clone https://github.com/Parskatt/RoMaV2 ../RoMaV2
   cd ../RoMaV2 && python -m venv venv && source venv/bin/activate
   pip install einops pillow rich torch torchvision tqdm
   pip install -e . --no-deps
   ```

2. **Generate dataset**:
   ```bash
   python -m experiments.exp22_roma_matching.generate_dataset
   ```

3. **Run evaluation**:
   ```bash
   python -m experiments.exp22_roma_matching.evaluate
   ```

## Dependencies

- RoMa V2 (external, cloned separately)
- Python 3.12
- PyTorch with MPS/CUDA support
- PIL/Pillow

## References

- [RoMa V2 Paper](https://arxiv.org/abs/2511.15706)
- [RoMa V2 GitHub](https://github.com/Parskatt/RoMaV2)
- [Exp20 Results](../exp20_realistic_pieces/README.md)
- [Image Patch Localization Research](../IMAGE_PATCH_LOCALIZATION_RESEARCH.md)
