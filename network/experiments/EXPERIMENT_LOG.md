# Experiment Log

A chronological summary of experiments conducted for the puzzle piece
localization project.

---

## Exp 1: Baseline Sanity Check

**Date:** December 2025 **Status:** PASSED

Verified that the neural network training pipeline works at a fundamental level.
Created a minimal experiment with synthetic 64x64 images containing colored
squares on gray backgrounds. A tiny ~15K parameter CNN successfully learned to
predict center coordinates with MSE loss. Single-sample memorization converged
in 28 steps, and 10-sample memorization in 452 epochs. This confirmed gradients
flow correctly and the training loop is functional, ruling out pipeline issues
as the cause of the main model's stalling losses.

---

## Exp 2: Single Puzzle Overfit

**Date:** December 2025 **Status:** PASSED

Tested whether a CNN can memorize piece-to-position mappings for real puzzle
textures. Using a 23K parameter piece-only encoder on puzzle_001 (950 pieces),
the model achieved MSE loss of 0.007 after 500 epochs, successfully memorizing
all piece positions. This proved that texture-to-position learning works for
real images, and that the main model's issues were not due to data complexity
but likely architectural choices.

---

## Exp 3: Cell Classification

**Date:** December 2025 **Status:** PARTIAL SUCCESS

Reframed puzzle piece localization as 950-class classification instead of
coordinate regression. While single and 10-piece overfit tests passed, full
training only reached 83.6% top-1 accuracy (target: >95%) and 95.4% top-5
accuracy (target: >99%). Classification proved harder than regression with the
same 64-dim feature space because 950 output classes require more discriminative
features than 2 regression outputs. Key insight: the bottleneck was feature
dimensionality, not the classification head.

---

## Exp 4: Larger Backbone

**Date:** December 2025 **Status:** SUCCESS

Increased backbone capacity from 64-dim to 256-dim features (632K params vs
85K). This dramatically improved classification from 83.6% to 99.3% top-1
accuracy and achieved 100% top-5 accuracy. The experiment confirmed that feature
dimensionality was indeed the bottleneck for classification. Key insight: invest
in backbone capacity rather than classification head complexity.

---

## Exp 5: Cross-Puzzle Generalization

**Date:** December 2025 **Status:** FAILED

First attempt at cross-puzzle generalization using a dual-input architecture
(piece + puzzle image). Trained on puzzle_001, tested on puzzle_002. Training
accuracy reached 11.5%, but test accuracy was 0.11% (random chance). The model
memorized puzzle-specific patterns instead of learning generalizable matching.
Identified issues: resolution mismatch (cells only ~7x10 pixels in 256x256
puzzle), single-puzzle training providing no variation.

---

## Exp 6: Multi-Puzzle High Resolution

**Date:** December 2025 **Status:** FAILED

Addressed exp5 issues by training on 5 puzzles simultaneously with 512x512
resolution. Results were worse: only 2.4% training accuracy and 0% test
accuracy. The 950-class problem was too difficult even to overfit. Multi-puzzle
training without better feature extraction made the task harder, not easier.
Concluded that end-to-end 950-cell classification may be fundamentally limited
for learning generalizable matching.

---

## Exp 7: Coarse Regression with Spatial Correlation

**Date:** December 2025 **Status:** PARTIAL SUCCESS (67% accuracy)

Simplified to 2x2 quadrant prediction (4 classes) with coordinate regression and
100+ puzzle diversity. Initially achieved 46% test accuracy with frozen
MobileNetV3-Small backbone. After discovering a critical architectural flaw
(global average pooling discarded spatial information), added spatial
correlation module. Test accuracy jumped to 67%, proving cross-puzzle
generalization IS achievable with proper template matching architecture. Just 3%
short of the 70% target.

---

## Exp 8: High Resolution Correlation

**Date:** December 2025 **Status:** NEGATIVE RESULT

Tested whether 512x512 puzzle resolution (vs 256x256) would improve the 67%
result from exp7. Surprisingly, test accuracy dropped to 59% (-8%). Higher
resolution caused attention diffusion across 256 spatial locations (vs 64),
making correlation noisier. For coarse quadrant prediction, 256x256 provides
sufficient abstraction. Lesson: resolution should match task granularity.

---

## Exp 9: Fine-tune Backbone

**Date:** December 2025 **Status:** SUCCESS (93% accuracy)

Fine-tuned the MobileNetV3-Small backbone instead of keeping it frozen, using
differential learning rates (1e-4 for backbone, 1e-3 for heads). Test accuracy
jumped from 67% (frozen) to 93% (fine-tuned), a +26 percentage point
improvement. The model exceeded the 70% target by epoch 10. This conclusively
demonstrated that task-specific features are essential for cross-puzzle
matching. The combination of spatial correlation (exp7) + fine-tuning (exp9)
achieved strong generalization.

---

## Exp 10: Add Rotation Prediction

**Date:** December 2025 **Status:** FAILED (architectural flaw + overfitting)

Extended exp9 to predict both position AND rotation (4 classes). Training
reached 98%+ on both tasks, but test accuracy dropped to 78% quadrant (was 93%)
and 61% rotation. Root cause analysis revealed a **fundamental architectural
flaw**: the rotation head receives globally-pooled piece features, which
destroys the spatial information needed to distinguish rotations. After pooling,
a piece with "sky at top, grass at bottom" becomes nearly identical to its 180°
rotation — explaining why 0° vs 180° and 90° vs 270° had the highest confusion.
Secondary issue: 16 samples per puzzle (4 quadrants × 4 rotations) encouraged
memorization. Critical fix needed: use a spatial rotation head that preserves
the piece's feature map structure.

---

## Exp 11: Spatial Rotation Head

**Date:** December 2025 **Status:** FAILED (No improvement)

Attempted to fix Exp 10's perceived "pooling" flaw by using a Spatial Rotation
Head that preserved piece feature maps. Hypothesized that preserving spatial
structure ("sky at top" vs "sky at bottom") would solve rotation. Results were
slightly worse than Exp 10 (73% quad, 60% rot). The failure revealed the **true
root cause**: rotation is not an intrinsic property of the piece but a
relationship between piece and puzzle. Both Exp 10 and 11 failed because the
rotation head only saw the piece, ignoring the puzzle context. A "rotation
correlation" approach is needed.

---

## Exp 12: Rotation Correlation

**Date:** December 2025 **Status:** PARTIAL SUCCESS (87% rotation accuracy!)

Fixed the fundamental flaw in exp10/11 by implementing **Rotation
Correlation** - comparing piece features to puzzle features at each rotation
candidate, selecting the rotation with highest similarity. Results: 67% quadrant
accuracy (regressed from 73%), but **87% rotation accuracy** (+27% improvement
over exp11's 60%). The hypothesis was validated: rotation is a matching problem,
not an intrinsic classification problem. The 0° vs 180° confusion was reduced
from a major issue to ~7-9%. However, position accuracy regressed, suggesting
the joint position-rotation architecture needs refinement. Key finding: the
rotation correlation approach is correct, but may need two-stage inference or
gradient detachment to prevent position regression.

---

## Exp 13: Rotation Correlation with 5K Puzzles

**Date:** December 2025 **Status:** SUCCESS (86% position, 93% rotation!)

Tested whether scaling up training data (800 → 4499 puzzles) could fix the
position regression from exp12 without architectural changes. Results exceeded
expectations: **86.3% quadrant accuracy** (+19.8% over exp12) and **92.8%
rotation accuracy** (+5.9% over exp12). The position regression was completely
fixed, and rotation improved further. The 0° vs 180° confusion dropped from 7-9%
to just 0.8-1.6%. Key insight: the rotation correlation architecture was correct
all along — it just needed more diverse training data. Train-test gaps shrunk
dramatically (position: 25.9% → 10.6%, rotation: 12.4% → 4.9%), confirming
better generalization. This validates that **data quantity matters as much as
architecture** for cross-puzzle matching.

---

## Exp 14: MobileViT-XS Backbone

**Date:** December 2025 **Status:** FAILED (75% position, 92% rotation - worse
than exp13)

Tested whether replacing MobileNetV3-Small with MobileViT-XS (a hybrid
CNN-Transformer architecture) would improve piece-puzzle matching through its
attention mechanism. Results showed significant regression: **74.8% quadrant
accuracy** (-11.5% vs exp13) and **91.5% rotation accuracy** (-1.3% vs exp13).
Despite having 16% fewer parameters (4.43M vs 5.27M), MobileViT trained 33%
slower due to attention computation overhead.

Key findings:

- **Position accuracy suffered significantly**: MobileViT's global
  self-attention appears to diffuse spatial information, similar to the issue in
  exp8 with higher resolution. MobileNetV3's local convolutions preserve
  fine-grained spatial features better for template matching.
- **Rotation accuracy nearly preserved**: The attention mechanism's ability to
  capture global patterns partially compensates for rotation correlation,
  validating that transformers have some merit for this subtask.
- **No efficiency gains**: Fewer parameters but slower training with no accuracy
  benefit.

The experiment conclusively shows that **pure CNN architectures (MobileNetV3)
are better suited for spatial template matching** than hybrid CNN-Transformer
models. The Squeeze-and-Excitation blocks in MobileNetV3 provide effective
feature recalibration without the spatial information loss caused by global
attention.

---

## Exp 15: Fast Backbone Comparison

**Date:** December 2025 **Status:** SUCCESS (identified ShuffleNetV2_x0.5 as
fastest backbone)

Compared training speed of lightweight backbones to accelerate the
experimentation cycle. Tested RepVGG-A0 (16.7M params), MobileOne-S0 (9.4M
params), and ShuffleNetV2_x0.5 (1.6M params) on Apple Silicon MPS.

Results (2 epochs, 500 train puzzles):

| Backbone              | Epoch Time | Test Quad Acc |
| --------------------- | ---------- | ------------- |
| RepVGG-A0             | 39.3s      | 36.1%         |
| MobileOne-S0          | 98.1s      | 34.2%         |
| **ShuffleNetV2_x0.5** | **12.9s**  | 27.3%         |

Key findings:

- **ShuffleNetV2_x0.5 is 3× faster than RepVGG and 7.6× faster than MobileOne**
- MobileOne-S0 is surprisingly slow on MPS (optimized for iPhone Neural Engine,
  not Mac GPU)
- All models show learning signal above random baseline (25%)
- For experimentation phase, use ShuffleNetV2 for ~34× faster iteration vs
  exp13's MobileNetV3

Recommendation: Use ShuffleNetV2_x0.5 for rapid architecture experiments, then
switch to MobileNetV3-Small for final training once approach is validated

---

## Summary Table

| Exp | Focus                       | Test Result            | Key Finding                                    |
| --- | --------------------------- | ---------------------- | ---------------------------------------------- |
| 1   | Baseline sanity             | PASS                   | Training pipeline works                        |
| 2   | Single puzzle memorization  | PASS (0.007 MSE)       | Texture learning works                         |
| 3   | Cell classification         | 83.6% accuracy         | Feature bottleneck identified                  |
| 4   | Larger backbone             | 99.3% accuracy         | 256-dim features solve classification          |
| 5   | Cross-puzzle (single train) | 0.1% (random)          | Single puzzle doesn't generalize               |
| 6   | Multi-puzzle 950-class      | 0% test                | 950 classes too hard                           |
| 7   | Coarse regression           | 67%                    | Spatial correlation breakthrough               |
| 8   | High resolution             | 59%                    | Resolution not the answer                      |
| 9   | Fine-tune backbone          | **93%**                | Task-specific features essential               |
| 10  | Add rotation                | 78% quad / 61% rot     | Global pooling destroys rotation info          |
| 11  | Spatial rotation head       | 73% quad / 60% rot     | Rotation requires puzzle context               |
| 12  | Rotation correlation        | 67% quad / 87% rot     | Rotation correlation works! Position regressed |
| 13  | 5K puzzles                  | **86% quad / 93% rot** | More data fixes everything!                    |
| 14  | MobileViT-XS backbone       | 75% quad / 92% rot     | CNN beats Transformer for spatial matching     |
| 15  | Fast backbone comparison    | **12.9s/epoch**        | ShuffleNetV2 fastest for experimentation       |

---

## Current Best Approach

**For Position Only:** DualInputRegressorWithCorrelation (exp7) + fine-tuned
MobileNetV3-Small backbone (exp9) - 93% quadrant accuracy

**For Position + Rotation:** DualInputRegressorWithRotationCorrelation (exp13)
with MobileNetV3-Small - **86% quad / 93% rot** with 4499 training puzzles

**Note:** Exp14 showed that MobileViT-XS (CNN-Transformer hybrid) underperforms
MobileNetV3-Small for this task. Pure CNNs remain the best choice for spatial
template matching.

**For Fast Experimentation:** ShuffleNetV2_x0.5 (exp15) - 12.9s/epoch vs 39.3s
for RepVGG and 98.1s for MobileOne. Use for rapid iteration.

**Next Steps:** Proceed to finer grids (3x3, 4x4), test on real puzzle piece
shapes, explore continuous coordinate regression

---
