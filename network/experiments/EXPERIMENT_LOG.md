# Experiment Log

A chronological summary of experiments conducted for the puzzle piece localization project.

---

## Exp 1: Baseline Sanity Check
**Date:** December 2025
**Status:** PASSED

Verified that the neural network training pipeline works at a fundamental level. Created a minimal experiment with synthetic 64x64 images containing colored squares on gray backgrounds. A tiny ~15K parameter CNN successfully learned to predict center coordinates with MSE loss. Single-sample memorization converged in 28 steps, and 10-sample memorization in 452 epochs. This confirmed gradients flow correctly and the training loop is functional, ruling out pipeline issues as the cause of the main model's stalling losses.

---

## Exp 2: Single Puzzle Overfit
**Date:** December 2025
**Status:** PASSED

Tested whether a CNN can memorize piece-to-position mappings for real puzzle textures. Using a 23K parameter piece-only encoder on puzzle_001 (950 pieces), the model achieved MSE loss of 0.007 after 500 epochs, successfully memorizing all piece positions. This proved that texture-to-position learning works for real images, and that the main model's issues were not due to data complexity but likely architectural choices.

---

## Exp 3: Cell Classification
**Date:** December 2025
**Status:** PARTIAL SUCCESS

Reframed puzzle piece localization as 950-class classification instead of coordinate regression. While single and 10-piece overfit tests passed, full training only reached 83.6% top-1 accuracy (target: >95%) and 95.4% top-5 accuracy (target: >99%). Classification proved harder than regression with the same 64-dim feature space because 950 output classes require more discriminative features than 2 regression outputs. Key insight: the bottleneck was feature dimensionality, not the classification head.

---

## Exp 4: Larger Backbone
**Date:** December 2025
**Status:** SUCCESS

Increased backbone capacity from 64-dim to 256-dim features (632K params vs 85K). This dramatically improved classification from 83.6% to 99.3% top-1 accuracy and achieved 100% top-5 accuracy. The experiment confirmed that feature dimensionality was indeed the bottleneck for classification. Key insight: invest in backbone capacity rather than classification head complexity.

---

## Exp 5: Cross-Puzzle Generalization
**Date:** December 2025
**Status:** FAILED

First attempt at cross-puzzle generalization using a dual-input architecture (piece + puzzle image). Trained on puzzle_001, tested on puzzle_002. Training accuracy reached 11.5%, but test accuracy was 0.11% (random chance). The model memorized puzzle-specific patterns instead of learning generalizable matching. Identified issues: resolution mismatch (cells only ~7x10 pixels in 256x256 puzzle), single-puzzle training providing no variation.

---

## Exp 6: Multi-Puzzle High Resolution
**Date:** December 2025
**Status:** FAILED

Addressed exp5 issues by training on 5 puzzles simultaneously with 512x512 resolution. Results were worse: only 2.4% training accuracy and 0% test accuracy. The 950-class problem was too difficult even to overfit. Multi-puzzle training without better feature extraction made the task harder, not easier. Concluded that end-to-end 950-cell classification may be fundamentally limited for learning generalizable matching.

---

## Exp 7: Coarse Regression with Spatial Correlation
**Date:** December 2025
**Status:** PARTIAL SUCCESS (67% accuracy)

Simplified to 2x2 quadrant prediction (4 classes) with coordinate regression and 100+ puzzle diversity. Initially achieved 46% test accuracy with frozen MobileNetV3-Small backbone. After discovering a critical architectural flaw (global average pooling discarded spatial information), added spatial correlation module. Test accuracy jumped to 67%, proving cross-puzzle generalization IS achievable with proper template matching architecture. Just 3% short of the 70% target.

---

## Exp 8: High Resolution Correlation
**Date:** December 2025
**Status:** NEGATIVE RESULT

Tested whether 512x512 puzzle resolution (vs 256x256) would improve the 67% result from exp7. Surprisingly, test accuracy dropped to 59% (-8%). Higher resolution caused attention diffusion across 256 spatial locations (vs 64), making correlation noisier. For coarse quadrant prediction, 256x256 provides sufficient abstraction. Lesson: resolution should match task granularity.

---

## Exp 9: Fine-tune Backbone
**Date:** December 2025
**Status:** SUCCESS (93% accuracy)

Fine-tuned the MobileNetV3-Small backbone instead of keeping it frozen, using differential learning rates (1e-4 for backbone, 1e-3 for heads). Test accuracy jumped from 67% (frozen) to 93% (fine-tuned), a +26 percentage point improvement. The model exceeded the 70% target by epoch 10. This conclusively demonstrated that task-specific features are essential for cross-puzzle matching. The combination of spatial correlation (exp7) + fine-tuning (exp9) achieved strong generalization.

---

## Exp 10: Add Rotation Prediction
**Date:** December 2025
**Status:** FAILED (architectural flaw + overfitting)

Extended exp9 to predict both position AND rotation (4 classes). Training reached 98%+ on both tasks, but test accuracy dropped to 78% quadrant (was 93%) and 61% rotation. Root cause analysis revealed a **fundamental architectural flaw**: the rotation head receives globally-pooled piece features, which destroys the spatial information needed to distinguish rotations. After pooling, a piece with "sky at top, grass at bottom" becomes nearly identical to its 180° rotation — explaining why 0° vs 180° and 90° vs 270° had the highest confusion. Secondary issue: 16 samples per puzzle (4 quadrants × 4 rotations) encouraged memorization. Critical fix needed: use a spatial rotation head that preserves the piece's feature map structure.

---

## Exp 11: Spatial Rotation Head
**Date:** December 2025
**Status:** FAILED (No improvement)

Attempted to fix Exp 10's perceived "pooling" flaw by using a Spatial Rotation Head that preserved piece feature maps. Hypothesized that preserving spatial structure ("sky at top" vs "sky at bottom") would solve rotation. Results were slightly worse than Exp 10 (73% quad, 60% rot). The failure revealed the **true root cause**: rotation is not an intrinsic property of the piece but a relationship between piece and puzzle. Both Exp 10 and 11 failed because the rotation head only saw the piece, ignoring the puzzle context. A "rotation correlation" approach is needed.

---

## Summary Table

| Exp | Focus | Test Result | Key Finding |
|-----|-------|-------------|-------------|
| 1 | Baseline sanity | PASS | Training pipeline works |
| 2 | Single puzzle memorization | PASS (0.007 MSE) | Texture learning works |
| 3 | Cell classification | 83.6% accuracy | Feature bottleneck identified |
| 4 | Larger backbone | 99.3% accuracy | 256-dim features solve classification |
| 5 | Cross-puzzle (single train) | 0.1% (random) | Single puzzle doesn't generalize |
| 6 | Multi-puzzle 950-class | 0% test | 950 classes too hard |
| 7 | Coarse regression | 67% | Spatial correlation breakthrough |
| 8 | High resolution | 59% | Resolution not the answer |
| 9 | Fine-tune backbone | **93%** | Task-specific features essential |
| 10 | Add rotation | 78% quad / 61% rot | Global pooling destroys rotation info |
| 11 | Spatial rotation head | 73% quad / 60% rot | Rotation requires puzzle context |

---

## Current Best Approach

**Architecture:** DualInputRegressorWithCorrelation (exp7) + fine-tuned MobileNetV3-Small backbone (exp9)
**Task:** 2x2 quadrant prediction with coordinate regression
**Test Accuracy:** 93%
**Next Steps:** Rotation correlation (compare piece TO puzzle), joint position-rotation correlation, finer grids (3x3, 4x4)
