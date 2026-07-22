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

**Date:** December 2025 **Status:** FAILED (75% position, 92% rotation)

Tested whether MobileViT-XS (a hybrid CNN-Transformer) would improve matching
through its attention mechanism. Results showed significant regression:
**74.8% quadrant accuracy** (-11.5% vs exp13) and **91.5% rotation** (-1.3%).
Despite 16% fewer parameters (4.43M vs 5.27M), it trained 33% slower due to
attention overhead. Position accuracy suffered because global self-attention
diffuses spatial information, while rotation was nearly preserved since
attention helps with global pattern matching. Conclusion: pure CNN architectures
(MobileNetV3) are better suited for spatial template matching than hybrids.

---

## Exp 15: Fast Backbone Comparison

**Date:** December 2025 **Status:** SUCCESS (ShuffleNetV2_x0.5 fastest)

Compared training speed of lightweight backbones on Apple Silicon MPS to
accelerate experimentation. ShuffleNetV2_x0.5 was the clear winner at **12.9s
per epoch** — 3× faster than RepVGG-A0 (39.3s) and 7.6× faster than MobileOne-S0
(98.1s). MobileOne was surprisingly slow on MPS because it's optimized for
iPhone Neural Engine, not Mac GPU. All backbones showed learning signal above
random baseline (25%). For rapid iteration, ShuffleNetV2 enables ~34× faster
experimentation vs MobileNetV3-Small from exp13, then switch back for final
training once the approach is validated.

---

## Exp 16: 3x3 Grid Position Prediction

**Date:** December 2025 **Status:** PARTIAL SUCCESS (39% cell, 61% rotation)

First attempt at scaling from 2x2 quadrants (4 positions) to 3x3 grid (9
positions). Used ShuffleNetV2_x0.5 backbone for fast experimentation with 500
training puzzles. Results: **39.1% cell accuracy** (3.5× random baseline of
11.1%) and **60.9% rotation**. However, severe overfitting emerged — rotation
had a 30.1% train-test gap, and test accuracy plateaued around epoch 60-70.
Compared to exp13's 86.3% quadrant accuracy with 4,499 puzzles, the drop to
39.1% with only 500 puzzles confirms the architecture works but needs
significantly more training data for the harder 9-class task.

---

## Exp 17: 3x3 Grid with More Data

**Date:** December 2025 **Status:** SUCCESS (81% cell, 93% rotation!)

Scaled up training data from 500 to 10,000 puzzles to reduce overfitting
observed in exp16. Tested two configurations: 1 cell/puzzle and 9 cells/puzzle.
Surprisingly, 1 cell/puzzle made overfitting *worse* (31% gap vs 11%) because
the model never sees all cells of a puzzle together. With 9 cells/puzzle,
results were dramatically better: **80.9% cell accuracy** (+41.8% vs exp16) and
**92.7% rotation accuracy** (+31.8% vs exp16). Train-test gap dropped to -1.4%
(test slightly exceeds train), indicating no overfitting. Key insight: **how**
you use data matters as much as **how much** — must see all cells per puzzle
together for proper feature learning.

---

## Exp 18: 3x3 Grid with 20K Puzzles

**Date:** December 2025 **Status:** SUCCESS (82% cell, 95% rotation - NEW BEST!)

Doubled training data from 10,000 to 20,000 puzzles to test continued scaling
benefits. Also optimized training with larger batch size (128 vs 64) and 2x
learning rates (linear scaling rule). Results: **82.2% cell accuracy** (+1.3%
vs exp17) and **95.1% rotation accuracy** (+2.4% vs exp17). Train-test gaps
remained small (1.8% cell, 1.2% rotation), confirming no overfitting. Rotation
benefited more from additional data than position. Training took 21.2 hours on
M4 Mac Mini (~12.7 min/epoch). This validates that data scaling continues to
help at 20K puzzles.

---

## Exp 19: Siamese Architecture

**Date:** December 2025 **Status:** FAILED (79% cell, 92% rotation)

Tested whether a Siamese architecture (single shared backbone for both piece and
puzzle) could match exp18's dual-backbone performance while reducing parameters.
Results were worse: 79.4% cell accuracy (-2.8% vs exp18) and 91.7% rotation
(-3.4% vs exp18), despite 21.6% fewer parameters. The train-test gap also
increased from 1.8% to 4.5%, indicating worse generalization. The hypothesis
that weight sharing would help was disproven — piece (128×128) and puzzle
(256×256) images are too different to benefit from shared feature extraction.
Siamese networks excel when inputs are similar (face verification, signature
matching), but here specialized encoders outperform weight sharing. Dual
backbone from exp18 remains the best approach.

---

## Exp 20: 4x4 Grid with Realistic Pieces

**Date:** December 2025 **Status:** PARTIAL SUCCESS (73% cell, 25% rotation)

First attempt at scaling to 4×4 grid (16 cells) using **realistic puzzle pieces**
with Bezier-curve interlocking edges (tabs and blanks) instead of square-cut
pieces. Generated 12,000 puzzles with the new piece generator. Training ran on
RunPod RTX 4090 (~2.9 hours vs estimated 33 hours on Mac M4). Results:
**72.9% cell accuracy** (11.7× random baseline of 6.25%) but only **24.8%
rotation accuracy** (at random baseline). Position prediction successfully
scaled from 3×3 to 4×4, exceeding the 50% target. However, rotation prediction
**completely failed to generalize** — train rotation reached 95.1% while test
stayed flat at 24.8% from epoch 1, indicating severe overfitting. The rotation
correlation module that worked well for square pieces (exp13-19) does not
transfer to realistic pieces. Hypothesis: the irregular Bezier edges make
rotation matching fundamentally harder because the piece silhouette changes
unpredictably with rotation. **Key insight:** Position and rotation may need
completely different approaches for realistic pieces — position via texture
correlation (works), rotation perhaps via edge-shape matching or
rotation-invariant features.

> **UPDATE (July 2026):** The rotation result was a test-label bug, not a
> model failure. Re-evaluating this same checkpoint with fixed labels gives
> **94.6% rotation** — see "Exp 20 Re-Evaluation" below.

---

## Exp 21: Masked Rotation Correlation

**Date:** December 2025 **Status:** FAILED (74% cell, 25% rotation)

Attempted to fix exp20's rotation failure by masking out the black background
during feature extraction, hypothesizing that black pixels from irregular Bezier
edges interfere with rotation correlation. Results: 73.9% cell accuracy
(unchanged) and 24.7% rotation accuracy (still random baseline). Masking had
zero effect. The rotation correlation approach fundamentally cannot learn
rotation for realistic pieces because irregular Bezier edges create highly
variable piece silhouettes that texture-based correlation cannot capture.

> **UPDATE (July 2026):** Invalid — the test-label bug capped rotation at
> 25% for any model, so this experiment could not have measured an effect.
> See "Exp 20 Re-Evaluation" below.

---

## Exp 22: RoMa V2 Dense Feature Matching

**Date:** January 2026 **Status:** PARTIAL SUCCESS (64% cell, 58% rotation)

Tested RoMa V2, a pretrained dense feature matcher (DINOv3 + Multi-view
Transformer), as an alternative to CNN rotation correlation. For each piece,
tried all 4 rotations and selected the one with highest overlap score, then
extracted position from correspondence centroids. On 780 pieces: 63.5% cell
accuracy (regressed from 73% CNN) but 58.1% rotation accuracy (2.3× improvement
over 25% baseline!). This is the first approach to beat random on Bezier-edge
pieces. The rotation breakthrough comes from RoMa's pretrained features that
cannot overfit to puzzle data, combined with explicit rotation search rather
than implicit learning. Position regressed due to noisy correspondences.
Recommendation: hybrid approach using RoMa for rotation (58%) and CNN for
position (73%), expecting ~42% combined accuracy.

> **UPDATE (July 2026):** The CNN's 25% rotation baseline was a test-label
> bug; the corrected CNN scores 94.6% rotation and 72.9% cell — better than
> RoMa on both metrics. The hybrid recommendation is withdrawn. See "Exp 20
> Re-Evaluation" below.

---

## Critical Review of Experiments 1–22

**Date:** July 2026 **Status:** REVIEW (full details in
[CRITICAL_REVIEW.md](CRITICAL_REVIEW.md))

Audited all experiments against their code and result JSONs. What holds
up: spatial correlation for position (exp7), rotation-as-matching
(exp12), and data scaling (exp13, the cleanest experiment in the
series). What doesn't:

1. **exp20's test set has broken rotation labels** — the rotation baked
   into each generated piece is discarded when building test samples
   (`exp20_realistic_pieces/dataset.py:312-320`), capping test rotation
   accuracy at 25% for any model, including a perfect one. The dead-flat
   24.7% curve was a measurement bug, not a model failure. This
   invalidates exp20's rotation conclusion, all of exp21 (whose code is
   also missing from the repo), and exp22's RoMa-vs-CNN comparison,
   which scored RoMa against correct labels and the CNN against broken
   ones.
2. **All experiments measure a toy surrogate**: pieces are pixel-exact
   crops of the same digital image used as puzzle input — no camera, no
   lighting gap, no background. No classical baseline (NCC, SIFT) was
   ever run to establish a floor.
3. **No validation set anywhere**; the test set drove checkpoint
   selection and design decisions from exp7 onward.
4. **Key comparisons were confounded or used wrong baselines**: exp17's
   "cells together" insight is confounded with 9× more data; exp14/15
   compared against a fictional exp13 baseline (12 h / 5.27M params vs
   the logged 4 h / 2.52M) — the "34× faster" ShuffleNet claim is ~1.24×
   when normalized, so the backbone choice since exp16 rests on an
   error. exp5/exp6's negative results were invalid by construction.

**Immediate next steps:** fix the exp20 test labels and re-evaluate the
existing checkpoint before any further RoMa/hybrid work; add classical
baselines; introduce a proper train/val/test split; build a small
photographed-piece test set as the real benchmark.

---

## Exp 20 Re-Evaluation: Fixed Test Rotation Labels

**Date:** July 2026 **Status:** SUCCESS (72.9% cell, **94.6% rotation**)

Fixed the exp20 test-label bug from the critical review and re-evaluated
the original (unmodified) RunPod checkpoint. `RealisticPieceTestDataset`
now composes the rotation baked into each piece PNG with the applied
test rotation (`(base + applied) % 360`), matching the training path.
The original test split was reproduced exactly: the generator seeds each
puzzle with `42 + source_index`, so the 1,200 test puzzles
(seed-42 shuffle of the 11,998 generated puzzles, indices 10798+) were
regenerated deterministically (`regenerate_test_split.py`) using the
December 2025 `puzzle_shapes` (commit 6b61eb7 — the library's edge
geometry changed in January 2026, after the dataset was generated).

Results on all 76,800 test samples (1,200 puzzles × 16 cells × 4
rotations): **cell accuracy 72.9%** — matching the original run to the
decimal, confirming the regenerated test set is faithful — and
**rotation accuracy 94.6%** (94–95% for each of the four rotations;
residual errors are almost entirely 180° confusions: 0↔180 and 90↔270).
Cell AND rotation correct together: 72.2%.

Conclusions:

1. **Rotation never failed on realistic pieces.** The dead-flat 24.8%
   was purely the scrambled-label measurement; the model generalizes
   rotation nearly as well as on square pieces (95% in exp18).
2. **exp21 is moot** — it "fixed" a bug that did not exist in the model.
3. **exp22's comparison inverts:** the CNN beats RoMa on both position
   (72.9% vs 63.5%) and rotation (94.6% vs 58.1%), at a fraction of the
   inference cost (~ms vs 5.6 s/piece). The RoMa/hybrid roadmap is
   dropped.

---

## Exp 23: Classical Baselines (NCC, SIFT, ORB)

**Date:** July 2026 **Status:** BASELINE (best classical: **82.2% both** —
above the CNN's 72.2%)

Ran the non-learned baselines called for by item #2 of the critical
review on the exact exp20 re-evaluation protocol (1,200 regenerated test
puzzles × 16 pieces × 4 applied rotations = 76,800 samples, labels
composed as `(baked + applied) % 360`). Classical methods used native
resolution (256×256 puzzles, ~110–130 px piece crops, no 128×128
squashing). Full results in `exp23_classical_baselines/`.

| Method              | Cell      | Rotation  | Both      | Coverage | ms/sample |
| ------------------- | --------- | --------- | --------- | -------- | --------- |
| SIFT→NCC hybrid     | **82.9%** | 90.3%     | **82.2%** | 99.6%    | 45        |
| Masked NCC          | 77.5%     | 87.2%     | 76.9%     | 99.6%    | 77        |
| CNN (exp20 re-eval) | 72.9%     | **94.6%** | 72.2%     | 100%     | ~10 (GPU) |
| SIFT                | 43.9%     | 44.1%     | 43.6%     | 45.4%    | 3.2       |
| ORB (tuned)         | 38.4%     | 38.3%     | 37.7%     | 40.4%    | 4.8       |

NCC = `cv2.matchTemplate` (`TM_CCOEFF_NORMED`, black-background mask)
over the 4 candidate rotations. SIFT/ORB = ratio test + partial-affine
RANSAC; position from inlier centroid, rotation from the transform angle
snapped to 90°. SIFT is nearly perfect when it matches (96.7% cell /
97.3% rotation on the 45% of pieces with enough texture) — its headline
number is a coverage failure, not wrong answers — hence the hybrid:
SIFT when it matches, NCC otherwise, still zero-training.

Conclusions:

1. **The classical floor is above the CNN.** Even plain masked NCC beats
   the trained model on position and both-correct; the hybrid beats it
   by 10 points. The CNN only wins rotation (94.6% vs 90.3%).
2. **The surrogate task is largely solvable by template matching**, as
   the review suspected. Learned results on this benchmark must clear
   ~82% both-correct to demonstrate value over classical matching —
   until the benchmark is made adversarial to pixel matching
   (photometric jitter, perspective, backgrounds) or replaced with
   photographed pieces.

---

## Methodology Harness Fix: Frozen Split + Val Selection

**Date:** July 2026 **Status:** INFRASTRUCTURE (no new model results)

Fixed methodology issue #3 from the critical review for the realistic
4x4 benchmark (`exp20_realistic_pieces/`):

1. **Frozen train/val/test split**, checked in at
   `exp20_realistic_pieces/splits/realistic_4x4_v1.json` and loaded by
   all future runs (`splits.py`, `create_datasets_from_split`). The test
   portion (1,200 puzzles) is byte-identical to the original exp20
   seed-42 test split — verified against the regenerated test set's
   `test_ids.txt` — so results stay comparable to the exp20
   re-evaluation and exp23 baselines. A 600-puzzle **val** split is
   carved from the end of the original train portion (shuffle indices
   10198–10797), leaving 10,198 train puzzles; a frozen 600-puzzle
   `train_eval` subset of train supports train-metric measurement.
2. **Checkpoint selection on val, not test** (`harness.py`): the best
   checkpoint is the highest *validation* both-correct accuracy; the
   test set is never evaluated during training and is touched once per
   experiment, on the val-selected checkpoint, only when `--eval-test`
   is passed.
3. **Train metrics in eval mode**: the optimization pass records losses
   only; per-epoch train accuracies are measured with `model.eval()` on
   the frozen `train_eval` subset using the same deterministic
   all-4-rotations protocol as val/test, making the train/val gap
   apples-to-apples.

`train.py` and `train_cuda.py` were merged into a single entry point
(AMP enabled automatically on CUDA); the RunPod packaging now ships the
harness and the frozen split. Caveats: val metrics are only meaningful
for models trained on this split (the December 2025 checkpoint saw the
val puzzles during training), and new runs get 10,198 train puzzles vs
the original 10,798 (~5.6% less) — a small, acceptable price for having
a validation set at all.

---

## Exp 24: Piece / Not-a-Piece Classifier (preview gating)

**Date:** July 2026 **Status:** SUCCESS (99.1% balanced accuracy, faces at 0.000 mean probability)

Binary classifier replacing the heuristic confidence (area/aspect/skin-tone
bands) of the live piece-preview endpoint, which passed false positives — a
face in profile showed "Piece detected" — and whose skin gate cannot cover the
mobile case (cups, keys, phones on a table). A single **MobileNetV3-Small**
backbone with a small binary head (**1,000,993 params**, 128px input, ~25 ms
single-image CPU inference) scores the rembg-segmented crop: largest opaque
component composited on black, bbox-cropped with 8% margin, padded square.
Training data: 6,341 synthetic positives (exp20 Bezier-edge generator, 400
puzzles, seed 2024), 944 real positives (north-star piece photos through the
exact preview rembg pipeline), 1,090 negatives (Caltech-101 faces + 14
household/table-object categories and COCO128 scenes, all through the same
downscale-to-320 + rembg pipeline). Group-level 70/15/15 splits (positives by
puzzle, negatives by 10-file chunks per category); augmentation: full-circle
rotation, flips, color jitter, random downscale (simulates 320px preview
crops). Selected on val balanced accuracy (best epoch 9/12).

**Test (one-shot on val-selected checkpoint): 99.0% accuracy / 99.1%
balanced accuracy / AUC 0.9996** — confusion matrix tn=156 fp=1 fn=12
tp=1092. The targeted false-positive cases are dead: faces score **0.000**
mean piece-probability (40/40 rejected), household objects ≤0.02 except
scissors (0.109 mean, the one false positive), COCO scenes 0.010. Real photo
pieces: 97.9% accuracy, 0.981 mean probability. The backend
(`app/services/piece_classifier.py`) loads `outputs/checkpoint_best.pt` and
uses the probability as the preview confidence (cheap area/aspect gates stay
as pre-filters); without a checkpoint it falls back to the old heuristic
including the skin gate. Dev setting `SAVE_PREVIEW_CROPS=true` harvests
accepted preview crops as future hard negatives. Known gap: no dedicated
bare-hand negatives yet.

---

## Exp 25: North-Star Real-Photo Evaluation

**Date:** July 2026 **Status:** MILESTONE (classical hybrid **76.7% both**;
CNN collapses to 14.8%)

First evaluation of any method on real photographs — the north-star v1 set
(14 physical kids' puzzles, 236 pieces, 944 piece photos on 4 backgrounds,
grids 2x3–5x5/4x6). Protocol: pieces prepared through the exact deployed
preview path (bbox crop → rembg → black composite, 0 segmentation failures),
4 applied rotations per photo (3,776 samples), predictions binned into each
puzzle's own grid against the auto-cropped overview photo. Random = ~5.9%
cell / 25% rotation. Along the way a real dataset defect was found and fixed:
the overview JPEGs' raw pixel orientation was arbitrary (stale EXIF, wrong
for 4/14 puzzles); orientation is now *measured* by SIFT-matching the
verified-upright pieces against the overview at all 4 rotations
(unanimous: 180° for all 14) and baked into the pixels by `ingest.py`.

| Method              | Cell      | Rotation  | Both      | vs synthetic (both) |
| ------------------- | --------- | --------- | --------- | ------------------- |
| SIFT→NCC hybrid     | **77.9%** | 90.3→**89.2%** | **76.7%** | 82.2% → 76.7% (−5.5) |
| SIFT (85% coverage) | 72.6%     | 80.1%     | 71.5%     | —                   |
| NCC multi-scale     | 50.5%     | 68.9%     | 48.9%     | 76.9% → 48.9% (−28) |
| CNN + rot search    | 24.2%     | 48.1%     | 18.0%     | —                   |
| CNN (exp20)         | 22.4%     | 44.0%     | 14.8%     | 72.2% → 14.8% (−57) |

Conclusions:

1. **The critical review's headline fear is confirmed**: the CNN learned
   pixel-identical matching that does not survive a camera (barely above
   random on position; diffuse rotation confusion). All synthetic results to
   date say nothing about the real task. Test-time rotation search (review
   item #7) recovers only +3.2.
2. **Classical matching survives**: the zero-training SIFT→NCC hybrid loses
   only 5.5 points and leads the real benchmark by 28. SIFT itself is
   near-perfect when it matches (84% both on its 85% coverage); wood is the
   hardest background (69% both), `frozen_scene`/`unicorn_pink` the hardest
   puzzles (low-texture artwork where the NCC fallback is weak).
3. **The real-task bar is now 76.7% both-correct on north_star v1.** exp26
   (realism-augmented training per review item #5) must beat that — and the
   backend's current CNN path is, on real photos, far below what shipping
   the classical hybrid would give today.

---

## Exp 26: Domain Randomization (realism-augmented training)

**Date:** July 2026 **Status:** FAILED on the real task (synthetic
improved to **76.2% both**; north_star unchanged at **12.7% both**)

Retrained the exp20 architecture with domain randomization (critical-review
item #5) to attack the exp25 collapse. Pieces regenerated as RGBA and pushed
through toggleable realism augmentations — **independent** photometric
jitter on piece vs puzzle (the anti-pixel-shortcut lever), scale ±15%, mild
perspective, ±8° rotation jitter, backgrounds composited behind the piece
(black/solid/gradient/other-puzzle texture), rembg-style mask erode/dilate,
sensor noise, JPEG — while val/test stay black-composited and directly
comparable to exp20 (same frozen split, protocol and appearance; not
guaranteed bitwise-identical at piece edges), with val selection and test
touched once. Trained on a
RunPod RTX 4090 (50 epochs, 2.2 h); train and val curves overlap for all 50
epochs — under DR the model cannot memorize appearance.

Results: **synthetic test 76.4% cell / 99.0% rotation / 76.2% both** — beats
exp20 (72.9/94.6/72.2) on every metric; DR acted as a regularizer. But on
north_star v1 (touched once): **22.3% cell / 33.5% rotation / 12.7% both**
vs exp20's 14.8% — the transfer gain is zero. Diagnostics: the 4x4-only
subset scores the same (11.5%), all four backgrounds score the same
(10–14%), and the rotation confusion is not diffuse but biased (predicts
90°/270° for everything) — i.e. the features latch onto artifacts that
photographed pieces don't carry.

Key finding: **the sim-to-real gap does not live in the nuisance factors
input-level augmentation can randomize.** Every augmentation transforms the
same digital pixels; a photographed physical piece differs structurally
(printed halftone, gloss, embossed relief, camera ISP, photographed box art
as overview). SIFT survives because it matches local gradient geometry; the
learned features encode nothing that survives print-and-photograph. Next
levers (exp27): pretrained robust features (DINOv2/LoFTR) under the
correlation head (review item #8), a small real-capture training set
(train-split puzzles only), print-and-photograph simulation — and ship the
classical hybrid meanwhile.

---

## Exp 27: Frozen Pretrained Features (DINOv2) Under the Correlation Heads

**Date:** July 2026 **Status:** ZERO-SHOT SUCCESS, TRAINED FAILED on the
real task (zero-shot **49.2% both** on north_star; trained readout
collapses to **7.0%** despite 64.3% synthetic)

Attacked the exp25/exp26 sim-to-real collapse with critical-review item #8:
replace the synthetic-trained backbones with a frozen self-supervised ViT
(DINOv2-S/14) under the validated correlation heads.

**Stage 0 — zero-shot probe.** Before any training, frozen DINOv2 patch
tokens with a *non-learned* cosine readout were evaluated on north_star v1
under the exact exp25 protocol. Dense feature-space template matching
(piece token grid cross-correlated against the puzzle grid, masked, scale
sweep) scored **52.1% cell / 70.1% rotation / 49.2% both** — 3.3× the
trained exp20 CNN (14.8%), level with NCC (48.9%), robust across
backgrounds, failing on the same low-texture puzzles as classical methods.
A pooled-descriptor variant managed only 18.0%, confirming the signal
lives in *dense spatial* correlation. Features hypothesis confirmed:
generic pretrained features beat everything ever trained on the synthetic
data, on the real task.

**Stage 1 — trained adapters + heads.** One shared frozen encoder, ~1.9M
trainable params (per-branch adapters, dense heatmap position head,
rotation-by-re-encoding through the exp20 comparison net), trained on the
exp26 DR data under the frozen-split harness (RunPod 4090, 25 epochs).
Three run-time defects were found by measurement and fixed: fp16 overflow
in the dense correlation (nan under AMP → fp32 island); MSE-through-
expectation starving the 625-window heatmap (val_cell 11.6% at epoch 3 →
switched to SiamFC-style window cross-entropy → 31.5% at epoch 3); and
BatchNorm running stats permanently stale downstream of a frozen encoder
(48% eval rotation vs 97% with batch stats → GroupNorm). Final:
val 64.2% both (epoch 24, still improving); synthetic test (once)
**70.4% cell / 89.5% rotation / 64.3% both**; north_star (once)
**16.3% cell / 35.1% rotation / 7.0% both** — below the zero-shot probe
and even below exp20, uniformly across backgrounds and puzzles, with the
exp26-style biased rotation confusion.

Key finding: **the sim-to-real failure lives in whatever parameters are
trained on synthetic data — not in the features.** Same frozen encoder,
three readouts: non-learned cosine 49.2% real; 1.9M-param readout trained
on synthetic-DR 7.0% real (while 64.3% synthetic); fully-trained CNNs 12.7–
14.8% real. Non-learned readouts survive the camera (SIFT 76.7%, cosine
49.2%); every synthetic-trained one collapses, and DR does not save it.
Next lever by elimination: give the (small) readout **real training data**
— a second real-capture set from train-split-style puzzles, never
north_star — and/or constrain the readout toward the non-learned cosine
form. Durable engineering lessons are in the exp27 README (fp16 correlation
overflow, heatmap CE vs MSE, the frozen-encoder BatchNorm trap).

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
| 16  | 3x3 grid (9 cells)          | 39% cell / 61% rot     | Architecture scales, but needs more data       |
| 17  | 3x3 grid + 10K puzzles      | **81% cell / 93% rot** | Data scaling approach validated for 3x3!       |
| 18  | 3x3 grid + 20K puzzles      | **82% cell / 95% rot** | Data scaling continues to help (NEW BEST)      |
| 19  | Siamese architecture        | 79% cell / 92% rot     | Dual backbone > Siamese for dissimilar inputs  |
| 20  | 4x4 realistic pieces        | **73% cell / 95% rot** | Scales to 4x4 (rot re-measured after label fix)|
| 21  | Masked rotation correlation | 74% cell / 25% rot     | INVALID — capped by test-label bug             |
| 22  | RoMa V2 dense matching      | 64% cell / 58% rot     | Loses to CNN on both metrics (after label fix) |
| 23  | Classical baselines         | **83% cell / 90% rot** | SIFT→NCC hybrid beats the CNN with no training |
| 24  | Piece/not-piece classifier  | **99.1% bal. acc**     | Kills preview false positives (faces at 0.000) |
| 25  | North-star real-photo eval  | **77% both (hybrid)**  | CNN collapses on real photos (14.8% both)      |
| 26  | Domain randomization (4x4)  | 76% synth / 13% real   | Realism augs lift synthetic, don't transfer    |
| 27  | Frozen DINOv2 + corr heads  | **49% real zero-shot** / 7% real trained | Sim-to-real gap lives in the trained readout, not the features |

---

## Current Best Approach

**For Position Only (2×2):** DualInputRegressorWithCorrelation (exp7) +
fine-tuned MobileNetV3-Small backbone (exp9) - 93% quadrant accuracy

**For Position + Rotation (2×2):** DualInputRegressorWithRotationCorrelation
(exp13) with MobileNetV3-Small - **86% quad / 93% rot** with 4,499 training
puzzles

**For Position + Rotation (3×3):** FastBackboneModel (exp18) with
ShuffleNetV2_x0.5 dual backbone - **82% cell / 95% rot** with 20,000 training
puzzles. This is the current production-ready model for 3x3 grids.

**Architecture recommendations:**
- **Use dual backbone (separate piece/puzzle encoders)** - exp19 proved that
  Siamese architectures underperform for this task because piece and puzzle
  images are fundamentally different inputs
- **Pure CNNs beat Transformers** - exp14 showed MobileViT-XS (CNN-Transformer
  hybrid) underperforms for spatial template matching

**For Fast Experimentation:** ShuffleNetV2_x0.5 (exp15) - 12.9s/epoch vs 39.3s
for RepVGG and 98.1s for MobileOne. Use for rapid iteration.

**For Position + Rotation (4×4 realistic pieces):** the zero-training
SIFT→NCC classical hybrid (exp23) — **82.9% cell / 90.3% rotation / 82.2%
both** at ~45 ms/piece on CPU — currently beats the FastBackboneModel CNN
(exp20, re-evaluated July 2026: 72.9% cell / 94.6% rotation / 72.2% both).
The CNN wins only on rotation. RoMa (exp22) is worse than both and ~1000×
slower; the RoMa hybrid plan is dropped. Any future learned model on this
benchmark must clear the 82% classical floor to count as progress.

**For real photographed pieces (the actual product task):** the same
SIFT→NCC hybrid is the only usable method — **77.9% cell / 89.2% rotation /
76.7% both** on north_star v1 (exp25). Every trained CNN is effectively
broken on real photos (exp20 checkpoint: 14.8% both, barely above random on
position). The bar for any learned model is now **76.7% both on
north_star v1**, and the backend would be better served today by the
classical hybrid than by its current CNN path.

exp26 hardened this conclusion: input-level domain randomization
(independent photometrics, geometry, backgrounds, sensor noise) lifts the
*synthetic* benchmark to 76.2% both — the best learned synthetic result on
the realistic 4x4 benchmark — but leaves the real task unchanged (12.7%
both). Whatever separates digital crops from photographed physical pieces
is not reachable by augmenting digital pixels.

exp27 then isolated *where* the failure lives: frozen DINOv2 features with
a non-learned cosine readout score **49.2% both on north_star zero-shot**
(the best learned-feature real result to date, level with NCC), but
training even a tiny 1.9M-parameter readout on top of the same frozen
features — on domain-randomized synthetic data, under the full harness —
collapses the real result to 7.0% while reaching 64.3% synthetic. The
features transfer; **any parameters trained on synthetic data become the
contaminant**. The remaining levers are a small real-capture *training*
set for the readout (never north_star), constraining the readout toward
the non-learned cosine form, and print-and-photograph simulation.

**Key Learnings:**
1. **Data quantity AND quality matter** (exp17): Must see all cells per puzzle
   together for proper feature learning
2. **Siamese is not always better** (exp19): When inputs differ significantly
   (scale, content type), specialized encoders outperform weight sharing
3. **Data scaling helps** (exp18): More diverse training data improves
   generalization
4. **Rotation correlation transfers to realistic pieces** (exp20 re-eval):
   after fixing the test-label bug, the CNN scores 94.6% rotation on
   Bezier-edge pieces — the earlier "realistic pieces break rotation"
   finding (exp20-21) and the RoMa pivot (exp22) were artifacts of the bug.
5. **Measure before you pivot**: three experiments and a research survey
   were driven by a metric that was capped at 25% by construction. A
   perfect-model sanity check on any new test path would have caught it.

**Next Steps** (from the critical review; #1–#4 are done):
- ~~Fix the methodology harness: frozen train/val/test split, checkpoint
  selection on val (not test), train metrics in eval mode~~ — DONE July
  2026, see "Methodology Harness Fix" entry above
- ~~Photographed-piece "north star" test set and first real-photo
  evaluation~~ — DONE July 2026 (exp25): captured, ingested, evaluated;
  CNN collapses (14.8% both), classical hybrid holds (76.7% both)
- **exp26 — train for realism** (critical review item #5): independent
  photometric jitter on piece and puzzle, random scale/perspective warps,
  real backgrounds instead of black, sensor noise/JPEG artifacts in the
  exp20 generator/dataset; train under the frozen-split harness, report
  north_star v1 as the headline metric, beat 76.7% both
- **Consider shipping the SIFT→NCC hybrid in the backend meanwhile** — on
  real photos it beats the served CNN by ~5x both-correct
- Dense heatmap head (review item #6) so one model covers arbitrary grid
  sizes (north-star puzzles range 2x3–4x6; real puzzles 500+ pieces)

---
