# Critical Review of Experiments 1–22

**Date:** July 2026

A critical audit of all experiments in this directory, based on the
per-experiment READMEs, the experiment code, and the raw result JSONs.
The goal: separate what we have actually established from what we only
believe we have established, and identify where the experimental
methodology led us astray.

---

## What we have actually achieved

Four findings survive scrutiny and are worth keeping:

1. **The spatial-correlation architecture (exp7) is a real insight.**
   Fusing globally-pooled feature vectors failed; explicit
   template-matching-style correlation between piece and puzzle feature
   maps works. This is consistent with the tracking/matching literature.
2. **Rotation-as-matching (exp12) is a real insight.** Predicting
   rotation from the piece alone fails to generalize; correlating the
   piece against the puzzle at each of the 4 candidate rotations works.
3. **Data scaling works (exp13, exp17→18).** exp13 is the cleanest
   experiment in the whole series — byte-identical model, one variable
   changed (800 → 4,499 puzzles), large gain.
4. **A working pipeline and mostly-clean puzzle-level splits** from exp7
   onward (no direct train/test puzzle leakage found in exp7–19).

Current best on the synthetic benchmark: 82% cell / 95% rotation on 3×3
grids (exp18), 73% position on 4×4 Bezier pieces (exp20).

---

## Where the experiments went wrong

### 1. The exp20 test set has broken rotation labels (CRITICAL)

> **RESOLVED (2026-07-13):** Fixed the label composition and re-evaluated
> the original exp20 checkpoint on a deterministic regeneration of the
> same test split: **72.9% cell (exact reproduction) / 94.6% rotation**.
> Rotation never failed; the RoMa/hybrid pivot is dropped. See the
> "Exp 20 Re-Evaluation" entry in EXPERIMENT_LOG.md.

This invalidates the last three experiments and the current roadmap.

Piece PNGs are generated with a random rotation baked in
(`exp20_realistic_pieces/generate_dataset.py:140-147`), but the test
dataset discards it when parsing the filename and hardcodes base
rotation to 0 (`exp20_realistic_pieces/dataset.py:312-320`), then labels
the sample with only the freshly applied test rotation. The true
rotation is `(baked + applied) % 360`, so **even a perfect model scores
exactly 25%**.

The results JSON proves it: test rotation is pinned at 24.6–24.9% for
all 50 epochs while train climbs to 95% — the signature of scrambled
labels, not overfitting. The training path composes labels correctly
(`dataset.py:224-233`), so the model genuinely learned rotation on the
training set. Consequences:

- exp20's conclusion "rotation fundamentally fails on realistic pieces"
  was never actually measured.
- exp21 ("masking had zero effect, still 24.7%") is exactly what a
  25%-capped metric returns for *any* intervention. exp21 also has no
  code in the repo at all — only the log entry exists.
- exp22's headline "RoMa 58% vs CNN 25%, 2.3× breakthrough" compares
  correct labels (RoMa reads ground truth from `metadata.csv` in
  `exp22_roma_matching/evaluate.py:171-173`) against broken ones. The
  comparison is invalid. RoMa also got a test-time 4-rotation search
  that the CNN was never given.
- The entire `IMAGE_PATCH_LOCALIZATION_RESEARCH.md` survey and the
  "hybrid RoMa + CNN" next-step plan are motivated by this unvalidated
  failure.

### 2. Every experiment measures a toy surrogate of the real task

> **CONFIRMED (2026-07-13):** exp23 ran the classical baselines on the
> exp20 benchmark. A zero-training SIFT→NCC hybrid scores **82.9% cell /
> 90.3% rotation / 82.2% both** — 10 points above the CNN's 72.2%
> both-correct — and even plain masked NCC beats the CNN on position
> (77.5% vs 72.9%). The CNN *under-performs* the classical floor on the
> surrogate, winning only rotation (94.6% vs 90.3%). See
> `exp23_classical_baselines/`.

The "piece" is a pixel-exact crop of the very same digital image the
network receives as the puzzle input — same pixels, same lighting, no
camera, no perspective, no background, no scale uncertainty (exp20 adds
only a Bezier outline on black). The product task is matching a
*photographed physical piece* against a box image.

Worse, no non-learned baseline was ever run: plain normalized
cross-correlation over cells × rotations would plausibly score near
100% on this setup, so we do not even know whether 82–93% represents
good performance or *under*-performance on the surrogate.

### 3. No validation set anywhere; the test set steered development

> **RESOLVED (2026-07-14):** the realistic 4x4 benchmark now has a
> frozen train/val/test split checked in at
> `exp20_realistic_pieces/splits/realistic_4x4_v1.json` (test portion
> identical to the original exp20 test split; 600-puzzle val carved from
> the original train portion). The training harness
> (`exp20_realistic_pieces/harness.py` + `train.py`) selects checkpoints
> on val, measures train metrics in eval mode on a frozen `train_eval`
> subset, and touches test once per experiment (opt-in `--eval-test`).

exp7–9 iterate on the identical seed-42 200-puzzle test set; exp10–14
select `model_best.pt` by test accuracy (e.g.
`exp12_rotation_correlation/train.py:340-387`). Every hyperparameter and
architecture decision since exp7 was implicitly tuned on "held-out"
data.

### 4. Nearly every A/B changed multiple variables at once

The most consequential:

- **exp17's** celebrated "must see all cells of a puzzle together" — the
  9-cells run also had **9× more samples and gradient steps** than the
  1-cell run. "More data/compute" was never controlled for.
- **exp18's** "+1.3% from 20K puzzles" also changed batch size, learning
  rate, *and* the test set (the split slicing means different test
  puzzles at different train sizes) — within noise for 200 clustered
  test puzzles.
- **exp14's** "Transformers unsuitable" — the MobileViT run also dropped
  the BatchNorm batch from 64 to 8, used untuned CNN hyperparameters,
  and its *train* accuracy was only 78.6%: it was underfitting, not
  failing to generalize.
- **exp11's** "spatial rotation head doesn't help" — that run also got
  4× fewer gradient steps than exp10.

### 5. Decisions were made on wrong baseline numbers

exp14 and exp15 both compare against an exp13 that took "~12 hours"
with "5.27M params"; exp13's own results JSON records **4.0 hours and
2.52M params**. The "ShuffleNetV2 is 34× faster" claim normalizes to
**~1.24×** at equal dataset size — and MobileNetV3 was never run in the
exp15 harness. The backbone used for every experiment since exp16 was
chosen on this faulty comparison.

### 6. Early negative results were invalid by construction

- **exp5** trained with a single puzzle: every batch's puzzle input was
  identical, and BatchNorm normalizes a constant input to a bias — the
  puzzle branch *could not* contribute, so "the dual-input architecture
  doesn't generalize" was unfalsifiable.
- **exp6** never fit its training set (2.4% train accuracy), so it says
  nothing about generalization either.

### 7. Chronic statistical and numeric sloppiness

- Every reported delta is a single run, no seeds, no error bars
  (exp19's "Siamese disproven" is a 2.8-point single-run difference).
- MSE numbers were repeatedly misread as ~1% error when they mean
  ~8–10% error (exp1/exp2).
- Train accuracy is measured with augmentation + dropout active, so
  "test > train = no overfitting" is a measurement artifact.
- A real geometry bug: `rotate(expand=False)` clips tab protrusions on
  non-square canvases at 90°/270°.

---

## Options and recommendations

### Do these first — cheap, and they reorder everything else

1. **Fix the exp20 test dataset** (compose base + applied rotation into
   the label) **and re-evaluate the existing exp20 checkpoint.** This is
   a few lines. If the CNN's rotation is actually decent, the
   RoMa/hybrid pivot dissolves and the roadmap changes completely.
   Nothing else should be decided before this.
2. **Run classical baselines** on the same benchmarks: OpenCV NCC
   template matching over cells × 4 rotations, and SIFT/ORB matching.
   This gives the floor every learned result must beat — and a reality
   check on whether the surrogate task is trivially solvable.

   > **DONE (2026-07-13, exp23):** the floor is 82.2% both-correct
   > (SIFT→NCC hybrid, no training), above the CNN's 72.2%. The
   > surrogate is largely solvable by template matching; learned models
   > must beat 82% here before any synthetic-benchmark gain counts.
3. **Fix the methodology harness once, centrally:** a frozen
   train/val/test split shared by all experiments (val for selection,
   test touched once per experiment), checkpoint selection on val, train
   metrics measured in eval mode, and READMEs auto-populated from the
   results JSON (that would have prevented the exp13 baseline fiction).
   Re-run the load-bearing comparisons (dual vs Siamese, MobileNetV3 vs
   ShuffleNet) with matched controls and ~3 seeds before trusting them.

   > **DONE (2026-07-14), except README auto-population and the
   > comparison re-runs:** frozen split + val-selection + eval-mode
   > train metrics landed in `exp20_realistic_pieces/`
   > (`splits.py`, `splits/realistic_4x4_v1.json`, `harness.py`,
   > unified `train.py`; `train_cuda.py` merged in). See the RESOLVED
   > note under issue #3 above.

### Then attack the real gap — realism, not grid size

4. **The biggest risk to the whole program is that the models have
   learned pixel-identical matching that will not survive a camera.**
   Before scaling to 5×5 or 8×8 grids, photograph pieces of one or two
   real puzzles and build a small "north star" test set. Even 100 real
   pieces will tell us more than another 20K synthetic puzzles.

   > **PLANNED (2026-07-13):** capture protocol, labeling scheme, and
   > repo-storage plan written up in `NORTH_STAR_GUIDE.md`.
5. Meanwhile, make the synthetic data adversarial to pixel-matching:
   apply *independent* photometric jitter to piece and puzzle, random
   scale/perspective warps on the piece, realistic backgrounds instead
   of black, sensor noise/JPEG artifacts, and non-axis-aligned rotations
   (±10° around each 90° candidate). If accuracy collapses under this,
   we learn the current architecture's real ceiling early and cheaply.

### Architecture/efficiency ideas, in order of expected value

6. **Replace cell classification with a dense heatmap output.**
   Cross-correlate piece features against the puzzle feature map and
   supervise a location heatmap (SiamFC-style tracking). This is a small
   step from the existing correlation module, scales naturally from 3×3
   to arbitrary granularity (real puzzles are 500–1000 pieces — a
   softmax over cells will not get there), and gives sub-cell position
   for free.
7. **Give the CNN the same test-time 4-rotation search RoMa got.** Run
   the position head on all 4 rotations of the piece and take the
   max-correlation one. It is the matched control exp22 skipped, and
   likely recovers most of RoMa's advantage at a fraction of its
   5.6 s/piece cost.
8. **Use pretrained matching features as a backbone or teacher, not at
   runtime.** A frozen DINOv2/LoFTR-style encoder under the correlation
   head, or RoMa as an offline label/distillation teacher, gets robust
   features that cannot overfit the synthetic domain — without shipping
   a 5-second-per-piece matcher.
9. Re-do the backbone speed/accuracy comparison honestly (equal data,
   converged runs, MobileNetV3 included) before committing to
   ShuffleNetV2 for anything final.

---

## One-sentence summary

The architecture insights (correlation for position,
correlation-over-rotations for rotation, data scaling) are solid, but
the two decisions currently steering the project — "rotation is
impossible for realistic pieces" and "ShuffleNet for speed" — are both
artifacts of measurement bugs, and no result yet says anything about
photographed pieces.
