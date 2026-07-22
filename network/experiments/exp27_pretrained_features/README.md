# Exp 27: Frozen Pretrained Features Under the Correlation Heads

**Status:** DONE (2026-07-17). Stage 0 (zero-shot probe): **49.2% both** on
north_star — frozen DINOv2 features transfer. Stage 1 (trained adapters +
heads on synthetic DR data): synthetic test **64.3% both**, but north_star
**7.0% both** — training the 1.9M-parameter readout on synthetic data
*destroyed* the transfer the same frozen features had zero-shot.

**Headline finding: the sim-to-real failure lives in any parameters trained
on synthetic data, not in the features.** Same encoder, three readouts:
non-learned cosine correlation → 49.2% real; adapters+heads trained on
synthetic → 7.0% real (64.3% synthetic); for reference, fully-trained CNNs →
14.8% (exp20) / 12.7% (exp26). Domain randomization does not prevent the
learned readout from latching onto synthetic-only cues (rotation confusion
on real photos is biased 0↔180, exp26-style). The remaining levers, in
order: **real-photo training data** for the readout (a second capture set,
never north_star), constraining the readout toward the non-learned cosine
form, and print-and-photograph simulation.

## Hypothesis

The correlation architecture (position via spatial correlation, rotation via
rotation-as-matching) is validated (exp7/12/13, exp20 re-eval). What collapses
on real photos is the *features*: backbones trained end-to-end on digital
crops encode nothing that survives print-and-photograph (exp25: 14.8% both;
exp26: input-level domain randomization lifts synthetic to 76.2% but leaves
the real task at 12.7%). SIFT — handcrafted, robust local features — loses
only 5.5 points synthetic→real. So: swap the trained-from-scratch backbones
for a **frozen self-supervised ViT (DINOv2-S/14)** and train only a small
adapter plus the existing heads. Frozen features cannot overfit the synthetic
domain, making the exp26 failure mode structurally impossible.

Headline metric: **both-correct on north_star v1** (exp25 protocol).
Production bar: 76.7% both (SIFT→NCC hybrid, shipped July 2026).
Hypothesis-confirmation bar: decisively beat the trained CNN's 14.8% /
exp26's 12.7%.

## Stage 0: zero-shot probe (`zero_shot_probe.py`)

Before training anything, measure what frozen DINOv2 features are worth on
their own, directly on north_star v1 with the exact exp25 protocol (same
segmented-crop cache, overview auto-crop, applied-rotation convention, grid
binning, metrics). No training, no learned parameters.

Two probe methods, both evaluated per candidate rotation (encode the 4
rotated piece images — ViT features are not rotation-equivariant, so
feature-map rotation is not a substitute):

- **`dino_mean`** — piece descriptor = foreground-masked mean of patch
  tokens; cosine map against the puzzle patch-token grid; cell-sized window
  smoothing; peak → position, best rotation by peak score. (The zero-shot
  analogue of exp20's `SpatialCorrelationModule` with a pooled piece vector.)
- **`dino_dense`** — the piece's full patch-token grid, foreground-weighted
  and resized to a cell-sized window, cross-correlated against the puzzle
  token grid via convolution (masked NCC in DINOv2 feature space), over a
  small scale sweep (pieces with tabs span ~1.0–1.5 cells). Peak → position,
  best rotation by peak score. (The zero-shot analogue of dense template
  matching with frozen features.)

Reference numbers on north_star v1 (exp25, both-correct): SIFT→NCC hybrid
76.7%, NCC multi-scale 48.9%, CNN+rotsearch 18.0%, CNN 14.8%. Random ≈ 1.5%
(5.9% cell × 25% rotation).

### Decision gates

- **Probe cell accuracy clearly above the CNN's 22.4%** (e.g. ≥35–40%) →
  features hypothesis confirmed; proceed to full exp27 training (adapter +
  heads on the exp26 DR data under the frozen-split harness).
- **Probe ≈ CNN or below** → the sim-to-real gap is not (mainly) feature
  robustness; pivot to real-capture training data / print-and-photograph
  simulation before spending a training run.
- **Probe already near or above NCC's 48.9%** → strong signal; training the
  heads should target the production bar (76.7%), and a SIFT→learned-fallback
  hybrid becomes the likely ship shape.

The probe touches north_star, so it is diagnostic only — no threshold tuning
against it, run it once (plus at most a smoke run on 1 puzzle).

## Stage 1: trained model (built 2026-07-16)

Code:

- `model.py` — `FrozenFeatureModel`: one shared frozen DINOv2-S/14
  (`FrozenViTEncoder`, ImageNet normalization applied INSIDE the model so
  callers keep feeding [0,1] tensors), per-branch trainable adapters,
  `DensePositionModule` (probe-validated dense template head; `--position-head
  pooled` gives the exp20 `SpatialCorrelationModule` ablation), and
  `PrecomputedRotationCorrelation` — the exp20 comparison net fed with
  re-encodings of the 4 rotated piece images. 23.9M params, **1.88M
  trainable**.
- `train.py` — mirrors exp26: same harness (`fit`/`evaluate`, val selection,
  `--eval-test` once), same exp26 DR data path, inputs 224/448.
- `north_star_eval.py` — the one-shot real-photo evaluation on the trained
  checkpoint (exp25 protocol; single forward — the model's rotation head is
  already a 4-rotation search).
- `runpod/` — packaging (`prepare_package.sh`, run with
  `DATASETS_DIR=<main-checkout>/network/datasets`) and pod script
  (`setup_and_train.sh`); ships the DINOv2 weights as an offline HF cache
  (`HF_HUB_OFFLINE=1`) so the pod never depends on hub availability. RGBA
  pieces are generated on the pod's container disk (exp26 MooseFS lesson).

Sanity checks run before the full training:

- Forward shapes + parameter count (`python -m ...exp27_pretrained_features.model`).
- 2-epoch smoke on a 100-puzzle subset (MPS): losses decrease, checkpoints +
  raw export written, rotation train accuracy above random by epoch 2.
- Learning-capability check: 20 epochs, 88 puzzles, aug off — see Results.
- Flat RunPod package imports verified locally.
- MPS timing (~86 ms/sample fp32) confirms local full training is
  infeasible (~4 h/epoch); the run goes to RunPod (CUDA + AMP).

### Stage 1 results (2026-07-17)

Final run (heatmap CE + GroupNorm, 25 epochs, RunPod RTX 4090, ~6.4 h,
batch 64, full DR preset, frozen split, val selection):

- **Val (600 puzzles):** best epoch 24 — 70.3% cell / 89.8% rotation /
  64.2% both. Train ≈ val for all 25 epochs (DR prevents memorization);
  the curve was still improving at epoch 25.
- **Synthetic test (touched once):** **70.4% cell / 89.5% rotation /
  64.3% both** — below exp26 (76.4/99.0/76.2) and exp20 (72.9/94.6/72.2),
  as predicted up front: frozen semantic features are worse than
  task-trained features at pixel-exact matching.
- **north_star v1 (touched once):** **16.3% cell / 35.1% rotation /
  7.0% both** — catastrophic, *below* the zero-shot probe (52.1/70.1/49.2)
  and below exp20's trained CNN (14.8% both). Uniform across backgrounds
  (6.4–7.8%) and across puzzles (even `frozen_closeup`, 91.7% both
  zero-shot, drops to 16%); rotation confusion biased 0↔180. Raw results:
  `outputs/north_star_results.json`.

Interpretation: the frozen encoder still produces transferable features
(the probe proves it), but the trained adapters+heads read them out through
synthetic-specific directions that photographed pieces don't populate. With
this, every learned-readout configuration has now collapsed on real photos
(end-to-end, end-to-end+DR, frozen-features+trained-readout), while the two
non-learned readouts survive (SIFT 76.7%, zero-shot cosine 49.2%). The
conclusion is structural: **synthetic supervision is the contaminant.**
Next: train the same 1.9M-parameter readout on a small real-capture set
(train puzzles only), possibly initialized from this checkpoint.

### Findings from the training runs (2026-07-16)

Three issues found and fixed during the RunPod runs, each caught by a
measurement rather than guessed:

1. **fp16 overflow in the dense correlation (run 1, epoch 2):** the grouped
   conv accumulates 128×8×8 products before normalization → `pos_loss=nan`
   under AMP; GradScaler silently skipped affected batches. Fix: the
   correlation + softmax run in fp32 outside autocast.
2. **MSE-through-expectation starves the dense head (run 2, epochs 1–3):**
   val_cell crawled 9.7→11.6%. Diagnostic on the epoch-3 checkpoint: the
   best 3×3 attention window held only ~10% of the mass, argmax decoding
   barely beat expectation (16.9% vs 13.6%). Fix: `heatmap_ce_loss` —
   cross-entropy against the true window (SiamFC-style) + small MSE to keep
   the expectation/refinement path calibrated. Result: val_cell 16.9/27.4/
   31.5% in epochs 1–3 (vs 9.7/11.3/11.6 with MSE), 54.6% by epoch 8.
3. **BatchNorm running stats break the rotation head under a frozen
   encoder (run 2, epochs 5–9):** train rotation loss fell to 0.14 while
   eval-mode rotation was stuck ~49%. Diagnostic on the epoch-9 checkpoint:
   48.4% rotation with running stats vs **97.4% with batch statistics**
   (and 43.2% on augmented inputs — so not an aug-vs-clean gap). With a
   frozen encoder, the fast-moving adapters (lr 1e-3) shift the head's
   input distribution every epoch and the BN EMA is permanently stale;
   exp20/26 never hit this because end-to-end backbones co-adapt slowly.
   Fix: GroupNorm in `PrecomputedRotationCorrelation`. **Durable lesson:
   don't put BatchNorm downstream of a frozen backbone with fast-training
   adapters.**

### Original stage-1 design notes

- One **shared frozen DINOv2-S/14** encoder (exp19's dual-vs-Siamese finding
  concerned *trained* encoders; frozen ones are identical by construction).
- Per-branch trainable adapters (1×1 conv 384→256 + GELU + 3×3 conv, ~0.7M
  params each) feeding the exp20 heads (`SpatialCorrelationModule`,
  refinement, `RotationCorrelationModule`) unchanged, `feature_dim=256`.
- **Rotation via re-encoding**: the 4 rotated piece images go through the
  frozen encoder (no-grad); `_rotate_feature_map` is kept only as an ablation.
- Inputs: puzzle 448×448 (32×32 patch map, exactly 8×8 per 4×4 cell), piece
  224×224 (16×16). ImageNet normalization (new — the current pipeline has
  none; must be mirrored in every evaluator).
- Data/harness: exp26 `aug_dataset` with the full DR preset, frozen split
  `realistic_4x4_v1.json`, val selection, synthetic test once via
  `--eval-test`, north_star once via the exp25 evaluator (needs a small
  model-factory extension).
- Trainable params ≈ 1.5–2M; heads-only lr 1e-3 cosine, 50 epochs, AMP on
  CUDA (RunPod RTX 4090, est. 3–5 h). Phase 2 (only if val plateaus): unfreeze
  the last ViT block at lr 1e-5 as a separate run.

Expectation stated up front: exp27 may score *below* exp26's 76.2% both on
the synthetic test — frozen semantic features are worse at pixel-exact
matching than features trained for it. The headline is north_star.

### Sanity checks before the full run

1. Perfect-model check on the eval path (ground-truth labels → 100%).
2. Overfit 10 samples with adapter+heads (gradient flow).
3. One eval-path round trip of the exp27 model through the extended exp25
   evaluator on a handful of pieces (transform mismatches — e.g. the new
   ImageNet normalization — silently zero results).

## Results

### Stage 0: zero-shot probe (2026-07-16)

Full north_star v1 (14 puzzles, 944 piece photos, 3,776 samples), exp25
protocol, ~90 s total on M4 MPS (23 ms/sample). Raw output:
`outputs/zero_shot_results.json`.

| Method                        | Cell      | Rotation  | Both      |
| ----------------------------- | --------- | --------- | --------- |
| SIFT→NCC hybrid (exp25)       | 77.9%     | 89.2%     | **76.7%** |
| **dino_dense (zero-shot)**    | **52.1%** | **70.1%** | **49.2%** |
| NCC multi-scale (exp25)       | 50.5%     | 68.9%     | 48.9%     |
| CNN + rot search (exp25)      | 24.2%     | 48.1%     | 18.0%     |
| dino_mean (zero-shot)         | 25.3%     | 52.0%     | 18.0%     |
| CNN exp20, trained (exp25)    | 22.4%     | 44.0%     | 14.8%     |

Findings:

1. **The features hypothesis is confirmed.** Frozen DINOv2 features with a
   dumb argmax — no adapter, no heads, no training — already beat the trained
   CNN 3.3× on both-correct and match the NCC classical baseline. Whatever
   the CNN backbones learned on synthetic data is *worse than generic
   pretrained features* on real photos. This clears the "proceed to stage 1"
   gate with room to spare (gate was ≥35–40% cell; got 52.1%).
2. **Spatial structure is where the signal lives.** `dino_mean` (pooled piece
   descriptor, the analogue of exp20's pooled position correlation) gets
   18.0% both; `dino_dense` (full token grid cross-correlated, the analogue
   of dense template matching) gets 49.2%. The trained heads should follow
   the dense shape — which is exactly review item #6's dense heatmap head.
3. **Background-robust**: 47.0–50.8% both across all four capture
   backgrounds (the trained CNN degraded on wood).
4. **Failure mode mirrors NCC's, not the CNN's**: per-puzzle spread is
   9.4% both (`unicorn_pink`) to 94.4% (`peppa_aquarium`) — the same
   low-texture puzzles that break NCC. High-texture puzzles are near-solved
   zero-shot (frozen_closeup 91.7%, peppa_aquarium 94.4% both). Rotation
   confusion is unbiased (uniform ~9–12% per wrong class), unlike exp26's
   pathological everything→90°/270°.
5. **Headroom for stage 1 is real**: the gap to the hybrid (49.2 → 76.7) is
   concentrated in low-texture puzzles, and a trained adapter + correlation
   heads gets to sharpen exactly the comparisons the zero-shot argmax leaves
   on the table.

Protocol notes: puzzle 448² (32×32 patch tokens), piece 224² (16×16), 4
rotated piece encodes per photo, dense scale sweep 1.0/1.25/1.5 of the
nominal cell size, foreground-masked (black-background) token weighting.
Since np.rot90 is lossless, the best net rotation is computed once per photo
and the prediction for applied rotation `a` is `(a − r*) % 4` — so the
rotation confusion matrix is circulant by construction.
