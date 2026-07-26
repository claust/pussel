# Experiment 30: Fix the Generator Bugs, Retrain exp26 Unchanged

## Objective

Test the **shortcut theory** end-to-end. The 2026-07-26 realism
investigation (`docs/synthetic-dataset-realism.html`, sections 4 and 7)
concluded that the sim-to-real collapse is not a realism deficit but
**shortcut learning**, and named two outright label leaks in the synthetic
piece pipeline. exp30 changes *nothing* about the exp26 training recipe
except those leaks and the framing mismatch that goes with them.

**Prediction:** north_star **rotation accuracy jumps well above 33.5%** and
the 90°/270° prediction bias disappears. If it does, the shortcut theory is
validated and further "realism" work (Tests 2–5) is justified; if it does
not, the theory is wrong and the ranking in section 7 needs rethinking.

## The three things fixed

### 1. Border-touch rotation shortcut (§4.1)

`cut_piece` crops tight to the silhouette (`getbbox()`,
`shared/puzzle_shapes/puzzle_shapes/image_masking.py:259-263`), and exp20 /
exp26 then rotate that **non-square** canvas with `expand=False`. 0°/180°
content fills the canvas — alpha touches all four borders — while 90°/270°
content is *clipped* (~60 px of tabs destroyed) and left with black bands.
"Touches all four borders ⟹ rotation ∈ {0°, 180°}" is a 100%-precision rule
in synthetic train **and** test. Real crops are `pad_to_square`'d with an 8%
margin and never touch a border, so at test time the model is forced down
the odd-rotation branch — exactly exp26's "predicts 90°/270° for
everything" diagnostic.

**Fix:** every multiple-of-90 rotation goes through `Image.transpose`
(`framing.rotate_lossless`). Transpose is an exact pixel permutation: the
canvas dimensions swap, nothing is clipped, nothing is resampled.

> PIL's `Transpose.ROTATE_90` is **counterclockwise**; exp20/exp26 label
> rotations **clockwise** (`image.rotate(-rotation, ...)`). The mapping is
> therefore crossed — clockwise 90° = `Transpose.ROTATE_270` — and
> `test_exp30.py` proves it against `rotate(-rot, expand=True)` on an
> asymmetric pattern rather than trusting the reasoning.

### 2. Half-pixel blur cue (§4.2)

`rotate(expand=False, BILINEAR)` on a non-square canvas puts 90°/270° on
half-pixel offsets, making them ~32% blurrier than the pixel-exact 0°/180°.
Sharpness alone leaks the rotation label. The same transpose fix removes it:
all four rotations are byte-identical permutations of one another.

### 3. Aspect / framing mismatch (§7, Test 1)

Training resized the tight crop straight to 128×128 (squashing aspect,
differently per rotation once dimensions swap) while deployment pads to
square with an 8% margin first. **Fix:** `framing.frame_rgba` reproduces
exp24's real preparation geometry — largest opaque component → 8% margin →
pad to square — on the RGBA piece, at **load** time. Pieces stay cropped
tight on disk, so one dataset serves both the augmented train path and the
deterministic eval path.

`frame_rgba` is the RGBA-preserving twin of exp24's
`pad_to_square(_crop_with_margin(...))`; `test_exp30.py` asserts the two are
pixel-identical after a black composite. The one deliberate difference: the
margin is *added* rather than clamped to the source bounds, because
generator pieces arrive with no surround to take it from.

## Why framing happens *before* `augment_piece`

The framed canvas is **square and rotation-symmetric**. Every step
downstream of it — exp26's rotation jitter, perspective, scale, halo, the
background sampled at `piece.size`, the composite, and the final 128×128
resize — therefore treats the four rotation classes identically *by
construction*, rather than by an argument about each augmentation in turn.
Concretely this buys:

- **aspect preserved**: square in, square out (verified: 192/192 augmented
  train pieces square), so the 128×128 resize never squashes;
- **no rotation-correlated border contact**: the 8% transparent margin means
  the silhouette cannot reach the input border (verified: 0/1536 augmented
  train draws touch all four borders, uniformly across labels);
- **exp26-identical augmentation**: `augment_piece` is imported from exp26
  **unchanged** — exp30 edits no file in exp20, exp24, exp25, exp26 or
  `shared/puzzle_shapes`.

The one accepted side effect: the piece now occupies a slightly smaller
fraction of its frame (square canvas + 8% margin instead of a tight crop).
That moves the "mask area fraction" metric from §5 toward the real value
(0.44) rather than away from it, so it is a fix, not a regression.

**exp26's `augment.py` needs no neutralisation.** Audited for the cues this
experiment removes: its only geometric rotation (`_rotate_rgba`) already
uses `expand=True`, and `_perspective_rgba` maps the canvas corners
*inward* (torchvision's `RandomPerspective.get_params` perturbs endpoints
into the interior), so neither clips content nor reintroduces a
rotation-correlated blur. Its bilinear rotation jitter does blur, but
uniformly and independently of the label. No config override is required.

## Measured effect of the fix

3-puzzle smoke sample, 48 pieces × 4 load-time rotations = 192 inputs per
scheme, alpha > 0:

| Label rotation | Old (exp26): touches 4 borders | New (exp30) | Old mean \|∇\| | New mean \|∇\| |
| --- | --- | --- | --- | --- |
| 0° | **56.2%** | 0.0% | 7.522 | 6.216 |
| 90° | 6.2% | 0.0% | 7.272 | 6.216 |
| 180° | **56.2%** | 0.0% | 7.522 | 6.216 |
| 270° | 6.2% | 0.0% | 7.272 | 6.216 |

Under the old scheme the border-touch signature predicts rotation parity
with **90% precision** (54/60) and the 0/180-vs-90/270 sharpness gap is
3.3% on this near-square sample (the §4.2 measurement used a 128×188 piece,
where it is 32%). Under exp30 both cues read **exactly zero**, and every
rotation yields an identical square canvas with an identical opaque-pixel
count. Aspect ratio W/H goes from 1.019 mean (0.67–1.58 per piece) to
1.000 for every sample.

## How to run

**1. Generate the dataset** (own root — the pieces are *not* interchangeable
with exp26's `realistic_4x4_rgba`):

```bash
cd network
uv run python -m experiments.exp30_generator_fixes.generate_dataset \
    --source-dir datasets/puzzles \
    --output-dir datasets/realistic_4x4_rgba_v2 \
    --n-puzzles 100000 --workers 8 --skip-existing
```

**2. Run the acceptance probes** (`probes.py`) **before retraining** — §8
requires border-touch independence, classical parity, proxy 𝒜-distance,
NCC headroom and the RAPSD overlay to pass on the new dataset first.

**3. Train:**

```bash
cd network
uv run python -m experiments.exp30_generator_fixes.train --epochs 50 --eval-test
```

**RunPod (the real run)** — generation and 50-epoch training happen on the
pod, generation is CPU-parallel and resumable:

```bash
cd network/experiments/exp30_generator_fixes
./runpod/prepare_package.sh          # code + frozen split + source puzzles
# scp runpod_package_exp30/runpod_training.tar.gz to the pod, then:
#   cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh
```

**4. North-star evaluation (ONCE, after training):**

```bash
cd network
uv run python experiments/exp25_north_star_eval/evaluate.py \
    --dataset-root datasets/north_star/v1 \
    --checkpoint experiments/exp30_generator_fixes/outputs/generator_fixes/checkpoint_best_state_dict.pt
```

`train.py` exports `checkpoint_best_state_dict.pt` (a raw state_dict) in the
same format and under the same name as exp26, so the north-star evaluator
runs against it unchanged.

**Tests:**

```bash
cd network
uv run pytest experiments/exp30_generator_fixes/test_exp30.py -q
```

## Training recipe — deliberately identical to exp26

`FastBackboneModel` / ShuffleNetV2_x0.5, AdamW (backbone 1e-4, head 1e-3,
wd 0.01), 128 px piece / 256 px puzzle, batch 64 (128 on the pod), 50
epochs, `AugmentConfig` defaults and the same `--aug-preset` ablation
entry point, the frozen exp20 split `splits/realistic_4x4_v1.json`, and the
exp20 harness. Val selects the checkpoint; the synthetic test set is touched
**once** with `--eval-test`; north_star is evaluated exactly once at the
very end. Any difference in the results is therefore attributable to the
data fix and nothing else.

Known limitation, inherited from exp26: there is no *real* validation set,
so the checkpoint is selected on clean synthetic val.

## What success looks like

- **Primary:** north_star rotation accuracy ≫ 33.5% (exp26's chance-level
  score), and the rotation confusion matrix is diffuse rather than piling
  every prediction onto 90°/270°.
- **Secondary:** north_star both-correct moves off the 12.7% floor. Beating
  the SIFT→NCC hybrid's 76.7% is *not* expected from this fix alone —
  §4.3's pixel-identity shortcut survives exp30 untouched and is Test 2's
  job.
- **Expected on synthetic:** rotation accuracy *falls* from exp26's 99.0%.
  That is the point — 99.0% was the shortcut being read off. A synthetic
  test score that stays at 99.0% means a leak is still present.
- **Gate:** the §8 acceptance probes must pass on the regenerated dataset
  before any retraining is started.

## Files

- `framing.py` — `rotate_lossless` (transpose) + `frame_rgba` (exp24's
  8%-margin square geometry, RGBA-preserving). The whole fix lives here.
- `framed_dataset.py` — exp26's train/eval datasets with lossless rotation
  and real-path framing; `create_datasets_from_split` over the frozen split.
  Also exposes `piece_to_model_input(piece_path, applied_rotation_idx)`, the
  single-sample form of the eval piece branch, so `probes.py` can read
  exactly what the model sees without building a dataset over a full split.
- `generate_dataset.py` — RGBA piece generator (parallel, resumable),
  identical to exp26's except the lossless base rotation
- `train.py` — training entry point (exp26 recipe, exp20 harness)
- `probes.py` — §8 realism acceptance probes
- `test_exp30.py` — unit tests for rotation direction, content preservation,
  exp24 framing parity and border non-contact
- `runpod/` — `prepare_package.sh` + `setup_and_train.sh`

## Results

Trained 50 epochs on a RunPod RTX 5090 (~223 s/epoch); artifacts in
`outputs/generator_fixes/` (`results.json`) and `outputs/north_star_results.json`.

| Metric | exp26 | **exp30** | bar (SIFT→NCC) |
| --- | --- | --- | --- |
| Synthetic test — both | 76.2% | **78.5%** | 82.2% |
| north_star — cell | 22.3% | 21.5% | 77.9% |
| north_star — rotation | 33.5% | **37.9%** | 89.2% |
| north_star — both | 12.7% | 13.2% | **76.7%** |

**The fix worked; the hypothesis did not.** The predicted mechanism changed
exactly as expected — exp26 predicted 90°/270° for nearly everything, while
exp30's predicted-rotation distribution is near-uniform (990/902/989/895 over
0/90/180/270, uniform = 944) and the acceptance probes pass at border touch
0.000 / sharpness ratio 1.0000. But rotation accuracy rose only 33.5% → 37.9%
against a 25% chance floor, cell accuracy did not move, and the 76.7% bar is
untouched. Removing label leakage is necessary data hygiene worth ~4 points,
not the cause of the sim-to-real collapse. See `../EXPERIMENT_LOG.md` (Exp 30)
and exp31, which tests the remaining pixel-identity hypothesis.

> Evaluation gotcha: the north_star eval is only trustworthy when the classical
> baselines reproduce exp25 exactly (`sift_else_ncc` = 77.9/89.2/76.7). A first
> run mixed a stale pre-orientation-fix copy of the overviews with the current
> piece-crop cache and collapsed every classical method to ~4% cell; the CNN
> number from that run was meaningless. Always eval against the **main
> checkout's** `network/datasets/north_star/v1`.
