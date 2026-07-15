# Experiment 25: First Evaluation on the North-Star Real-Photo Benchmark

## Objective

Answer the critical review's headline question (item #4): do any of our
methods survive a camera? Every result so far — synthetic pieces that are
pixel-exact crops of the puzzle input — left open whether the CNN learned
generalizable matching or pixel-identical template lookup. This experiment
runs the existing methods, unchanged, on `north_star v1`: photographs of 14
physical kids' puzzles (236 pieces, 944 piece photos on 4 backgrounds).

## Setup

- **Dataset**: `datasets/north_star/v1` (local-only, regenerated with
  `north_star/ingest.py`; all human review overrides applied). 14 puzzles,
  grids from 2x3 to 5x5 and 4x6; each piece photographed upright on
  red carpet, gray fabric, cardboard, and wood.
- **Samples**: each of the 944 piece photos evaluated at all 4 applied
  clockwise rotations = **3,776 samples** per method. Pieces are verified
  upright, so true rotation = applied rotation (exp20 label convention).
- **Piece preparation mirrors the deployed preview path** (exp24/backend):
  crop the photo to the piece bbox from `metadata.csv` (this removes the
  orientation arrow), segment with rembg (u2net), composite the largest
  opaque component on black, pad square. rembg succeeded on all 944 photos
  (0 fallbacks). Crops cached in `datasets/north_star/v1_eval_cache/`.
- **Overview**: the photographed box art / assembled puzzle, auto-cropped to
  the puzzle region (union of non-background components; review sheet written
  to `outputs/overview_crops.jpg`). All methods see the same cropped overview.
- **Cell prediction**: continuous position normalized to the cropped
  overview, binned into each puzzle's own rows x cols grid (row-major). The
  exp20 CNN outputs continuous position, so it is evaluated on all 14 grids,
  not just 4x4.
- Random guessing scores ~5.9% cell / 25% rotation / ~1.5% both on this
  sample mix.

### Dataset fix made along the way

The overview JPEGs produced by ingest kept their EXIF orientation tag as
metadata, so their raw pixel orientation was arbitrary (and EXIF was *wrong*
for 4 of 14 puzzles — iPhone orientation is unreliable for top-down shots,
and the poster sometimes lay rotated relative to the camera). Fixed by
measurement: `check_overview_orientation.py` SIFT-matches the
verified-upright piece crops against the overview at all four rotations; the
rotation where pieces match with zero residual rotation wins. The vote was
180° CW for all 14 puzzles, near-unanimously (24–94 votes for, ≤8 against).
`ingest.py` now bakes the measured rotation into the overview pixels and
strips the stale EXIF tag (`OVERVIEW_ROTATIONS`, `normalize_overview`).

### Methods

1. **NCC (multi-scale)** — exp23's masked `TM_CCOEFF_NORMED`, extended with a
   scale search (piece resized to {0.9, 1.1, 1.3, 1.5} x the nominal cell
   size) because real photos have scale uncertainty that the synthetic
   benchmark lacked. Overview at max side 256 (matching the synthetic
   protocol's resolution).
2. **SIFT** — exp23's recipe (Lowe ratio 0.75, partial-affine RANSAC) at
   real-photo sizes: overview max side 768, piece crop max side 384, FLANN
   matching. Position = inlier centroid; rotation = transform angle snapped
   to 90°. SIFT handles the scale gap natively.
3. **SIFT→NCC hybrid** — SIFT when it matches, NCC otherwise (exp23's
   winner, still zero-training).
4. **CNN (exp20 checkpoint)** — unchanged `FastBackboneModel`
   (ShuffleNetV2_x0.5 dual backbone), piece squashed to 128x128, cropped
   overview to 256x256, exactly as trained.
5. **CNN + test-time rotation search** (critical-review item #7) — forward
   all 4 un-rotations of the piece and pick the candidate with the highest
   P(rotation=0); cell and rotation read from the winning pass.

## Results (3,776 samples)

| Method              | Cell      | Rotation  | Both      | Coverage | ms/sample |
| ------------------- | --------- | --------- | --------- | -------- | --------- |
| **SIFT→NCC hybrid** | **77.9%** | **89.2%** | **76.7%** | 100%     | 85        |
| SIFT                | 72.6%     | 80.1%     | 71.5%     | 85.1%    | 38        |
| NCC (multi-scale)   | 50.5%     | 68.9%     | 48.9%     | 100%     | 329       |
| CNN + rot search    | 24.2%     | 48.1%     | 18.0%     | 100%     | 10 (MPS)  |
| CNN (exp20)         | 22.4%     | 44.0%     | 14.8%     | 100%     | 3 (MPS)   |

Synthetic → real, both-correct: hybrid 82.2% → 76.7% (−5.5), NCC 76.9% →
48.9% (−28), CNN 72.2% → **14.8% (−57)**.

Detail worth keeping:

- **SIFT when it matches is still nearly perfect**: 85.3% cell / 94.1%
  rotation / 84.0% both on the 85.1% of samples it covers — and its coverage
  nearly doubled vs the synthetic benchmark (45%), because real piece photos
  are larger and more textured than 110px synthetic crops.
- **The CNN's collapse is uniform**: 12.5–16.7% both across all four
  backgrounds; its rotation confusion is diffuse (not the clean 180°
  ambiguity it shows on synthetic data), i.e. the features transfer barely
  at all. Test-time rotation search recovers only +3.2 both.
- **Per-background (hybrid)**: red carpet 78.0%, gray fabric 79.9%,
  cardboard 80.0%, wood 69.0% both — wood is the hardest (grain texture and
  low piece/background contrast leak into segmentation).
- **Per-puzzle (hybrid)**: 8 of 14 puzzles at 75–100% both; the weak ones are
  `frozen_scene` (43.8%) and `unicorn_pink` (54.7%), both low-texture /
  repetitive artwork where SIFT fails and the NCC fallback is weak.

## Conclusions

1. **The critical review's headline fear is confirmed, quantitatively.** The
   trained CNN learned pixel-identical matching that does not survive a
   camera: 72.2% → 14.8% both-correct (cell accuracy 22.4% vs 5.9% random).
   Every synthetic-benchmark result in exp1–exp20 said nothing about the
   real task.
2. **Classical matching mostly survives.** The zero-training SIFT→NCC hybrid
   drops only 5.5 points (82.2% → 76.7%) and is the best method on the real
   benchmark by 28 points. It is also fast enough to deploy (85 ms/sample on
   CPU). The production backend currently serves the exp18 CNN; on real
   photos that model family is far below the classical hybrid.
3. **NCC is the fragile half of the hybrid** (−28 on real photos even with
   scale search): pure pixel correlation suffers from camera photometry and
   scale error. Improving the fallback (or the learned model) for
   low-texture pieces is where the headroom is.
4. **exp26 has a concrete target: beat 76.7% both-correct on north_star v1.**
   The obvious lever is training with realism (independent photometric
   jitter, scale/perspective, real backgrounds, sensor noise — critical
   review item #5), evaluated against this benchmark under the frozen-split
   harness. A model that cannot beat a 2004 feature detector on the real
   task does not earn its inference cost.

## Files

- `evaluate.py` — full harness; writes `outputs/results.json`
- `check_overview_orientation.py` — measures overview orientation from the
  pieces; writes `outputs/overview_orientation.json`
- `outputs/results.json` — all metrics (overall, per-background, per-puzzle,
  4x4 subset, confusion matrices, runtimes, protocol parameters)
- `outputs/overview_crops.jpg` — review sheet of the overview auto-crops

## Reproduce

```bash
cd network
uv run python experiments/north_star/ingest.py          # rebuild v1 (~10 min, macOS)
uv run python experiments/exp25_north_star_eval/evaluate.py \
    --dataset-root datasets/north_star/v1 \
    --checkpoint experiments/exp20_realistic_pieces/outputs/checkpoint_best.pt
```

The first run segments all 944 piece photos with rembg (~25 min, cached
afterwards). North-star usage discipline applies: this set is test-only —
evaluate at milestones, never train on it, never use it for model selection.
