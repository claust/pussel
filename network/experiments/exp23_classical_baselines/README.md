# Experiment 23: Classical Baselines (NCC, SIFT, ORB)

## Objective

Run the non-learned baselines that item #2 of `../CRITICAL_REVIEW.md` called
for: establish the floor every learned result must beat on the exp20 4x4
realistic-pieces benchmark, and answer whether the synthetic surrogate task is
trivially solvable by template matching.

## Setup

Exact same protocol as the exp20 CNN re-evaluation (July 2026, fixed rotation
labels):

- Test set: `realistic_4x4_20k_test` — 1,200 puzzles x 16 pieces x 4 applied
  rotations = **76,800 samples**
- Labels composed as `(baked_rotation_from_filename + applied) % 360`; the
  applied rotation uses the same PIL `rotate(expand=False)` call as
  `RealisticPieceTestDataset`
- Metrics: cell accuracy (16-way), rotation accuracy (4-way), both-correct
- Matched subsample: 200 puzzles (seed 42, 12,800 samples) reported for all
  methods so any slower method can be compared on identical data

**Resolution:** classical methods use the images at native resolution —
puzzle JPEGs are 256x256 and piece PNGs are variable-size tight crops
(~110–130 px) on black backgrounds. Unlike the CNN input pipeline, pieces are
*not* squashed to 128x128 squares. Runtimes are per sample, single CPU core
(Apple M-series); the run is parallelized over puzzles with 8 workers.

### Methods

1. **NCC** — masked `cv2.matchTemplate` with `TM_CCOEFF_NORMED` (mask =
   pixels brighter than 8, i.e. not the black background). The piece is slid
   over the full puzzle at each of the 4 candidate un-rotations (lossless
   `np.rot90`); the best-scoring location + rotation wins. Predicted cell =
   4x4 cell containing the matched template center.
2. **SIFT** — keypoints on the masked piece matched to puzzle keypoints
   (BFMatcher, Lowe ratio 0.75), partial-affine RANSAC
   (`cv2.estimateAffinePartial2D`, reproj threshold 3.0, min 4 good matches /
   3 inliers). Position = inlier centroid; rotation = transform angle snapped
   to the nearest 90°.
3. **ORB** — same harness as SIFT with Hamming matching. Default ORB detects
   almost no keypoints on these small low-texture pieces (~8% coverage), so it
   runs with `nfeatures=3000, fastThreshold=0, edgeThreshold=10, nlevels=12`
   (~40% coverage).
4. **SIFT→NCC hybrid** — still fully classical: use the SIFT prediction when
   SIFT matches, otherwise fall back to NCC. Derived from the same per-sample
   records, so its runtime is SIFT plus NCC on the ~55% of samples that need
   the fallback.

Keypoint methods sometimes produce no prediction (too few keypoints/matches/
inliers). Headline metrics count those samples as wrong; `*_covered` values in
`outputs/results.json` condition on a prediction existing. NCC coverage is
99.6% (the remaining 0.4% are samples where the masked response was NaN/inf at
every candidate rotation).

## Results (full 76,800 samples)

| Method              | Cell      | Rotation  | Both      | Coverage | ms/sample |
| ------------------- | --------- | --------- | --------- | -------- | --------- |
| **SIFT→NCC hybrid** | **82.9%** | 90.3%     | **82.2%** | 99.6%    | 45.4      |
| NCC (masked CCOEFF) | 77.5%     | 87.2%     | 76.9%     | 99.6%    | 76.8      |
| CNN (exp20 re-eval) | 72.9%     | **94.6%** | 72.2%     | 100%     | ~10 (GPU) |
| SIFT                | 43.9%     | 44.1%     | 43.6%     | 45.4%    | 3.2       |
| ORB (tuned)         | 38.4%     | 38.3%     | 37.7%     | 40.4%    | 4.8       |

On the matched 200-puzzle subsample (seed 42, 12,800 samples): hybrid
83.5% both, NCC 78.0% both, SIFT 44.3% both, ORB 39.1% both — same ordering,
so the subsample is a faithful stand-in if a slower method ever needs it.

When SIFT does match, it is nearly perfect: 96.7% cell / 97.3% rotation on
the 45% of samples it covers. Its failures are coverage failures (too little
texture on the piece), not wrong answers. SIFT/ORB puzzle keypoint extraction
adds ~9 ms per puzzle, amortized over 64 samples.

## Conclusions

1. **The floor is above the CNN.** The best classical method (SIFT→NCC
   hybrid, 82.2% both-correct) beats the exp20 CNN (72.2%) by 10 points, and
   even plain masked NCC beats it on position (77.5% vs 72.9%) and
   both-correct (76.9% vs 72.2%). The CNN only wins on rotation
   (94.6% vs 90.3%).
2. **The surrogate task is largely solvable by template matching**, as the
   critical review suspected. It is not literally trivial (pixel-identical
   crops would give ~100%; Bezier masking, `expand=False` rotation clipping
   and JPEG resampling cost NCC ~23 points), but a zero-training afternoon
   baseline outperforming the trained model means synthetic-benchmark gains
   cannot be read as progress on the real task.
3. **Learned-model results on this benchmark must now clear 82% both-correct**
   to demonstrate any value over classical matching — until the benchmark
   itself is made adversarial to pixel matching (independent photometric
   jitter, perspective, backgrounds, sensor noise) or replaced with
   photographed pieces, per items #4–5 of the critical review.

## Files

- `evaluate.py` — runs all methods; writes `outputs/results.json`
- `outputs/results.json` — full + subsample metrics, parameters, CNN reference

## Reproduce

```bash
cd network
uv run python experiments/exp23_classical_baselines/evaluate.py \
    --dataset-root datasets/realistic_4x4_20k_test \
    --puzzle-root datasets/puzzles \
    --workers 8
```

The test set is regenerated (not checked in); see the "Exp 20 Re-Evaluation"
entry in `../EXPERIMENT_LOG.md` for how to rebuild it with the historical
`puzzle_shapes` geometry.
