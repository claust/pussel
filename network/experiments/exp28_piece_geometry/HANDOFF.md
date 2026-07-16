# exp28 Handoff — M1 (contour extraction) + M2 (corner detection)

Date: 2026-07-16 · Status: **both milestone success criteria met** · Plan: [docs/piece-geometry-scanning.html](../../../docs/piece-geometry-scanning.html)

## M1 — High-fidelity contour extraction

Ran both extraction methods over the full north_star set (236 pieces × 4 backgrounds = 944 photos).

| Background   | rembg clean rate | Otsu-threshold clean rate |
|--------------|------------------|---------------------------|
| gray_fabric  | 233/236 (98.7%)  | 53/236 (22.5%)            |
| red_carpet   | 231/236 (97.9%)  | 6/236 (2.5%)              |
| wood         | 227/236 (96.2%)  | 2/236 (0.8%)              |
| cardboard    | 225/236 (95.3%)  | 40/236 (16.9%)            |

**Criterion (≥95% clean on the 2 friendly backgrounds): met — on all four backgrounds, not just two.**

Findings beyond the numbers:

- **Gate-failure modes** (28 rembg failures): 17 multi-component segmentations, 8 border-touching
  crops, 3 implausible shape metrics. Spread thinly across puzzles; worst case was
  lion_king × wood (4).
- **Silent failures exist on wood** (visual review of contact sheets): a few pieces per sheet pass
  the quality gate but the contour follows *dark artwork boundaries* instead of the physical piece
  edge (dark art blends into wood tones/shadows). Gray fabric and red carpet sheets are essentially
  flawless — contours hug piece edges and correctly exclude drop shadows.
- **Otsu thresholding is dead on textured real backgrounds** — kept only as the documented baseline.
  rembg (u2net) is the extraction method going forward.
- **Capture-protocol recommendation for the future scanning mat: a neutral gray fabric-like
  surface** (best clean rate, zero observed silent failures).

Review artifacts: `outputs/review/*.png` (56 contact sheets, green = passed gate, red = failed),
`outputs/summary.csv`, per-piece contour JSONs in `outputs/contours/`.

## M2 — Corner detection bake-off

Three detectors (`corner_detect.py`), all sharing a candidate-pool → best-4-subset → refinement
pipeline: `curvature` (turn-angle extrema), `polydp` (approxPolyDP epsilon sweep), `shitomasi`
(goodFeaturesToTrack on the filled mask).

**Synthetic benchmark** (n=200 random `puzzle_shapes` pieces, random rotation, blur+noise, known
ground-truth corners; metric = worst-of-4 corner error as % of piece diagonal):

| Method    | Median err | ≤3% err (initial → final) |
|-----------|-----------:|---------------------------|
| polydp    | 0.72%      | 68.0% → **98.5%**         |
| curvature | 0.68%      | 29.5% → **97.0%**         |
| shitomasi | 27.08%     | 1.5% → 13.5%              |

**Criterion (≤3% worst-corner error on ≥90% of pieces): met by polydp and curvature.**

What made the difference (the initial versions failed on 30–70% of pieces by picking tab-bulb tips):

1. **Cornerness prior**: true corners have locally *straight* contour shoulders on both sides; tab
   tips curve away immediately. Per-candidate score = straightness(before) × straightness(after) ×
   local-angle-90° score, from PCA line fits over arc-length windows.
2. **Rebalanced subset score**: quad area normalized by hull area (0–1) so it can no longer dominate
   the angle/spacing terms.
3. **Corner refinement**: intersect the two shoulder lines (tight window) — recovers the un-rounded
   corner position that corner_radius hides; p97 error dropped 3.6% → 0.8%.
4. **Candidate-pool fixes**: the true corner must be *in* the pool before scoring can pick it
   (polydp keeps all sweep vertices; curvature NMS separation halved).

`shitomasi` fails structurally, not by tuning: goodFeaturesToTrack responds to sharp tab armpits and
misses rounded true corners — ~33% of true corners never enter its candidate pool. Dropped from
consideration.

**Real photos** (all 944, no ground-truth labels yet): corner-overlay sheets in
`outputs/review_corners/` look very strong — on gray fabric, nearly every piece has all methods
agreeing on the true corners; occasional single-method outliers. Notably shitomasi performs fine on
real pieces (real corners are sharper than the extreme synthetic ones), reinforcing that the
synthetic benchmark is the *harder* test for corner geometry.

**Quantitative real-photo eval is prepared but needs hand labels**: run
`uv run python experiments/exp28_piece_geometry/label_corners.py` (click 4 corners per piece,
~60-piece subset suggested), then `eval_corners.py --labels-file outputs/corner_labels.json`
produces the per-method/per-background error table automatically. Optional but recommended before
M4 conclusions.

## Recommendation going into M3 (edge splitting + tab/blank/flat classification)

- Use **polydp as the primary detector, curvature as cross-check** — when their corner sets
  disagree materially, flag the piece as low-confidence (this doubles as the quality gate for the
  future scan-lock UX).
- Split contours at the detected corners into 4 edge arcs; classify by the chord-midpoint test.
  Ground truth for edge types is nearly free from north_star grid positions (border edges flat,
  interior edges tab/blank with parity constraints).
- Wood-background silent contour failures will pollute M3/M4 metrics slightly — either restrict
  M3/M4 development to gray_fabric + red_carpet, or add a per-piece cross-background consistency
  check later.

## Reproduce

```bash
cd network
uv run python experiments/exp28_piece_geometry/extract_contours.py          # ~35 min (rembg over 944 photos)
uv run python experiments/exp28_piece_geometry/contact_sheets.py
uv run python experiments/exp28_piece_geometry/synth_benchmark.py --n 200 --seed 42
uv run python experiments/exp28_piece_geometry/debug_synth_failures.py      # renders failure cases
uv run python experiments/exp28_piece_geometry/eval_corners.py              # corner sheets (+ eval if labels exist)
```

Dataset: `network/datasets/north_star/v1` (local only, regenerate via `experiments/north_star/ingest.py`).
