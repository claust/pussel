# exp28 Handoff — M1 (contour extraction) + M2 (corner detection) + M3 (edge classification)

Date: 2026-07-16 · Status: **all three milestone success criteria met** · Plan: [docs/piece-geometry-scanning.html](../../../docs/piece-geometry-scanning.html)

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

## M3 — Edge splitting + tab/blank/flat classification

`edge_split.py` splits each clean contour at the polydp corners (curvature as cross-check →
`corner_disagreement` flag), maps the 4 arcs to grid directions N/E/S/W (pieces are photographed
upright), classifies each by signed chord deviation, and emits a **piece record** per photo
(corners + 4 typed edges, each with a 100-point polyline — the direct input for M4). Coverage:
916/944 photos (the 28 non-clean contours skipped), 0 splitting failures, 135 disagreement flags.

The deviation distribution is strongly bimodal exactly as hoped: flat edges live in [0, ~0.04]
(median 0.009, as fraction of chord length), tab/blank features in [~0.11, 0.42] (median 0.292).
`FLAT_THRESHOLD = 0.07` sits mid-gap.

**Evaluation against near-free ground truth** (`eval_edges.py`, `outputs/edge_eval.json`):

| Metric | Result |
|---|---|
| Flat-edge accuracy vs grid borders — gray_fabric | **99.68%** |
| — red_carpet | 98.38% |
| — cardboard | 98.00% |
| — wood | 97.25% |
| Neighbor tab↔blank complementarity — gray_fabric | 99.43% of adjacent pairs |
| — red_carpet / cardboard / wood | 95.9% / 96.3% / 92.2% |
| Cross-background signature consistency | 80.5% of pieces (190/236) |

**Criterion (≥98% edge-type accuracy on clean contours): met on the friendly backgrounds
(99.68% / 98.38%).**

Failure taxonomy (visually verified on review sheets in `outputs/review_edges/`):

1. **Wood segmentation leaks** (dominates wood's errors and the consistency shortfall): silhouettes
   that merged wood-grain shadow or neighboring props passed `is_clean`, corrupting corners and
   hence edge types. An M1 issue surfacing downstream, exactly as predicted — the
   `corner_disagreement` flags concentrate on these records, validating the flag as a quality gate.
2. **Figural borders on toddler puzzles are GT noise, not classifier error**: peppa_kitchen (90.2%)
   and peppa_aquarium (94.1%) have border edges with real shaped cutouts/protrusions (verified
   visually — e.g. peppa_kitchen r01c01's bottom border). The classifier honestly reports non-flat
   geometry; the "border ⇒ flat" assumption is what's wrong. Standard-die-cut puzzles score
   99.5–100%.
3. Residual tab↔blank flips across backgrounds trace to occasional corner mis-snaps rotating an
   arc's chord — revisit only if M4 edge matching turns out sensitive to it.

## Recommendation going into M4 (edge complementarity matching)

- Match on the per-edge 100-point polylines already stored in the piece records; gray_fabric
  records are the cleanest development set (99.7% edge types, 99.4% complementarity).
- Exclude records with `corner_disagreement` from the first matching benchmark (135/916), then
  measure how much including them hurts — that quantifies the quality gate's value for the scan UX.
- True-neighbor ranking GT comes from grid adjacency, same as the complementarity check.

## Reproduce

```bash
cd network
uv run python experiments/exp28_piece_geometry/extract_contours.py          # ~35 min (rembg over 944 photos)
uv run python experiments/exp28_piece_geometry/contact_sheets.py
uv run python experiments/exp28_piece_geometry/synth_benchmark.py --n 200 --seed 42
uv run python experiments/exp28_piece_geometry/debug_synth_failures.py      # renders failure cases
uv run python experiments/exp28_piece_geometry/eval_corners.py              # corner sheets (+ eval if labels exist)
uv run python experiments/exp28_piece_geometry/edge_split.py                # M3: piece records + edge_summary.csv
uv run python experiments/exp28_piece_geometry/eval_edges.py                # M3: eval + edge-type sheets
```

Dataset: `network/datasets/north_star/v1` (local only, regenerate via `experiments/north_star/ingest.py`).
