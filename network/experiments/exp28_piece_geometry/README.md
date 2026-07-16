# Experiment 28: Piece Geometry Scanning — Contour Extraction + Corner Bake-off

## Objective

Implements milestones **M1** (high-fidelity contour extraction) and **M2**
(corner detection bake-off) of the piece-geometry-scanning plan
(`../../../docs/piece-geometry-scanning.html`): turn an existing real photo
of a puzzle piece into a clean, closed contour and 4 labeled corners, as the
first step toward extracting exact piece geometry (contour + tab/blank/flat
edge classification) from real photos.

- **M1**: upgrade the existing rembg pipeline (`piece_detector.py` /
  `background_remover.py` in the backend) from a coarse UI polygon to a
  full-resolution, morphologically-cleaned, closed contour, run over the
  `north_star v1` dataset (236 pieces x 4 backgrounds).
- **M2**: bake off three corner-detection strategies (contour curvature,
  `approxPolyDP` epsilon sweep, Shi-Tomasi corners on the filled mask)
  against both a synthetic benchmark with known ground truth and hand-labeled
  real photos.

## Dataset

`network/datasets/north_star/v1/` (local-only, see
`../north_star/README.md`): 14 puzzles, 236 pieces, each photographed
upright on 4 backgrounds (red_carpet, gray_fabric, cardboard, wood) = 944
piece photos, 1024x768 JPEGs.

## Files

- `common.py` — shared utilities: metadata loading, cropping, rembg session
  management, mask -> contour extraction, quality scoring, arc-length
  resampling.
- `extract_contours.py` (**M1**) — per-piece contour extraction via rembg
  and/or Otsu thresholding; writes per-piece JSON + a run summary CSV +
  a clean-rate table.
- `contact_sheets.py` (**M1 review**) — renders per-puzzle x background grid
  contact sheets with the extracted contour overlaid (green = clean, red =
  flagged), for human QA.
- `corner_detect.py` (**M2**) — three pure, rotation-agnostic corner
  detectors (`detect_corners_curvature`, `detect_corners_polydp`,
  `detect_corners_shitomasi`) sharing a brute-force best-4-subset scorer.
- `synth_benchmark.py` (**M2 quantitative**) — generates random pieces via
  `puzzle_shapes`, rasterizes + rotates + degrades them, and scores all three
  detectors against known ground-truth corners.
- `label_corners.py` (**M2 ground truth**) — interactive matplotlib tool to
  hand-click the 4 true corners of real piece photos.
- `eval_corners.py` (**M2 evaluation + review**) — runs the detectors on
  real photos, renders corner contact sheets, and (given hand labels) scores
  them the same way as `synth_benchmark.py`.
- `edge_split.py` (**M3**) — splits each clean contour at the polydp corners
  (curvature as cross-check -> `corner_disagreement` flag) into 4 arcs,
  maps them to grid directions N/E/S/W, and classifies each as
  tab/blank/flat by dominant chord deviation (`FLAT_THRESHOLD`); writes
  per-piece records + `edge_summary.csv` and prints the deviation histogram.
- `eval_edges.py` (**M3 evaluation + review**) — scores edge types against
  grid ground truth (flat accuracy vs border edges, cross-background
  consistency, neighbor tab/blank complementarity), renders color-coded
  review sheets, and writes `edge_eval.json`.
- `edge_match.py` (**M4**) — canonical edge frames (chord-normalized;
  outward = negative canonical y), the mate `flip_edge` transform, and the
  match distances (l2, chamfer, scalar6 feature vector, chord-penalty
  variants). Its CLI runs a synthetic self-check on a `puzzle_shapes`
  shared edge.
- `eval_matching.py` (**M4 evaluation + review**) — ranks each interior
  edge's true mate among all type-compatible edges of the other pieces in
  the same puzzle x background; reports top-k / median rank per metric,
  the corner_disagreement gate delta, genuine-vs-impostor stats, renders
  query/mate/impostor curve sheets, and writes `matching_eval.json`.

## Usage

All commands run from `network/`:

```bash
# M1: extract contours (rembg + threshold) for a few pieces
uv run python experiments/exp28_piece_geometry/extract_contours.py --limit 8

# M1: extract the full dataset, rembg only
uv run python experiments/exp28_piece_geometry/extract_contours.py --method rembg

# M1 review: build contact sheets for one puzzle
uv run python experiments/exp28_piece_geometry/contact_sheets.py --puzzle bambi

# M2: synthetic corner-detector benchmark
uv run python experiments/exp28_piece_geometry/synth_benchmark.py --n 200

# M2: hand-label ground-truth corners on a subset
uv run python experiments/exp28_piece_geometry/label_corners.py --limit 60

# M2: evaluate detectors on real photos (with review sheets + optional scoring)
uv run python experiments/exp28_piece_geometry/eval_corners.py \
    --labels-file outputs/corner_labels.json

# M3: split contours into classified N/E/S/W edges, then evaluate
uv run python experiments/exp28_piece_geometry/edge_split.py
uv run python experiments/exp28_piece_geometry/eval_edges.py

# M4: edge-match self-check, then mate-ranking evaluation
uv run python experiments/exp28_piece_geometry/edge_match.py
uv run python experiments/exp28_piece_geometry/eval_matching.py
```

## Success criteria

- **M1**: >=95% of pieces yield a single clean closed contour (`is_clean` in
  `extract_contours.py`'s summary) on at least the 2 friendliest backgrounds.
- **M2**: the best detector places all 4 corners within 3% of piece size
  (the `synth_benchmark.py` / `eval_corners.py` max-corner-error metric) on
  >=90% of the evaluated subset.
- **M3**: flat-vs-nonflat edge accuracy >=98% (vs grid-position ground
  truth) on the 2 friendly backgrounds (gray_fabric, red_carpet).

## Output layout

```
outputs/
  contours/{puzzle_id}/{piece_stem}.json   # per-piece metadata + per-method contour + quality
  summary.csv                              # one row per piece x method
  review/{puzzle}_{background}_{method}.png       # M1 contact sheets
  review_corners/{puzzle}_{background}.png        # M2 contact sheets
  synth_benchmark.csv                      # per synthetic piece x method scores
  corner_labels.json                       # hand-labeled ground truth (label_corners.py)
  corner_eval.csv                          # per method x background error stats
  piece_records/{puzzle_id}/{piece_stem}.json  # M3: corners + 4 classified edges per piece
  edge_summary.csv                         # M3: one row per edge
  review_edges/{puzzle}_{background}.png   # M3 review sheets (flat=yellow tab=green blank=red)
  edge_eval.json                           # M3 evaluation numbers
  review_matching/{puzzle}_{background}.png    # M4: query vs mate vs impostor curves
  matching_eval.json                       # M4 evaluation numbers
```

`outputs/` is gitignored, matching other experiments (e.g. `exp1`,
`exp2`, `exp4`, `exp5`); rerun the scripts above to regenerate it.
