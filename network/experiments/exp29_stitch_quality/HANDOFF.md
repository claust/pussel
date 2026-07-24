# exp29 Handoff — Benchmark tool built, verified on synthetic data AND two real dumps

Date: 2026-07-24 · Status: **tool complete, self-tested (11 tests), and validated
against the first two real DEBUG-build device dumps** — two metric weaknesses
found during that validation (glare-healing blindness, healed-vs-misaligned
conflation) are fixed and re-verified · Plan: none (standalone tooling, not
a milestone track)

## What's here

`score_stitch.py` scores an existing capture-dump's `composite.jpg` against
its `reference.jpg` on 6 axes (global SIFT+RANSAC geometry, local
phase-correlation ghosting map with healed-patch exclusion, Canny/gradient
edge doubling, variance-of-Laplacian sharpness, near-saturated-pixel glare
reduction, and per-pixel-darkening glare healing), prints a table, and writes
`metrics.json` plus 3 diagnostic images (ghost heatmap, absdiff heatmap,
worst-patch flicker crop). Optional `--quad` restricts all six axes to a
region, reported alongside full-frame. `stitch.py` is an independent offline
Python reimplementation of the app's highlight-cap → SIFT+RANSAC →
warp-with-white-fill → min-composite pipeline, giving a second composite to
score the app's own output against. See `README.md` for full metric
definitions and usage.

## Round 1 — synthetic-only validation

Original 7-test synthetic suite (still present, unchanged): a textured scene,
5 per-shot glare discs at different positions, a ground-truth-registered
"aligned" composite and a deliberately ~8px-shifted "misaligned" one. The
metrics cleanly separated the two — local ghosting `p95_shift_px` from <1px
(aligned) to 4+px (misaligned), worst patch from ~5px to 40+px.

## Round 2 — real-dump validation (2026-07-24) found and fixed two weaknesses

Run against the first two real device dumps
(`~/Pictures/puzzles/glare_stitch_dumps/GlareFreeDumps/20260724-115252` and
`.../20260724-115323`, not committed):

**Finding 1 — glare-reduction metric was blind to matte-print glare.**
`20260724-115323`'s reference has an obvious, broad, desaturating glare sheen
by eye. It never reaches gray 250, so near-saturation glare reduction
reported 0.000% on both images — a total miss. Fix: a **glare-healing**
metric based on a per-pixel darkening map,
`max(0, blur(reference) - blur(composite))` — the min-composite's only
possible source of legitimate per-pixel change, so it's an unambiguous,
saturation-independent healing signal. On this dump it correctly reports
57.32% of pixels darkened, mean darkening 29.60/255 over those pixels.
`near_saturated_fraction` / `canny_edge_count` / `mean_gradient_magnitude` /
`variance_of_laplacian` all gained an optional `mask` parameter along the way
(needed for `--quad` too, see finding 3).

**Finding 2 — local ghosting conflated healing with misalignment.** The same
dump's worst local-ghosting patch (row 12, col 15, shift 11.56px) turned out
to be exactly the glare sheen — visually confirmed via
`worst_patch_flicker.png` to be a pure appearance change with zero positional
edge shift. Fix: patches whose mean value in the darkening map exceeds
`HEALED_PATCH_DARKENING_THRESHOLD` (25.0/255, empirically tuned against this
dump) are excluded from the shift statistics and marked `healed=True`,
reported separately as `healed_patches` (232/768 on this dump). After the
fix, the worst patch moved to row 14, col 16 (shift 3.67px) — visually
confirmed to show genuine ghosting (a doubled card-edge/hair-curl boundary).
`ghost_heatmap.png` now overlays healed patches in translucent green,
distinct from both real-ghosting (inferno) and skipped-uniform (no overlay).

**Finding 3 — ghost stats were dominated by background.** Added optional
`--quad "x1,y1 x2,y2 x3,y3 x4,y4"` (unit coords, clockwise from top-left) to
restrict all six axes to a region (e.g. the puzzle itself), reported as
`region` in `metrics.json` alongside the unchanged full-frame numbers. Global
geometry reuses the single full-frame SIFT+RANSAC fit and subsets it by
region rather than refitting, so region and full-frame numbers stay directly
comparable.

Threshold-tuning note: `20260724-115323` also has a global exposure/white-
balance mismatch between the reference and corner shots that darkens the
ENTIRE frame (including background carpet, ~18-27/255 baseline even far from
the puzzle) — not purely glare-specific. `HEALED_PATCH_DARKENING_THRESHOLD =
25.0` sits above that baseline and below the confirmed sheen cluster
(~40-76/255); `DARKENED_PIXEL_THRESHOLD = 8.0` (the per-pixel glare-healing
report threshold) was left at the value given in the brief and does report a
high 57% darkened-fraction on this dump as a result — that's an honest
reflection of this specific capture, not a metric bug (the other dump,
`20260724-115252`, reports 0.05%).

## Round-2 test additions (11 tests total, all passing)

- Two new fixtures for glare healing: "darkens nothing" (composite identical
  to a glared reference — darkening must read ~0) and "sheen the composite
  heals" (broad, non-saturating haze — darkening must be clearly positive
  while near-saturation glare reduction reports exactly 0% on both images,
  reproducing the real-dump finding).
- One new fixture for the healed-patch exclusion mechanism: a genuinely
  ~8px-shifted patch competing against a patch that's BOTH swapped for
  unrelated content (a large raw phase-correlation shift) AND darkened
  (crosses the healed threshold) — asserted that the darkened patch's raw
  shift actually exceeds the genuine one's, so the test exercises exclusion
  changing the outcome, not just leaving an already-correct answer alone.
- One new test for `--quad`: region stats populate, full-frame stats stay
  unchanged, region patch count is a strict subset of full-frame.

## Known gaps / next steps

1. **Only two real dumps exist.** The "expected value ranges" in the README
   now include both synthetic AND these two real numbers, but two captures
   (one clean, one glare-heavy) isn't a calibration set — revisit
   `HEALED_PATCH_DARKENING_THRESHOLD` and the qualitative good/bad guidance
   as more real dumps accumulate.
2. **Edge-doubling ratio remains a weak global signal in practice** — on
   both the synthetic misaligned case and `20260724-115323`'s real ghosting,
   it barely moved, because the ratio is diluted by the whole image
   (including white-filled warp margins and, now, healed regions the ratio
   doesn't know to exclude). Local ghosting (`p95_shift_px`, `worst_patch`)
   remains the metric to trust; edge doubling doesn't currently get the same
   healed-exclusion treatment (Canny/gradient are diffuse per-pixel signals,
   not per-patch, so "excluding a region" would mean literally masking it
   out of the ratio -- `--quad` is the closer tool for that if a region is
   known to be problematic).
3. **`stitch.py` is intentionally not byte-identical to the Swift pipeline**
   (approximate highlight-cap threshold, no attempt to match the app's exact
   SIFT parameters or RANSAC settings) — it's an offline iteration aid, not a
   reference implementation to diff against for correctness.
4. **No automatic `--quad` estimation.** The region has to be hand-supplied
   (e.g. from the puzzle's known bounding box in the capture UI, if that
   becomes available); this tool doesn't detect the puzzle boundary itself.

## Reproduce

```bash
cd network
uv run pytest experiments/exp29_stitch_quality/test_exp29.py -v
uv run python experiments/exp29_stitch_quality/score_stitch.py /path/to/real/dump
uv run python experiments/exp29_stitch_quality/score_stitch.py /path/to/real/dump --quad "0.28,0.26 0.72,0.26 0.72,0.74 0.28,0.74"
uv run python experiments/exp29_stitch_quality/stitch.py /path/to/real/dump --out /tmp/restitched.jpg
```
