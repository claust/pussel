# Stitch-quality tooling handoff — Benchmark tool built, verified on synthetic data AND three real dumps

Date: 2026-07-24 · Status: **tool complete, self-tested (16 tests), and validated
against the first three real DEBUG-build device dumps** — three metric weaknesses
found during that validation (glare-healing blindness, healed-vs-misaligned
conflation, bright-detail erasure invisibility) are fixed and re-verified;
`stitch.py --mode masked`, an alternative stitch that restricts compositing to a
glare mask, is built and tuned against the same three dumps (Round 4), then its
mask signal replaced with a median-based robust estimate that suppresses most of
a real carpet background's order-statistics false-positive rate (Round 5), then
the median replaced with a stricter all-of-N/3-of-4 vote that further cuts
border-region carpet leakage (Round 6) — the same recipe, including Round 6's
vote, is also ported and validated in the iOS app's Swift composer ·
Plan: none (standalone tooling, not a milestone track)

## What's here

`score_stitch.py` scores an existing capture-dump's `composite.jpg` against
its `reference.jpg` on 7 axes (global SIFT+RANSAC geometry, local
phase-correlation ghosting map with healed-patch exclusion, Canny/gradient
edge doubling, variance-of-Laplacian sharpness, near-saturated-pixel glare
reduction, per-pixel-darkening glare healing, and bright-speck retention),
prints a table, and writes
`metrics.json` plus 3 diagnostic images (ghost heatmap, absdiff heatmap,
worst-patch flicker crop). Optional `--quad` restricts all seven axes to a
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
restrict all axes to a region (e.g. the puzzle itself), reported as
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

## Round-2 test additions (12 tests total after round 3, all passing)

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

## Round 3 — bright-detail erasure (third real dump, 2026-07-24)

`20260724-123532` (glossy starry-night box, 4/4 frames aligned) revealed that
min-compositing under 1-3px misalignment ERASES small bright details (stars) —
and the darkening-based healing metric counted that destruction as benefit.
Fix: a **bright-speck retention** metric (white top-hat → threshold →
small-area connected components; retention ratio composite/reference, whole
frame and excluding healed patches), plus subtraction of dilated
reference-speck pixels from the darkening map so erased specks no longer
inflate glare healing. Real numbers: retention_excl_healed ≈ 1.01 (no-op dump
115252), 1.16 (sheen dump 115323 — healing revealed specks), **0.53 on
123532 — half the stars destroyed**; its darkened_fraction dropped
63.35% → 60.35% once erased specks stopped counting as healing. One new
combined test (starfield fixture: preserved vs erased dots, healing must not
credit erasure) brings the suite to 12.

## Round 4: glare-masked compositing (2026-07-24)

The first three rounds built and validated the *scoring* tool. Round 4 uses it to
evaluate a candidate *fix* for what Round 3 found: a global min-composite erases
fine bright detail under residual misalignment. Two more failure modes showed up
in the same real dumps while building the fix: matte glare is a desaturating
gray sheen that never approaches white (not new — Round 2 already covered this
for scoring — but it means a fix can't key off saturation either), and
background micro-parallax (e.g. carpet) that no single homography can fit
smears under compositing, independent of glare entirely. `stitch.py --mode
masked` (the existing `--mode app` is unchanged, still the default) addresses
all three by keeping compositing confined to a feathered glare mask instead of
applying it to the whole frame: SIFT+RANSAC registration refined with
`cv2.findTransformECC` for sub-pixel accuracy, per-frame photometric gain
compensation, a gain-corrected min-composite, and a mask built from
`max(0, reference - min_composite)` darkening (with a reference-brightness
floor to exclude moving shadows) that the min-composite only wins where it
actually indicates glare. See `README.md`'s
[`stitch.py --mode masked`, in detail](README.md#stitchpy---mode-masked-in-detail)
for the full per-step description; the point of this section is what tuning
against the real dumps found.

### ECC refinement: real, large improvement

Before trusting the masking logic at all, ECC's own directional convention had
to be nailed down (`cv2.findTransformECC(templateImage, inputImage, ...)` fits a
warp from TEMPLATE coordinates onto INPUT coordinates, the opposite direction
from `homography`'s corner → reference convention used everywhere else in this
file — see `refine_homography_ecc`'s docstring for the derivation). Once
correct, on `20260724-123532`'s 4 corner frames, ECC dropped mean central-crop
absdiff (the same metric `frame_is_verified` gates on) from 38-47/255
(SIFT+RANSAC alone) to 9-16/255 — the difference between "would fail
`--skip-unverified`'s default 18.0 threshold on 3 of 4 frames" and "comfortably
passes on all 4."

### The naive darkening threshold (10/255) blanketed 80-93% of the frame

The spec's starting point for `MASK_DARKENING_THRESHOLD` was 10 (matching
`common.DARKENED_PIXEL_THRESHOLD`, the per-pixel threshold `score_stitch`
already uses for its glare-healing benefit metric). At 10, the mask's alpha
exceeded 0.5 over 93% of `20260724-123532` and 76% of `20260724-115323` — not a
restricted glare region at all, essentially the whole frame. Bright-detail
retention (whole-frame, via `common.detect_bright_specks`) on `123532` was only
0.739 — barely better than `--mode app`'s own 0.644 — because the mask was
blanketing almost everything anyway.

The cause: a "darkening" floor with nothing to do with glare. Two sources,
both structural to any min-composite of several real photos:

1. **Order statistics.** The min of several independent noisy samples reads
   lower than any one sample, everywhere — not just where there's glare.
2. **Resampling softens fine bright detail.** `cv2.warpPerspective` interpolates
   a warped frame's pixels; for a small bright feature like a star, that
   dilutes its peak into its darker surroundings. The result reads as
   "darkening" exactly AT the star's own location, from the resample alone —
   which perversely means the naive mask targeted stars for "healing"
   specifically because resampling had already dimmed them, compounding the
   loss the mask was supposed to prevent.

A sweep (script not committed; see the threshold/floor grid this section's
numbers come from) over `MASK_DARKENING_THRESHOLD ∈ {10..40}` on both dumps
showed whole-frame bright-detail retention on `123532` jumping from 0.74 (at
10) to 0.97 (at 15) to >1.0 (at 20+) — a sharp knee, not a gradual slope,
consistent with "raising the threshold past the noise floor" rather than
"raising it past a real signal." `MASK_DARKENING_THRESHOLD` was set to **30**
(`MASK_BRIGHTNESS_FLOOR` stayed at the spec's starting point of 110;
`MASK_DILATE_RADIUS_PX`/`MASK_FEATHER_SIGMA` stayed at 15/8) — comfortably past
the knee, still far below the sustained, broad darkening a real sheen or
exposure mismatch produces (median blurred darkening was 22.6/255 on `123532`,
9.4/255 on `115323`, vs. p95 74/44 respectively for genuinely glare/exposure-
affected pixels).

### Final numbers, all three real dumps (`--mode masked` vs `--mode app`)

`app` columns are the SHIPPED APP'S OWN `composite.jpg` (not this tool's `--mode
app` reimplementation), scored directly — the same numbers as the
[Real-dump validation](README.md#real-dump-validation-2026-07-24) table in
`README.md`. `masked` columns are `stitch.py --mode masked`'s output, scored the
same way (composite copied into a scratch dump alongside the same
reference/corners, `score_stitch.py` run on that).

| Metric | `20260724-115252` app / masked | `20260724-115323` app / masked | `20260724-123532` app / masked |
|---|---:|---:|---:|
| Corner frames verified | 0/4 · 0/4 | 3/4 · 3/4 | 4/4 · 4/4 |
| Local ghosting `p95_shift_px` | 0.15 · 0.09 | **1.11 · 0.69** | 1.49 · 32.21† |
| Local ghosting `median_shift_px` | 0.08 · 0.04 | 0.41 · 0.05 | 0.37 · 1.32† |
| Local ghosting `worst_patch.shift_px` | 0.24 · 0.13 | 3.67 · 4.16 | 14.71 · 437.42† |
| Glare healing `darkened_fraction` | 0.04% · 0.00% | 49.88% · 4.76% | 60.35% · 50.00% |
| Glare healing `mean_darkening_over_darkened` | 9.62 · 0.00 | 30.27 · 27.06 | 36.34 · 38.87 |
| Bright detail `retention_ratio` | 1.008 · 1.053 | 1.140 · 0.954 | 0.644 · 1.159 |
| Bright detail `retention_ratio_excl_healed` | 1.008 · 1.053 | 1.155 · 1.002 | **0.525 · 1.067** |

† See caveat below — this is very likely a scoring-tool artifact on background
carpet texture, not a real quality regression in the masked composite itself.

Against the two targets set for this round:

- **`123532`: bright-speck `retention_ratio_excl_healed` ≥ 0.9 while the
  artwork glare still visibly heals.** Met: 1.067 (up from the app's 0.525),
  with `darkened_fraction` still 50.00% at a mean 38.87/255 — the sheen is
  still clearly, visibly healing (confirmed by eye in `scores/masked.jpg`
  against `reference.jpg`), just no longer at the cost of the stars.
- **`115323`: the matte sheen still heals, ghost p95 no worse than the app
  composite's.** Met: 0.69px vs. the app's 1.11px, `darkened_fraction` 4.76%
  at a mean 27.06/255 — heals a smaller, more targeted fraction of the frame
  (by design — that's the whole point of the mask) at comparable per-pixel
  magnitude to the app's own healing.
- `115252` (0/4 frames aligned in both) is an unaffected no-op in both modes,
  as expected — `--mode masked`'s composite is byte-identical to the
  reference when no frame passes verification.

### Caveat: `123532`'s local-ghosting numbers, carpet background

`123532`'s masked composite scores far WORSE on local ghosting than the app's
own composite (p95 32.21px vs. 1.49px; `worst_patch` 437px, which is larger
than the image itself — clearly degenerate). This was investigated directly
(`scores/masked_scores/worst_patch_flicker.png`,
`scores/masked_scores/ghost_heatmap.png`): the flagged patches sit in the
CARPET BACKGROUND surrounding the box, not inside the masked/healed box region
— and `scores/masked_scores/absdiff_heatmap.png` confirms the masked
composite is visually near-identical to the reference there (as expected: the
glare mask should leave alpha ≈ 0 on background carpet entirely). The
suspected cause is `cv2.phaseCorrelate`'s known instability on repetitive,
near-periodic texture (fabric weave) — a second, independent JPEG re-encode of
an already near-identical region is enough to nudge which DCT quantization bin
a block lands in, and phase correlation's FFT-based peak search can return a
degenerate, effectively-random large shift when the correlation surface is
nearly flat. This is plausibly a pre-existing `score_stitch.py` limitation
surfaced by this specific dump's heavy carpet texture, not something
`--mode masked` introduced — but it wasn't chased further because the task's
actual targets for `123532` (bright-detail retention, visible healing) don't
depend on this metric, and it would need its own investigation (e.g. a
patch-level periodicity/uniformity check alongside the existing
`PATCH_MIN_STD` gate) to fix properly. Flagged here rather than silently
tuned around.

### Reproduce

```bash
uv run python scripts/stitch_quality/stitch.py /path/to/dump --mode masked --out /tmp/masked.jpg
# writes /tmp/masked_mask.png too (the alpha mask, as a diagnostic)
uv run pytest scripts/stitch_quality/test_stitch_quality.py -v
```

## Round 5: median-based robust darkening for the mask (2026-07-24)

Round 4's glare mask keyed directly off `darkening = max(0, gray(reference) -
gray(min_composite))` — the same signal `score_stitch.py` uses for its glare-
healing benefit metric. That's an honest signal for SCORING (the min-composite
really did pick that darker pixel), but as the mask's own candidate-selection
signal it conflates real glare with a pure order-statistics artifact: the min
of several independent, sub-pixel-misaligned samples of a high-variance
texture (gray carpet) reads systematically lower than any individual sample,
everywhere, with nothing to do with glare. On `20260724-123532` (gray carpet,
median gray ≈135, above `MASK_BRIGHTNESS_FLOOR=110`), that noise floor alone
pushed the Round-4 mask's alpha above 0.5 over **93% of the carpet** — the
mean absdiff between the masked composite and the reference over background
carpet was 36/255, visibly smoothed/blocked, defeating the design goal that
non-glare regions stay pristine reference pixels.

### The fix: median instead of min

`stitch.compute_darkening_robust` replaces the mask's darkening signal (step 5
of [`stitch.py --mode masked`, in detail](README.md#stitchpy---mode-masked-in-detail)
above; the min-composite itself, and the final blend, are unchanged — this is
purely about WHERE the mask fires): at each pixel covered by ≥2 gain-corrected
verified warped frames, take the **median** of the covered frames' gray values
(exactly 2 covered frames: the **max**, i.e. the brighter one — a 2-sample
median is just their average, which understates darkening at a genuinely
glared pixel where one frame glares and the other shows the true dark
surface), then `darkening_robust = max(0, gray(reference) - median_gray)`.
Pixels covered by <2 frames get 0 (no mask, matching the pipeline's existing
bias toward leaving pixels alone when unsure). Rationale: glare moves between
shots (each corner frame glares, if at all, at a different spot — the whole
point of the 5-shot technique), so at a genuinely glared reference pixel MOST
covered frames show the true darker surface and the median stays dark; at an
unglared carpet pixel the median is a typical sample of that pixel's noisy
texture value, not the extreme minimum, so it doesn't inherit the min's
systematic downward bias. The rest of the mask pipeline (Gaussian blur sigma
3 → threshold + `MASK_BRIGHTNESS_FLOOR` → dilate 15px → feather sigma 8) is
unchanged, just fed this new signal instead.

### Threshold re-sweep on `20260724-123532`

With the darkening signal itself fixed, `MASK_DARKENING_THRESHOLD` was
re-swept over `{10, 15, 20, 30}` (registration/compositing run once and
reused across all four thresholds — see `stitch.register_and_composite_masked`
/ `blend_with_glare_mask`, split out for exactly this). Carpet vs. box regions
approximated as rows 21–77% × cols 25–76% = box (28.5% of the frame), the rest
carpet (71.5%):

| `MASK_DARKENING_THRESHOLD` | alpha>0.5, carpet | alpha>0.5, box | alpha>0.5, whole frame | `darkened_fraction` | mean darkening (darkened px) | `retention_ratio_excl_healed` | ghost `p95_shift_px` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 83.8% | 37.8% | 70.7% | 46.3% | 39.4/255 | 0.975 | 32.23px |
| 15 | 80.1% | 22.1% | 63.5% | 43.1% | 39.9/255 | 1.065 | 27.17px |
| 20 | 68.6% | 17.1% | 53.9% | 38.9% | 39.5/255 | 1.074 | 8.53px |
| 30 | 31.3% | 9.6%  | 25.1% | 23.3% | 36.9/255 | 1.084 | 0.46px |

(For comparison, Round 4's MIN-based signal at the same `MASK_DARKENING_THRESHOLD=30`
gave carpet alpha>0.5 of **92.6%** — the median fix's 31.3% is a large
reduction, just not all the way to the ~10% originally hoped for at any of
the four swept values.)

**30 was kept** (unchanged from Round 4, but now applied to a much-less-
contaminated signal): it has the lowest carpet leakage and the best local-
ghosting p95 of the four by a wide margin, `retention_ratio_excl_healed`
clears the 0.9 target at every swept value (0.975–1.084, monotonically
increasing with threshold), and `darkened_fraction`/mean-darkening at 30
(23.3%, 36.9/255) still show clear, visible healing — a lower threshold buys
more coverage at a steep carpet-leakage and ghosting cost for little
additional retention benefit (retention is already comfortably above target
at every value in the sweep).

**Caveat: even at 30, carpet leakage is not under the originally-hoped ~10%.**
31.3% of carpet still shows alpha > 0.5 at the best swept threshold. The
median substantially — but not completely — suppresses the order-statistics
floor: with only 4 frames and genuine (if small, 1-3px) residual registration
error, a texture edge shifting under warp can still put 2 of 4 frames on the
darker side of that edge at the same location often enough to produce a
median-based dip that clears even a 30/255 threshold at a meaningful fraction
of carpet pixels — this is no longer pure sensor-noise order statistics, it's
a real (if small) registration-driven appearance change on high-frequency
texture that a per-pixel median genuinely cannot fully separate from a
"typical" sample of that texture. Getting under ~10% would need either a
threshold outside the swept range (untested, and higher risks under-healing
real glare, whose own median darkening at threshold 30 was still comfortably
above the noise on both dumps below), more than 4 frames feeding the median,
or a texture-aware gate (e.g. excluding high-local-variance regions from mask
candidacy directly) — flagged as a next step rather than chased further here,
since the round's stated verification targets (bright-detail retention ≥ 0.9,
`115323` ghost p95 no worse than the app's) are both met.

**Side effect: this also resolves most of Round 4's "degenerate carpet
ghosting" caveat.** That caveat attributed `123532`'s masked-mode
local-ghosting blowup (p95 32px, `worst_patch` 437px) entirely to
`cv2.phaseCorrelate` instability on repetitive fabric texture, independent of
`--mode masked` itself. That explanation undersold how much the OLD mask's
~93% carpet coverage contributed: with most of the carpet partially
blended by the min-composite (not left as pure, untouched reference pixels),
phase correlation was comparing genuinely-perturbed local texture, not just
two independent JPEG re-encodes of identical content. With carpet leakage cut
to 31.3%, the masked composite's local-ghosting p95 on `123532` dropped to
**0.46px** and the worst patch to **5.24px** — both now BETTER than the app's
own composite (1.49px / 14.71px), and the worst patch is a plausible small
real registration error, not a 437px degenerate reading. The residual
phase-correlation sensitivity to repetitive texture likely still exists as a
minor contributor (see Round 4's caveat for the mechanism), but it is no
longer the dominant effect once most of the carpet is no longer touched by
the blend at all.

### Full evaluation, all three dumps (`--mode masked` vs `--mode app`)

Same protocol as Round 4's table: `app` columns score the shipped app's own
`composite.jpg` directly; `masked` columns score `stitch.py --mode masked`'s
freshly-regenerated output (`<dump>/scores/masked.jpg`, overwriting Round 4's
files), via a scratch dump directory (reference + corners + metadata,
`masked.jpg` copied in as `composite.jpg`) scored with `score_stitch.py --out
<dump>/scores/masked_scores/`.

| Metric | `20260724-115252` app / masked | `20260724-115323` app / masked | `20260724-123532` app / masked |
|---|---:|---:|---:|
| Corner frames verified | 0/4 · 0/4 | 3/4 · 3/4 | 4/4 · 4/4 |
| Local ghosting `median_shift_px` | 0.08 · 0.04 | 0.41 · 0.04 | 0.37 · 0.06 |
| Local ghosting `p95_shift_px` | 0.15 · 0.09 | **1.11 · 0.12** | 1.49 · **0.44** |
| Local ghosting `worst_patch.shift_px` | 0.24 · 0.13 | 3.67 · 2.79 | 14.71 · **5.24** |
| Local ghosting `healed_patches` | 0/768 · 0/768 | 232/768 · 11/768 | 386/768 · 102/768 |
| Glare healing `darkened_fraction` | 0.04% · 0.00% | 49.88% · 1.50% | 60.35% · 23.34% |
| Glare healing `mean_darkening_over_darkened` | 9.62 · 0.00 | 30.27 · 34.63 | 36.34 · 36.86 |
| Sharpness `laplacian_variance_ratio` | 1.189 · 1.059 | 1.088 · 1.040 | 0.930 · 0.804 |
| Gradient `gradient_magnitude_ratio` | 1.042 · 1.003 | 0.800 · 1.000 | 0.743 · 0.837 |
| Bright detail `retention_ratio` | 1.008 · 1.053 | 1.140 · 1.004 | 0.644 · 1.181 |
| Bright detail `retention_ratio_excl_healed` | 1.008 · 1.053 | 1.155 · 1.039 | **0.525 · 1.130** |
| Mask coverage (alpha>0.5), whole frame | n/a · 0.00% | n/a · 2.58% | n/a · 25.13% |
| Mask coverage (alpha>0.5), carpet region | n/a · n/a | n/a · n/a | n/a · **31.33%** |
| Mask coverage (alpha>0.5), box region | n/a · n/a | n/a · n/a | n/a · **9.60%** |

Against the round's verification targets:

- **`115323`: sheen still heals, ghost p95 no worse than the app composite's.**
  Met, by a wide margin: 0.12px vs. the app's 1.11px (Round 4's MIN-based
  signal already got 0.69px here; the median-based robust darkening pushed it
  further down), `darkened_fraction` 1.50% at a mean 34.63/255 — a smaller, more targeted
  fraction of the frame heals (by design) at a comparable-to-larger per-pixel
  magnitude than the app's own healing, and `retention_ratio_excl_healed`
  (1.039) stayed comfortably ≥ 1.
- **`123532`: bright-speck `retention_ratio_excl_healed` ≥ 0.9 while the
  artwork glare still visibly heals, and carpet alpha>0.5 fraction < 10%.**
  Partially met: retention is 1.130 (target cleared with room to spare, up
  from the app's 0.525), and the sheen still clearly, visibly heals
  (`darkened_fraction` 23.34% at a mean 36.86/255, confirmed by eye in
  `scores/masked.jpg` against `reference.jpg`) — but carpet leakage is 31.33%,
  not under 10% (see the caveat above). This is the one target this round did
  not fully close, despite a large improvement over Round 4's 92.6%.
- `115252` (0/4 frames verified in both) is an unaffected no-op in both modes,
  as expected — mask coverage 0.00%, masked composite byte-identical to the
  reference.

### New test

`test_masked_mode_median_darkening_ignores_textured_background_misalignment`
(see `README.md`'s [Tests](README.md#tests) section) directly reproduces the
defect on a synthetic fixture: a high-contrast textured background sampled by
4 frames each shifted ~1-2px (the residual-misalignment scale, not raw corner
jitter) plus a moving glare disc. Asserts genuine glare still triggers the
mask while the textured background stays essentially unmasked under the
median signal, then sanity-checks that a plain MIN-based signal over the SAME
fixture WOULD have flagged much of that background — proving the fixture
actually exercises the fix. Brings the suite to **16 tests**, all passing.

### Reproduce

```bash
uv run python scripts/stitch_quality/stitch.py /path/to/dump --mode masked --out /tmp/masked.jpg
# CLI now also prints "Mask coverage (alpha > 0.5): NN.NN% of pixels"
uv run pytest scripts/stitch_quality/test_stitch_quality.py -v
```

## Round 6: a stricter vote for the robust darkening estimate (2026-07-24)

Round 5's median was a real improvement but didn't close the carpet-leakage gap
(see item 6 below, before this round): the underlying reason turned out to be
where the residual registration error concentrates, not just how much of it
there is. `frame_is_verified`'s central-crop absdiff gate only checks the
frame's central 50% — a warp can pass verification with a comfortable margin
there while still carrying several times more pixel error near the frame's
edges and corners, which is exactly where a lot of carpet background lives (a
puzzle box sits roughly centered; the carpet fills the border). At a border
carpet pixel with 3 covering frames, it's common for exactly 2 of them to
coincidentally land on the same side of a shifted texture edge — a median of
3 treats that 2-of-3 agreement as "most frames" and lets it through, exactly
the failure mode this round targets.

**The fix**: `compute_darkening_robust` replaces the median with an explicit
vote, unchanged at `n=2` (already the max, per Round 5) but tightened above
that:

- **2 or 3 covered frames → the max (brightest) covered gray** — an ALL-of-N
  vote. Every single covered frame must read darker than the reference for a
  pixel to register as darkened at all; one frame reading bright vetoes it.
  This is the material change from Round 5, whose median-of-3 was, in effect,
  a 2-of-3 vote (the median sits with whichever two of the three samples are
  closer together).
- **4 covered frames → the second-brightest covered gray** — a 3-of-4 vote,
  one notch looser than all-of-4, because four aligned frames is common
  enough on a real capture that requiring literal unanimity would make the
  mask too eager to reject on a single coincidentally-bright reading (e.g. a
  corner shot whose own glare happens to land at the same spot the reference
  glares). Round 5's median-of-4 was, in effect, closer to a 2-of-4 vote (the
  median of 4 averages the 2nd and 3rd values).

Real glare still clears this bar easily: it moves between shots by
construction (each corner frame glares, if at all, at a different spot — the
whole point of the 5-shot technique), so a genuinely glared reference pixel
typically has ALL (or all-but-one, at 4 frames) covering frames show the true
darker surface, not just a bare majority.

### Numbers, before (Round 5) → after (Round 6)

`20260724-123532` (4/4 frames aligned, the dump with the most carpet in
frame): mask coverage (alpha > 0.5) whole-frame dropped **25.13% → 13.03%**;
carpet-region (outside the puzzle box, rows 21-77% × cols 25-76%) **31.33% →
16.71%**; box (the puzzle itself) **9.60% → 3.80%**. Healing itself stayed
intact — `darkened_fraction` **23.34% → 14.42%**, mean darkening over
darkened pixels **36.86/255 → 33.19/255** (the sheen is still clearly,
visibly healing), `retention_ratio_excl_healed` **1.130 → 1.128** (comfortably
above the 0.9 target throughout). Local ghosting improved alongside the
leakage drop — `p95_shift_px` **0.443px → 0.300px**.

`20260724-115323` (3/4 frames aligned, the matte-sheen dump): `darkened_fraction`
**1.50% → 0.77%**, mean darkening **34.63/255 → 38.27/255**,
`retention_ratio_excl_healed` **1.039 → 1.060**, `p95_shift_px` **0.116px →
0.098px** (already small on the Python side both rounds — Round 6's real
payoff on this dump was on the **Swift port**, below).

`20260724-115252` (0/4 aligned in both rounds): unaffected no-op, as expected.

**Caveat, softened but not eliminated**: `123532`'s carpet leakage is down to
16.71%, well below Round 5's 31.33%, but still not under the ~10% originally
hoped for. The vote is a coarser, per-pixel-independent tool than a true
texture-aware gate; a stray 3-of-4 (or all-of-3) agreement on a shifted
texture edge can still happen often enough on a busy, high-frequency
background to clear the threshold at a nontrivial fraction of pixels. See
"Known gaps" below.

### Swift port validation

The iOS app's own masked-healing implementation (ported 1:1 from this
recipe, `ios/Pussel/Features/Capture/GlareFree/GlareFreeMaskedHealing.swift`)
received the identical vote change. Its own real-dump regression was more
dramatic than the Python side's: `20260724-115323`'s local-ghosting `p95_shift_px`
(scored the same way, via `score_stitch.py` against a `swift_masked.jpg`
copied in as `composite.jpg`) dropped from **11.42px to 0.46px** — the same
border-registration-error mechanism as above, but the Swift composer's own
Vision-based registration (unchanged, per its own verification gate also
being central-50%-only) apparently left even more per-pixel edge/corner error
on this specific carpet-heavy dump than Python's SIFT+ECC did, so Round 5's
median-based mask leaked far more heavily there. `20260724-123532` stayed
comfortably passing (`retention_ratio_excl_healed` 1.106, `p95_shift_px`
0.387px). See the Swift composer's own `robustDarkening` doc comment for the
identical vote rationale in Swift.

### Reproduce

```bash
uv run python scripts/stitch_quality/stitch.py /path/to/dump --mode masked --out /tmp/masked.jpg
uv run pytest scripts/stitch_quality/test_stitch_quality.py -v
```

## Known gaps / next steps

1. **Only three real dumps exist.** The "expected value ranges" in the README
   now include both synthetic AND these three real numbers, but three captures
   (one clean no-op, one glare-sheen, one glossy starfield box) isn't a
   calibration set — revisit `HEALED_PATCH_DARKENING_THRESHOLD` and the
   qualitative good/bad guidance as more real dumps accumulate.
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
5. **`--mode masked`'s `MASK_DARKENING_THRESHOLD` was tuned (both rounds) against
   the same two dumps** (`115323`, `123532`) that motivated it -- the same
   too-few-real-dumps caveat as (1), sharpened: a threshold picked to satisfy
   two specific captures' numeric targets isn't validated against a matte
   sheen or glossy surface the tuning set didn't include. Revisit once more
   real dumps accumulate, the same way `HEALED_PATCH_DARKENING_THRESHOLD` will.
6. **`123532`'s masked composite still leaks the glare mask onto ~17% of the
   carpet background** (down from Round 4's 92.6%, Round 5's 31.33%, to
   Round 6's 16.71% -- see Round 6's numbers above), still above the ~10%
   originally hoped for. Round 6's vote (all-of-N at 2-3 covered frames,
   3-of-4 at 4) is stricter than Round 5's median but still per-pixel and
   texture-blind: a stray 3-of-4 (or all-of-3) agreement on a shifted
   texture edge can still happen often enough on a busy, high-frequency
   background to clear the threshold at a nontrivial fraction of pixels,
   especially since registration error is largest right at the frame
   border where most of this dump's carpet lives (the verification gate
   only checks the central 50%). Getting further would need either a
   texture/local-variance-aware mask gate, tightening (or region-varying)
   the verification gate itself so border error is bounded too, or
   accepting a higher (untested) `MASK_DARKENING_THRESHOLD` at some healing
   cost -- not chased further here since Round 6's actual pass/fail targets
   (bright-detail retention, `115323`/`123532` ghosting, on both the Python
   and Swift implementations) were all met without it.

## Reproduce

```bash
cd network
uv run pytest scripts/stitch_quality/test_stitch_quality.py -v
uv run python scripts/stitch_quality/score_stitch.py /path/to/real/dump
uv run python scripts/stitch_quality/score_stitch.py /path/to/real/dump --quad "0.28,0.26 0.72,0.26 0.72,0.74 0.28,0.74"
uv run python scripts/stitch_quality/stitch.py /path/to/real/dump --out /tmp/restitched.jpg
uv run python scripts/stitch_quality/stitch.py /path/to/real/dump --out /tmp/masked.jpg --mode masked
```
