# Experiment 29: Glare-Free Stitch Quality Benchmark

## Objective

The iOS app's glare-free capture flow takes 5 photos of an assembled puzzle (1
centered reference + 4 corner-offset shots), registers each corner shot onto
the reference with a homography, and min-composites them (darkest pixel wins)
to remove glare. Misregistration shows up as ghosting/double edges in the
composite and confuses downstream piece matching. This experiment is an
**offline benchmark tool**, not a model-training experiment: it scores how
well that stitch worked on one concrete capture, given a DEBUG-build dump
directory (`reference.jpg`, `corner_1.jpg`..`corner_4.jpg`, `composite.jpg`,
optional `metadata.json`).

## Files

- `common.py` — shared utilities: dump loading + working-size normalization
  (`load_dump`), SIFT + ratio-test + RANSAC-homography matching
  (`match_sift_ransac`), sharpness/edge/saturation primitives (all
  mask-aware, for `--quad`), the per-pixel darkening map
  (`compute_darkening_map`), small-bright-speck detection
  (`detect_bright_specks`), the `--quad` parser/rasterizer (`parse_quad`,
  `quad_to_mask`), and the patch-grid phase-correlation helper
  (`phase_correlation_grid`).
- `score_stitch.py` — scores an existing `composite.jpg` against
  `reference.jpg`: global geometry, local ghosting, edge doubling, sharpness,
  glare reduction, glare healing, bright detail preservation. Writes
  `metrics.json` + 3 diagnostic images. Optional `--quad` restricts all seven
  axes to a region, reported alongside the full-frame numbers.
- `stitch.py` — an independent, offline Python reimplementation of the
  app's stitch, in two modes: `--mode app` (default; highlight-capping →
  SIFT+RANSAC per corner → warp with white fill → min-composite), so we can
  rebuild a second composite from the raw `corner_N.jpg` shots and score it
  the same way, e.g. to check whether a registration failure the app hit is
  a capture-pipeline bug or an inherently hard scene; and `--mode masked`
  (ECC-refined registration → photometric gain compensation →
  gain-corrected min-composite → a feathered glare mask that restricts
  compositing to regions that actually show glare, see below).
- `test_stitch_quality.py` — synthetic self-tests (no fixtures on disk); see
  [Tests](#tests) below.

## Usage

All commands run from `network/`:

```bash
# Score the app's own composite against the reference
uv run python scripts/stitch_quality/score_stitch.py /path/to/dump
uv run python scripts/stitch_quality/score_stitch.py /path/to/dump --out /path/to/report

# Restrict stats to a region (e.g. the puzzle itself), reported alongside full-frame --
# unit coordinates (0-1), clockwise from top-left, in the working-size frame:
uv run python scripts/stitch_quality/score_stitch.py /path/to/dump \
    --quad "0.28,0.26 0.72,0.26 0.72,0.74 0.28,0.74"

# Rebuild the composite offline from the raw corner shots and compare
uv run python scripts/stitch_quality/stitch.py /path/to/dump --out /tmp/restitched.jpg
uv run python scripts/stitch_quality/stitch.py /path/to/dump --out /tmp/restitched.jpg --skip-unverified

# `--mode masked`: ECC-refined registration + gain compensation + a glare mask that
# restricts compositing to actual glare regions (see below) -- also writes a diagnostic
# alpha-mask PNG next to --out (<out-stem>_mask.png)
uv run python scripts/stitch_quality/stitch.py /path/to/dump --out /tmp/masked.jpg --mode masked

# Score the offline reimplementation's output the same way, by pointing a
# second dump directory's composite.jpg at it (or just diff the two
# metrics.json files for the same reference/corners)
```

`score_stitch.py` also runs on **6 loose images** with no `metadata.json` —
just `reference.jpg`, `corner_1.jpg`..`corner_4.jpg`, `composite.jpg` in a
directory; missing corner shots (0-4 present) are fine too.

## What each metric means

All scoring runs at a normalized working size (long side 2048px by default)
so numbers are comparable across capture resolutions.

### Global geometry

SIFT keypoints on `composite.jpg` and `reference.jpg`, Lowe's-ratio-test
matches, then a RANSAC homography between them. Since the composite is
supposed to already live in the reference's coordinate frame, this
homography should be close to identity. The reported deviation is **not**
the homography's own RANSAC residual — it's the raw pixel distance between
each RANSAC-inlier match's composite and reference coordinates (`RMS` and
`p95`), which is what a global rigid misregistration actually looks like in
pixels, independent of whether a homography happens to fit the noise.

- `n_ratio_matches` / `n_inliers` / `inlier_ratio` — match quality; a low
  inlier ratio usually means the scene is too repetitive or low-texture for
  SIFT, not that the stitch itself is bad.
- `identity_deviation_rms_px` / `identity_deviation_p95_px` — None when
  there were too few matches (< 4) to attempt RANSAC at all.
- With `--quad`, the region report reuses the SAME SIFT+RANSAC fit (not a
  fresh one) and just subsets the inlier matches whose composite-side point
  falls inside the region — so region and full-frame numbers are directly
  comparable, not independently noisy fits. `n_keypoints_*` stay full-frame
  (SIFT isn't rerun per region).

### Local ghosting (the primary ghosting signal)

The working-size image is divided into a grid of `patch_size` (default 64px)
patches. For each patch, `cv2.phaseCorrelate` estimates the sub-pixel
translational shift between the composite and reference patch. Patches with
low reference-side contrast (blank cardboard, sky, a glare wash) are marked
invalid and excluded — phase correlation is unreliable there.

**Patches that were substantially DARKENED rather than misaligned are also
excluded**, as `healed`, separately from `invalid`: phase correlation reacts
to any appearance change, and a region the composite legitimately healed
(glare removed) can produce a large "shift" reading despite there being no
real registration error there at all. A patch counts as healed when its mean
value in the darkening map (see below) exceeds
`common.HEALED_PATCH_DARKENING_THRESHOLD` (25/255 by default — tuned against
a real capture with a broad glare sheen, see
[Real-dump validation](#real-dump-validation-2026-07-24)). Reported:
`median_shift_px` / `p95_shift_px` (over valid, non-healed patches only),
`healed_patches` (count excluded this way), and the single `worst_patch`
(used to render `worst_patch_flicker.png`) — also restricted to non-healed
patches, so the flicker crop lands on genuine ghosting instead of the most
dramatically healed region.

This is the most sensitive signal in practice: global geometry can look
fine (average pixel motion near zero) while one small region has 10+ px of
ghosting, because that region's contribution is diluted by the rest of the
image in aggregate/global metrics. `ghost_heatmap.png` renders every
patch's shift magnitude over the composite (matplotlib, `inferno`
colormap, capped at 8px); healed patches render as a translucent green
overlay instead, so they read visually as "explained by healing" rather
than "unmeasured" (no overlay) or "real ghosting" (inferno color).

### Edge doubling

`canny_edge_ratio` and `gradient_magnitude_ratio` (composite / reference).
Doubled edges from misregistration nominally push both above ~1 — but in
practice this is a coarser, noisier signal than local ghosting: it's a
single global number diluted by the whole image (including any white-filled
margins from the corner-shot warp), so a ghosting artifact confined to a
small region may barely move it. Treat it as a supporting signal, not the
primary read; local ghosting and the flicker crop are more sensitive.

### Sharpness

`laplacian_variance_ratio` (composite / reference), via variance of the
Laplacian. A min-composite of several photos is inherently a little softer
than a single photo (parallax and normal capture blur between different
frames average up as their sharpest-per-pixel selection still resamples
each frame through a homography warp), so ratios noticeably below 1 are
expected and not necessarily concerning by themselves — read this alongside
the ghosting metrics, not in isolation.

### Glare reduction (near-saturation)

`reference_saturated_fraction` / `composite_saturated_fraction`: fraction of
near-saturated pixels (gray ≥ 250 after a slight blur, to ignore isolated
sensor speckle) in each image, and `reduction_factor` = reference / composite
fraction. This exists so alignment quality can't be gamed by a stitch that
just returns the reference unmodified — but it is **blind to matte-print
glare**: a real capture (`20260724-115323`, see below) showed an obvious
glare sheen by eye that never once reached gray 250, so this metric reported
0.000% on both images even though the composite genuinely, visibly healed
it. Use glare healing (below) as the primary benefit metric; keep this as a
secondary signal for the (rarer) case of outright blown highlights.

### Glare healing — the primary benefit metric

The min-composite can only ever pick a darker (or equal) pixel than the
reference at any location — it never brightens. So the **darkening map**,
`max(0, blur(reference) - blur(composite))` (a slight blur first suppresses
resample/JPEG noise), is an unambiguous, saturation-independent signal of
what the stitch actually changed. From it:

- `darkened_fraction` — fraction of pixels darkened by more than
  `common.DARKENED_PIXEL_THRESHOLD` (8/255).
- `mean_darkening_over_darkened` — mean darkening magnitude, restricted to
  those darkened pixels.
- `p95_darkening` — 95th percentile of darkening over the WHOLE region (not
  just the darkened pixels) — the overall right tail.

This is the metric that catches matte-print glare (a desaturating gray wash
that never saturates) that near-saturation reduction misses entirely, and
like glare reduction, a stitch can't win it by returning the reference
unmodified (then darkening would be ~0 everywhere).

The SAME darkening map also drives the local-ghosting healed-patch exclusion
above — one computation, two uses.

**Reference bright specks are excluded before this metric is computed.** A
min-composite can delete a small bright detail (e.g. a star on a dark
background) wholesale: a 1-3px misregistration is enough for a neighboring
frame's darker background pixel to consistently win the darkest-pixel-wins
comparison at that exact location. Left alone, that erasure reads as
legitimate darkening — "healing" — even though nothing was healed, just
deleted (see [Bright detail preservation](#bright-detail-preservation)
below, which is what actually measures this). `score_dump` detects small
bright specks in the reference via `common.detect_bright_specks`, dilates
that mask by `common.SPECK_TOPHAT_KERNEL_PX` (the darkening map's own
Gaussian blur, `common.DARKENING_BLUR_SIGMA`, spreads a deleted speck's
brightness difference a couple pixels past the speck's own detected pixels,
so the plain mask alone would leave a thin residual-darkening halo at each
speck's rim), and zeroes the darkening map there before `compute_glare_healing`
runs (`score_stitch._darkening_map_excluding_specks`). Local ghosting's
healed-patch classification is unaffected — it still uses the unmodified
darkening map. On `20260724-123532` (see
[Real-dump validation](#real-dump-validation-2026-07-24) below), this dropped
`darkened_fraction` from 63.35% to 60.35% — the gap is exactly the credit the
metric was previously giving the composite for erasing ~3,200 stars.

### Bright detail preservation

Small bright specks (e.g. stars on a dark background) are detected
independently in the composite and the reference via a white top-hat filter
(`common.detect_bright_specks`: `cv2.morphologyEx(..., cv2.MORPH_TOPHAT, ...)`
with a `common.SPECK_TOPHAT_KERNEL_PX`-side ellipse kernel, thresholded at
`common.SPECK_BRIGHTNESS_THRESHOLD`, then connected-component-filtered to
`common.SPECK_MIN_AREA_PX`–`common.SPECK_MAX_AREA_PX` px areas so JPEG/resample
texture noise and larger bright features don't count as "specks"). This
exists because a 1-3px misregistration lets one frame's dark-sky pixel
consistently overwrite another frame's star in the darkest-pixel-wins
composite — a *deletion* failure mode the other six axes barely register
(edge/gradient ratios and sharpness are diluted by the whole image; global
geometry and local ghosting measure position, not presence).

Reported, both whole-frame and restricted to non-healed patches:

- `reference_speck_count` / `composite_speck_count` — raw speck counts in
  each image.
- `retention_ratio` — `composite_speck_count / reference_speck_count`. 1.0 is
  perfect retention; well below 1.0 means specks were lost.
- `*_excl_healed` variants — the same counts and ratio, but restricted to
  specks whose centroid falls outside any patch `LocalGhostingMetrics`
  excluded as "healed" (see Local ghosting above). A speck that was actually
  under glare was unrecoverable regardless of stitch quality, so counting its
  loss against the composite would penalize correct behavior — this is the
  ratio that isolates deletion caused by misregistration specifically.

Detection is intentionally count-based rather than spatially matching
individual specks between images: a speck can shift a pixel or two along
with the rest of a misregistered region while still being clearly present,
and exact spatial matching would conflate "moved slightly" with "erased" —
not the failure mode this metric targets.

## `--quad`: region focus

Ghosting/edge/sharpness stats over the full frame are easily dominated by
a large textured background (carpet, table) that has nothing to do with the
puzzle — a real capture showed local ghosting p95 drop from being computed
over 768 mostly-background patches to a puzzle-focused 160-patch region with
a *higher*, more honest p95 (the background's mostly-flat, near-zero-shift
patches were diluting the aggregate). `--quad "x1,y1 x2,y2 x3,y3 x4,y4"`
(unit coordinates 0-1, clockwise from top-left, in the working-size frame)
restricts all seven axes to a quadrilateral region — reported as `region` in
`metrics.json`, alongside the unchanged full-frame numbers (`quad` records
the coordinates used). Diagnostic images stay full-frame; `--quad` only
affects the reported statistics.

## Diagnostic images

Written to `<out>/` (default `<dump_dir>/score_stitch_output/`):

- `metrics.json` — the full report, same fields as printed to stdout.
- `ghost_heatmap.png` — per-patch shift magnitude over the composite; green
  overlay = healed (excluded from stats).
- `absdiff_heatmap.png` — colorized `|composite - reference|` (grayscale,
  `cv2.COLORMAP_INFERNO`); ghosted regions, healed regions, and
  warp-boundary seams all show up as bright (this one doesn't distinguish
  them — use `ghost_heatmap.png` / `metrics.json` for that).
- `worst_patch_flicker.png` — the composite and reference crops around the
  single worst-shift patch (excluding healed ones), side by side (upscaled
  4x), for visual confirmation of what the numbers are describing.

## `stitch.py`'s reimplementation, in detail

For each present `corner_N.jpg`: build a grayscale copy with pixels above
`~0.55` normalized gray (140/255) clamped flat *before* feature detection
only, mirroring the app's highlight-capping (glare blows a region to
near-white, which otherwise produces a cluster of spurious, poorly localized
SIFT keypoints along its rim) — the original color corner image is still
what gets warped and composited. Then: SIFT + ratio-test + RANSAC
homography onto the (similarly capped) reference, `cv2.warpPerspective` the
**original** corner image with that homography, fill any region the corner
image doesn't cover with white (white never wins a darkest-pixel-wins
comparison), and min-composite onto a running result seeded with the
reference.

`--skip-unverified` adds a sanity gate: after warping, compare mean
grayscale absdiff in the image's central 50% (a region every shot should
cover regardless of registration quality) against `--unverified-threshold`
(default 18.0); frames that fail are dropped from the composite entirely
instead of corrupting it with a bad registration.

This is **not** a byte-identical port of the Swift pipeline — it exists to
let us iterate on the stitching approach offline in Python, and to give
`score_stitch.py` a second composite to compare the app's own output against.

## `stitch.py --mode masked`, in detail

Motivated by three failure modes the exp29 benchmark tool exposed on real
captures (see [Round 4 in `HANDOFF.md`](HANDOFF.md#round-4-glare-masked-compositing-2026-07-24)): a global
min-composite erases fine bright detail (stars) under 1-3px residual
misalignment; matte glare is a desaturating gray sheen, never full white;
and background micro-parallax a homography can't fit (e.g. carpet) smears
under compositing. `--mode masked` keeps compositing confined to regions
that actually show glare, leaving pristine reference pixels everywhere
else:

1. **Registration**: the same SIFT + ratio-test + RANSAC homography as
   `--mode app`, then sub-pixel refinement with `cv2.findTransformECC`
   (`MOTION_HOMOGRAPHY`, on grayscale copies downscaled to
   `ECC_DOWNSCALE_LONG_SIDE` — full working-size ECC is slower and more
   easily trapped in a local optimum by fine texture). ECC failures
   (non-convergence, a singular Hessian) fall back to the SIFT+RANSAC
   homography unchanged — refinement is a bonus, never a regression. On the
   real dumps below, ECC dropped mean central-crop absdiff from ~40/255
   (SIFT alone) to ~10-16/255.
2. **Verification**: the same central-crop absdiff gate as
   `--skip-unverified`, but always on — there's no toggle, since a
   badly-registered frame poisoning a masked-blend region is exactly as bad
   as poisoning a plain min-composite.
3. **Photometric gain compensation**: per verified frame, a median
   reference/warped gray ratio over covered, mid-tone (`GAIN_GRAY_LOW`-
   `GAIN_GRAY_HIGH`) pixels corrects an exposure/white-balance mismatch
   between that corner shot and the reference — otherwise darkest-pixel-wins
   can be won by whichever frame is globally darker, not by which pixel
   actually has less glare.
4. **Min-composite**: gain-corrected, verified frames only, at pixels at
   least one of them covers; pixels none of them cover fall back to the
   reference (unlike `--mode app`, the reference itself is not seeded into
   the running min here — its contribution happens entirely through the
   blend in step 6).
5. **Glare mask**: a **robust darkening estimate**, `compute_darkening_robust`
   (see [Round 5](HANDOFF.md#round-5-median-based-robust-darkening-for-the-mask-2026-07-24)
   and [Round 6](HANDOFF.md#round-6-a-stricter-vote-for-the-robust-darkening-estimate-2026-07-24)
   in `HANDOFF.md` for why this replaced a plain
   `max(0, gray(reference) - gray(min_composite))` signal, and then why the
   median itself was replaced with a vote) — at each pixel covered by ≥2
   gain-corrected verified frames, take a **vote** among the covered frames'
   gray values: 2 or 3 covered, the brightest (an ALL-of-N vote — every
   covered frame must read darker than the reference); exactly 4 covered,
   the second-brightest (a 3-of-4 vote, tolerating one frame that happens to
   share the reference's own glare position); pixels covered by <2 frames
   get 0. `darkening_robust = max(0, gray(reference) - vote_gray)`. Blurred
   (`MASK_DARKENING_BLUR_SIGMA`) and thresholded at `MASK_DARKENING_THRESHOLD`
   **AND** `gray(reference) > MASK_BRIGHTNESS_FLOOR` — the brightness floor
   excludes a moving hand shadow (dark in the reference already) from ever
   reading as glare, since glare heals FROM a bright, washed-out reference
   pixel. The raw mask is dilated (`MASK_DILATE_RADIUS_PX`) to comfortably
   cover a glare patch's soft rim, then feathered with a Gaussian blur
   (`MASK_FEATHER_SIGMA`) into a smooth `[0, 1]` alpha.
6. **Output**: `reference * (1 - alpha) + min_composite * alpha` — the
   min-composite (not a median/vote composite) is still what actually fills
   the masked region, since it heals best; the vote in step 5 only decides
   WHERE the mask applies. The alpha mask is also written as
   `<out-stem>_mask.png` next to `--out`, as a diagnostic, and the CLI prints
   the fraction of pixels with alpha > 0.5 ("mask coverage").

`MASK_DARKENING_THRESHOLD` needed to be tuned well above the spec's naive
starting point of 10/255: real captures show a systemic, glare-unrelated
"darkening" floor from two sources — taking the min of several noisy photos
reads lower than any one of them everywhere (an order-statistics effect),
and `cv2.warpPerspective`'s resampling softens fine bright detail (diluting
a star's peak into its darker neighbors, which shows up as "darkening"
exactly where the star is, from the resample alone). At threshold 10, that
floor was large enough to blanket 80-93% of the frame in alpha and erase the
same fine detail the mask was meant to protect. See `stitch.py`'s
`MASK_DARKENING_THRESHOLD` docstring and
[Round 4 in `HANDOFF.md`](HANDOFF.md#round-4-glare-masked-compositing-2026-07-24) for the full tuning story and
measured numbers on all three real dumps. **Round 5** replaced the mask's
darkening signal itself (step 5 above) with the median-based
`compute_darkening_robust`, which suppresses most (not all) of that same
order-statistics floor directly on a highly-textured background (e.g.
carpet) — see
[Round 5 in `HANDOFF.md`](HANDOFF.md#round-5-median-based-robust-darkening-for-the-mask-2026-07-24)
for the re-swept threshold numbers and the residual-leakage caveat. **Round
6** found that Round 5's residual leakage was concentrated near frame
borders, where registration error is largest (the verification gate only
checks the central 50%) and a plain median let 2-of-3 or 2-of-4 covered
frames' coincidental agreement through; replacing the median with an
explicit all-of-N (2 or 3 covered) / 3-of-4 (4 covered) vote in
`compute_darkening_robust` closed most of that gap — see
[Round 6 in `HANDOFF.md`](HANDOFF.md#round-6-a-stricter-vote-for-the-robust-darkening-estimate-2026-07-24)
for the before/after numbers.

## Real-dump validation (2026-07-24)

Scored against the first two real DEBUG-build dumps
(`~/Pictures/puzzles/glare_stitch_dumps/GlareFreeDumps/`, not committed —
images stay local per the dump tool's own convention):

| Metric | `20260724-115252` (clean) | `20260724-115323` (glare sheen) |
|---|---:|---:|
| Global geometry `identity_deviation_p95_px` | 0.35 px | 1.80 px |
| Local ghosting `p95_shift_px` | 0.15 px | 1.11 px |
| Local ghosting `worst_patch.shift_px` | 0.24 px | 3.67 px |
| Local ghosting `healed_patches` | 0 / 768 | 232 / 768 |
| Glare reduction `reduction_factor` (near-saturation) | 1.00x (no glare) | 1.00x (**blind** — see below) |
| Glare healing `darkened_fraction` | 0.04% | 49.88% |
| Glare healing `mean_darkening_over_darkened` | 9.62/255 | 30.27/255 |

`20260724-115323`'s reference has an obvious, broad, desaturating glare
sheen over the puzzle card by eye. It never reaches gray 250 — near-
saturation glare reduction reports 0.000% on both images, completely missing
it — while glare healing correctly reports ~50% of pixels meaningfully
darkened. (Note this capture also had a global exposure/white-balance
mismatch between the reference and corner shots that darkens the ENTIRE
frame, including background carpet far from the puzzle — not just the
glare itself; `darkened_fraction` reflects that too, which is why it's so
high. `--quad` restricted to just the puzzle card showed a similar picture,
confirming the effect isn't purely a background artifact.)

Before the healed-patch exclusion fix, this same dump's worst local-ghosting
patch (row 12, col 15, shift 11.56px) was the glare sheen itself — visually
confirmed via `worst_patch_flicker.png` to be a pure brightness/appearance
difference with no positional edge shift at all. After the fix (232 patches
excluded as healed), the worst patch moved to row 14, col 16 (shift 3.67px)
— visually confirmed to show a genuine doubled edge (the puzzle card's
border and a hair curl sit at slightly different positions between composite
and reference). `HEALED_PATCH_DARKENING_THRESHOLD = 25.0` was chosen
empirically from this dump: comfortably above the ~18-27/255 baseline
darkening seen even on far-background patches unrelated to the sheen, but
below the ~40-76/255 range of the confirmed healed-sheen cluster.

### Third real dump — bright detail preservation caught a deletion the other six axes missed

A third real dump, `20260724-123532` (a glossy Ravensburger box, starry-night
artwork), motivated the [Bright detail preservation](#bright-detail-preservation)
metric above. By eye, the composite is visibly missing a large fraction of
the small stars present in the reference — but the pre-existing six axes
barely hinted at it (`gradient_magnitude_ratio` 0.743, `laplacian_variance_ratio`
0.930 — both read as "a little softer", the ordinary min-composite cost
described under [Sharpness](#sharpness), not "detail deleted"):

| Metric | `20260724-115252` (clean) | `20260724-123532` (stars deleted) |
|---|---:|---:|
| Bright detail `retention_ratio` (whole frame) | 1.008 | **0.644** |
| Bright detail `retention_ratio_excl_healed` | 1.008 | **0.525** |
| Bright detail `reference_speck_count` / `composite_speck_count` | 19273 / 19427 | 8997 / 5792 |
| Local ghosting `healed_patches` | 0 / 768 | 386 / 768 |
| Glare healing `darkened_fraction` | 0.04% | 60.35% |

`retention_ratio_excl_healed` (0.525) is even lower than the whole-frame
ratio (0.644) — the loss isn't confined to patches already excluded as
"healed" (which would mean the stars were simply under unrecoverable glare);
most of it is in patches local ghosting considers clean, i.e. genuine
1-3px-misregistration erasure this metric was built to catch. Glare
healing's `darkened_fraction` (60.35%) reflects the same broad
exposure/white-balance mismatch seen in `20260724-115323` above, not the
star deletion specifically — which is exactly why the darkening map has
detected reference specks subtracted out before that number is computed
(see [Glare healing](#glare-healing--the-primary-benefit-metric) above):
without the subtraction, this dump's `darkened_fraction` reads 63.35%,
crediting the composite for ~3,200 deleted stars as if they'd been healed.

## Expected value ranges

Measured on the synthetic self-test fixture (see [Tests](#tests)) — a
textured scene with per-shot glare discs, ground-truth-registered ("aligned")
vs. a deliberately ~8px-shifted frame ("misaligned"):

| Metric | Aligned (good) | Misaligned (bad) |
|---|---:|---:|
| Global geometry `identity_deviation_p95_px` | ~1.8 px | ~2.2 px |
| Local ghosting `p95_shift_px` | **< 1 px** | **4+ px** |
| Local ghosting `worst_patch.shift_px` | ~5 px | **40+ px** |
| Glare `reduction_factor` | > 2x (both) | > 2x (both) |

Local ghosting — especially `p95_shift_px` and the single `worst_patch` — is
the metric that separates good from bad most reliably; global geometry
barely moves because a single ~8px patch-scale error is averaged away across
hundreds of SIFT matches spread over the whole image. As a rule of thumb,
combining the synthetic numbers above with the real-dump validation:

- **Good stitch**: local ghosting `p95_shift_px` comfortably under ~1.5px
  (both real dumps: 0.15px and 1.11px), no single non-healed patch above
  ~5px, `worst_patch_flicker.png` shows no visible double edge, glare
  healing `darkened_fraction` low when the reference itself has little
  glare, high (with a correspondingly healed composite) when it does.
- **Bad stitch**: local ghosting `p95_shift_px` several pixels or more,
  `worst_patch_flicker.png` visibly shows a doubled/offset edge, and usually
  (but not always — see the edge-doubling caveat above) a `canny_edge_ratio`
  or `gradient_magnitude_ratio` past 1.
- **A large `healed_patches` count is not itself bad** — it means the app
  successfully healed a lot of glare/haze, which is the whole point of the
  technique. It only matters insofar as it changes which patches are trusted
  for the ghosting read.
- **Bright detail `retention_ratio` well below 1.0 (e.g. `20260724-123532`'s
  0.644 / 0.525 excl-healed) is a real problem the other six axes can
  otherwise miss almost entirely** — it means fine bright detail (stars,
  speckle, small highlights) is being deleted, not just softened. A ratio
  near 1.0, or even somewhat above it (`20260724-115323`'s 1.14-1.16, where
  healing a broad haze restored contrast and revealed MORE specks than the
  hazy reference had), is fine.

These are informed starting points, not rigorously calibrated thresholds —
revisit them as more real device dumps accumulate (only three exist so far).

## Tests

`test_stitch_quality.py` builds everything in-process with numpy/cv2, no fixtures on
disk:

- The main 5-shot fixture (`SyntheticFixture`): a random richly-textured
  "puzzle" scene, a reference view + 4 corner views each a small random
  perspective warp of the scene with a glare disc at a *different* position
  per shot (mirroring the real 5-shot technique), a perfectly aligned
  composite built from the KNOWN inverse homographies, and a deliberately
  misaligned composite (one frame shifted ~8px). Asserts the misaligned
  composite scores higher on ghosting p95 and edge-doubling, the aligned
  composite scores near-clean, both reduce glare relative to the raw
  reference, missing-`metadata.json` loading works, `--quad` restricts
  stats to a region while leaving full-frame numbers unchanged, and
  `stitch.py`'s own from-scratch SIFT registration (independent of the
  fixture's known homographies) also reduces glare when run end to end.
  Also exercises both CLIs directly.
- Smaller, purpose-built fixtures for the glare-healing metric: one where
  the composite is IDENTICAL to a glared reference (nothing for
  darkest-pixel-wins to fix — darkening must read ~0), and one with a
  broad, non-saturating haze the composite fully heals (darkening must be
  clearly positive while near-saturation glare reduction reports exactly
  0% on both images — reproducing the real-dump finding above).
- A fixture for the healed-patch exclusion mechanism: two touched 64px
  patches, one with a real ~8px content shift (genuine ghosting, low
  darkening) and one where the composite content is swapped for an
  unrelated crop (a large, organic phase-correlation shift reading) AND the
  reference is washed toward white there (genuine high darkening) — the
  darkened patch's raw shift is asserted to exceed the genuinely-shifted
  patch's, so the test actually exercises exclusion changing the outcome,
  not just leaving an already-correct answer alone.
- A fixture for bright detail preservation: a small dark, mildly textured
  "sky" scene sprinkled with tiny bright dots (`_make_starfield_scene`).
  A composite that keeps the dots must retain a `retention_ratio` near 1.0; a
  composite with every dot painted over in the local background color
  (simulating darkest-pixel-wins erasure) must retain far fewer. Also
  asserts the fixture actually exercises the exclusion path (a raw darkening
  map over the erased composite, with no speck exclusion, DOES show positive
  `darkened_fraction`) and that the real `score_dump` pipeline's
  speck-excluding darkening map keeps `darkened_fraction` from increasing —
  the erased stars must not register as glare-healing benefit.
- Three tests for `--mode masked`, reusing the same `SyntheticFixture` (plus
  `_make_starfield_scene` for the third): `stitch_masked` run end to end
  still heals the reference's own centered glare disc (none of the 4 corner
  shots glares there); a patch far from every glare disc stays within a
  couple of gray levels of the reference (the mask must not touch it); and
  `compute_glare_alpha` leaves a modestly-dimmed (not erased) synthetic
  star's pixels untouched, retention ~= 1.0 — modest dimming is what a
  well-registered masked-mode composite actually produces at a star under
  residual resampling softening, unlike the full-erasure fixture the
  bright-detail-retention test above uses to force a strong signal for
  `score_stitch`'s own detector.
- A Round 5/6 fixture for the robust darkening estimate
  (`test_masked_mode_median_darkening_ignores_textured_background_misalignment`,
  name unchanged from Round 5, now exercising Round 6's vote):
  a high-frequency, high-contrast textured background (mirroring a real
  carpet) sampled by 4 frames each shifted ~1-2px from the reference (the
  scale of residual misalignment SIFT+ECC leaves behind, not raw uncorrected
  corner jitter) plus a moving glare disc that no covered frame shares with
  the reference. Asserts the genuine glare still triggers the mask
  (alpha > 0.5) while the textured background well away from every glare
  disc stays essentially unmasked (mean alpha < 0.05, alpha > 0.5 fraction
  < 5%) — then, as a sanity check that the fixture actually exercises the
  fix, confirms a plain MIN-composite-based darkening signal (Round 4's
  approach) over the SAME fixture WOULD have flagged much of that same
  background region (> 30% at alpha > 0.5). See
  [Round 6 in `HANDOFF.md`](HANDOFF.md#round-6-a-stricter-vote-for-the-robust-darkening-estimate-2026-07-24)
  for why the median itself was later replaced with a stricter vote.

```bash
cd network
uv run pytest scripts/stitch_quality/test_stitch_quality.py -v
```

`outputs/` (and any `--out` directory under a dump) is gitignored.
