# Experiment 31: Break Piece↔Overview Pixel Identity Structurally

Test 2 of `docs/synthetic-dataset-realism.html` §7.

## Objective

Every CNN trained on synthetic puzzle data collapses on real photos
(12.7–14.8% both-correct against the classical SIFT→NCC bar of 76.7%). The
July 2026 realism investigation traced this to **shortcut learning**, not a
rendering-fidelity deficit, and named three cues. exp30 removed two of them —
both outright label-leaking generator bugs — and **provably** removed them
(the rotation-prediction bias is gone, the acceptance probes pass). Real
transfer still did not move: **13.2% both, versus exp26's 12.7%**. Synthetic
went *up*, to 78.5%.

That leaves the third cue, §4.3: **piece↔overview pixel identity**. The
synthetic "overview" is the byte-identical source JPEG the pieces were cut
from — shared demosaic, white balance, JPEG block grid, noise fingerprint,
sharpening. Measured masked NCC at the ground-truth overview location:

| Domain | GT median NCC | Decoy (wrong cell) |
| --- | --- | --- |
| Synthetic raw (exp20/exp30) | **0.990** | 0.471 |
| Synthetic + full exp26 augmentation | **0.937** | 0.528 |
| Real (north_star) | **0.730** | 0.407 |

exp26's stated purpose was to break this and it did not: photometric jitter is
a near-affine intensity map any contrast-normalised matcher cancels, and
geometric jitter is absorbed by a scale/rotation search.

exp31 models the production truth instead: **the piece and the overview are
two independent captures of two different physical prints, and never share a
pipeline instance.**

**Prediction:** the §8 NCC-headroom probe passes (median GT NCC falls from
0.99 toward the real 0.73 while the decoy stays near 0.41), and north_star
both-correct then moves off the 12.7–13.2% floor. If the probe passes and
transfer still does not move, pixel identity is *not* the remaining blocker
and §7's ranking needs rethinking again — Test 3 (corpus swap) and Test 4
(FDA) become the live hypotheses.

**Status: the gate passes, but marginally.** Masked NCC at the ground-truth
cell is down from exp30's 0.937 to **0.763** (real 0.679) with the
true-vs-decoy margin preserved at **0.320**. The training half of the
prediction is untested — that is the next run.

> **Measurement provenance — read before quoting any number here.** All gate
> figures in this README are `--sample 800 --seed 0` on the default config,
> which is now the probe's default. Do **not** quote them to three decimals as
> if exact: exp31 sits close enough to the thresholds that the *verdict* was
> seed-dependent at the old `--sample 200` default (seed 1 FAILED at
> 0.780 / 0.482 against limits of 0.779 / 0.460, while seeds 0 and 2 passed),
> and n=200 also read optimistically low (~0.740 vs the ~0.765 population
> value). At n=800 every seed tried agrees, but the pass is *narrow*: 0.763
> against a 0.779 ceiling, and 0.429 above-0.8 against a 0.460 ceiling. Treat
> "exp31 clears the gate" as "clears it by a hair on this corpus", not as
> comfortable headroom. Ablation rows below were measured at the old n=200 and
> are directional — their *ordering* is the useful part, not their absolute
> values.

## What is built

Everything is layered on **exp30's validated geometry** (lossless
`Image.transpose` rotation + exp24's 8%-margin square framing) and imported
from it. exp20, exp24, exp26, exp30 and `shared/puzzle_shapes` are **not
modified** — frozen experiments stay frozen. exp31 also reuses exp30's stored
pieces byte-for-byte (`datasets/realistic_4x4_rgba_v2`), because everything
exp31 adds happens at **load** time.

The six components the experiment was scoped around are below, each drawn
independently **per sample and per view**. That last part is the rule from the
face-anti-spoofing literature (§7): an identical simulated artifact in both
views just becomes the next shortcut. Two effects were added during the work
because the gate showed the original six were not enough — a two-octave
substrate/illumination field inside each chain, and the same room light falling
on the piece's close-up — and both are called out where they live (components 1
and 5).

### 1. Independent per-view degradation chains

`capture.apply_view_chain`, run once per view with that view's own
`ViewChainConfig`. Physical stage order: subpixel phase → optical blur →
sample down → sensor noise → sample up → substrate/illumination field → ISP
sharpen → tone curve → JPEG.

| Stage | Piece chain | Overview chain | Why |
| --- | --- | --- | --- |
| Subpixel phase | ±0.5 px | ±0.7 px | The two views do not share a pixel-lattice phase. |
| Optical blur | σ 0.2–0.6, p=0.35 | σ 0.3–1.1, p=0.6 | A wide box shot is softer than a macro shot. |
| Resample scale | 0.90–1.00 | **0.62–0.88** | Disjoint by construction, so the piece is always the finer-resolved view. Magnitudes are set by the real path, not invented: a real piece reaches 128 px by a 3× downsample from 382 px native (crisp, full MTF to Nyquist), and a real overview reaches 64 px/cell by a 1.6–2.3× downsample (also crisp), so the only honest loss to model on the overview is what the capture itself lost to lens, hand-shake and capture-side JPEG. See finding 2 below for why the first, far more aggressive range was both unfaithful *and* counterproductive. |
| Resample kernels | independent draw from {bilinear, bicubic, lanczos, box, hamming} down, {bilinear, bicubic, lanczos} up | same, independent draw | Mixing *kernels*, not just scales, is what stops the two views sharing an MTF. |
| Sensor noise | σ 2.0–7.0 (0–255) | σ 1.5–6.0 | Applied at the *sampled* resolution so the upsample correlates the grain. §5 measured the overview at 0.07 vs a real 0.84 — `augment_puzzle()` left it noise-free. |
| Substrate / illumination field | two octaves: coarse amp 0.06–0.18 at 12–30% of the view, fine amp 0.03–0.10 at 3–9% | same, independent draw | **One of the four load-bearing components** (+0.063 GT median when removed). A masked *zero-mean* NCC cancels global affine intensity maps, and a single smooth full-view ramp is nearly affine in space; two multiplicative octaves at a fraction of the view are neither. Physically: two different substrates (Ravensburger laminates linen-structured paper, the box is coated offset stock) plus uneven room light. |
| ISP sharpen | UnsharpMask r 0.6–1.6, 40–140%, p=0.6 | p=0.3, ≤90% | The iPhone ISP oversharpens; a close-up shows it far more. |
| Tone curve | per-channel γ 0.85–1.20 ±0.06, gain 0.92–1.08, lift ≤0.03 | same, independent draw | **Non-linear on purpose.** Masked NCC cancels affine intensity maps — which is exactly why exp26's ColorJitter did nothing to the correlation — but not a per-channel gamma. Also models two print runs' dot gain and gamut. |
| JPEG | q 55–95, p=0.8 | q 45–88, p=0.9 | **Independent 8×8 block grid**: the image is edge-padded by a random (ox, oy) ∈ [0,7]² before encoding and cropped back, moving the DCT lattice to (−ox, −oy). exp26 encoded both views with the lattice anchored at pixel (0,0), so the block artifacts lined up. |

Never a shared pass, never a shared draw. The parameters are drawn up front
into a `ChainDraw` struct rather than pulled inline, for a reason worth
recording: the obvious way to build the shared-pass control — save the RNG
state, run the chain twice, restore in between — **silently does not work**,
because the two views have different pixel dimensions, so the noise array and
the substrate octaves consume a size-dependent number of `np.random` values and
every draw after the first desynchronises. The first implementation had exactly
that bug, and the "shared" control measured *lower* NCC than the independent
one. With the parameters drawn once, `independent_chains=False` gives both
views provably identical camera settings (a unit test counts the draws), while
the pixel-level noise and field realisations necessarily differ — they are
realisations over differently shaped views. That is the strongest available
notion of "one shared pipeline instance", and the README's ablation table
reports it as such.

### 2. Asymmetric Random Patching (ARP)

`capture.apply_arp`. The stereo literature's anti-pixel-identity augmentation
(Chuah et al., [2106.08486](https://arxiv.org/abs/2106.08486): synthetic→real
stereo error **28.0% → 4.0%** from two data-side augmentations). With p=0.5,
2–4 patches are pasted into **exactly one** view — a unit test asserts one and
only one view is ever modified. It is the **largest single contributor** to the
§8 gate here too: removing it costs 0.082 of GT median and fails the gate.

- **Patch size is a fraction of the view's short side, not a fixed pixel
  range.** The doc's "50–100 px" is calibrated to KITTI stereo images, where
  it is 13–27% of the 375 px short side; on our 128 px piece a literal 50–100
  px patch would cover the whole view. exp31 uses **10–25%** of the view's
  short side (≈13–32 px on the piece, ≈26–64 px on the overview), which
  preserves the stereo work's relative occlusion.
- Aspect is jittered 0.6–1.6× so patches are not all square.
- **Patch content never comes from the image being patched.** 60% is a crop of
  another *training* puzzle JPEG, 25% a two-colour gradient, 15% uniform
  noise. Self-patching an overview would duplicate one cell's content
  somewhere else in the same overview and weaken the very label the model is
  learning; and the source list is training-only, so no val/test box art and
  no north_star imagery can leak in through a patch.
- 35% of patches are blended with a feathered alpha instead of pasted hard,
  because a single fixed blend mode is itself a learnable artifact
  (Cut-Paste-Learn, [1708.01642](https://arxiv.org/abs/1708.01642)).

### 3. Segmentation-artifact alignment — fitted to real rembg

`capture.apply_segmentation_slop`. The real pipeline segments pieces with
rembg/u2net, so real masks have boundary softness, tab-neck rounding and a
couple of pixels of slop; synthetic masks are exact Bézier lattice cuts.
Running true rembg on ~192k pieces every epoch is far too slow (**0.9–1.9 s per
piece** measured on this Mac → **47–99 h** for a single 192k pre-pass, and a
pre-pass would bake *one frozen mask per piece*, which both breaks the
per-sample randomisation rule and hands the model a memorisable per-piece
fingerprint). So exp31 models it explicitly: low-frequency boundary noise →
Gaussian blur → **soft** re-threshold (a hard step would recreate the
infinitely sharp boundary being removed) → a small randomised dilate/erode
biased outward.

The ranges are **fitted, not guessed** — `segmentation_validation.py` measures
the cheap model against real u2net on the same pieces. The first ranges tried
were 3.4× too soft. See "Validation" below.

RGB is never touched, only alpha, so nothing can leak through this stage.

**Supporting stage: the scene surface** (`capture.apply_scene_surface`). This
turned out to be load-bearing and is worth calling out. `cut_piece` writes
**black** RGB wherever alpha is zero (`image_masking.py`: an all-zero RGBA
canvas plus a masked paste), and transparent padding is black too — so every
mask *dilation* downstream revealed pure black, an artifact of a fill value
rather than of any camera. Measured effect: a heavy dark tail on the boundary
rim ratio (25th percentile 0.57) dragging the median to 0.83. exp31 therefore
pads the piece and fills the out-of-silhouette RGB with a plausible **surface**
(the table the piece is photographed on) before anything dilates into it, which
is also what "slight background bleed at the rim" physically *is*. Note the
generator does not leave the neighbouring piece's content there, so no cell
label can leak through this stage.

### 4. Bright die-cut cardboard rim + cast shadow

`capture.apply_die_cut_edge` and `capture.composite_with_shadow`. §5 measured
the boundary rim ratio at **1.08 real vs 0.98 synthetic**, and flagged that
**exp26's halo augmentation pushes the wrong way (0.69, darkening)**. exp31
therefore sets exp26's `halo = False` by default and adds a *brightening* rim
instead: a distance-transform band 3–7% of the piece's short side, blended
toward a light warm cardboard colour (luma 195–250) at strength 0.35–0.75.
Fitted on 160 pieces to a view-level rim ratio of **1.087** against the real
1.08.

> The band is wider than a literal 3 mm core would be. That is deliberate:
> what §5's 3-px boundary band actually measures is the core *plus* the domed
> face and the specular edge, and a sub-pixel band does not survive the
> downstream resampling at all.

The cast shadow is a directional band derived from the *final* silhouette
(after slop, framing and geometry — the silhouette the light actually meets),
with alpha capped at 0.45 so it stays below `ALPHA_THRESHOLD` and can never be
counted as subject by the largest-component bbox or by the mask probes. On a
black background it is invisible, which is exactly right: in deployment rembg
removes the surface the shadow fell on.

Together with the rim this partly closes §7's flagged gap (a) — the piece is a
die-cut 3D object with an exposed light cardboard core, not a flat page.

### 5. Box-photo overview realism, and the same light on the piece

`capture.apply_box_photo`. §2: in production the user photographs the **box** —
a printed motif on glossy cardboard — so the overview's target is
"print-and-photograph of box art". Mild residual perspective (the iOS pipeline
rectifies, imperfectly), 1–3 specular glare blobs, a directional lighting
gradient, vignetting. Pure numpy/PIL/torchvision — **no Augraphy dependency**
(CPU-only, document-oriented, and a new dep needs a root `uv lock`).

`apply_piece_lighting` gives the piece view its own independently drawn
gradient, vignette and specular highlight. The piece was the only one of the
two views with no spatially varying illumination at all, which is both
physically wrong — a phone macro shot of a laminated, slightly domed piece has
a falloff and usually a highlight — and, along with the substrate field, one of
the two effects that actually move the §8 metric.

> **On the residual perspective and labels.** Warping the overview does shift
> a cell's centre away from its nominal (cx, cy). torchvision displaces each
> corner by up to `distortion/2` of the side, so the default 0.05 moves a cell
> centre by ≤2.5% of the image = **≤10% of a cell**. That is bounded label
> noise which is *present in the real domain* (north_star's labels are ideal
> grid cells over imperfectly rectified photos), not a label bug — and it is
> behind its own `--no-box-photo` flag. The image is edge-padded before the
> warp and centre-cropped after, so no empty wedges appear in the view.

### 6. Crop / bbox jitter

`capture.frame_rgba_jittered`, §7's flagged gap (b): synthetic pieces are cut
on an exact lattice, real ones are segmented with pixel slop, and classical
print-scan analysis found mild cropping is the distortion that hits every
frequency band. Each of the four bbox sides is jittered independently and the
8% margin becomes 5–12%.

The jitter is deliberately **asymmetric — up to 5% outward but only 2%
inward** — against a margin of at least 5%, so the silhouette still cannot
reach the input border and exp30's §8 border-touch probe keeps reading ~0. It
is also drawn independently of the rotation label, so it cannot become a
second §4.1-style leak. (rembg errs outward in practice too: measured
`area_ratio` 1.011.) A test asserts border non-contact over 15 draws × 4
rotations.

## The 256 px corpus limitation — read this before judging the asymmetry

The source corpus (`network/datasets/puzzles`, 11,998 files) is **256×256 px**,
so a 4×4 cell is **64 px** natively. Real data (§5) has pieces at **382 px**
native and overviews at ~100–150 px/cell.

**The doc's "render the piece from a ≥350 px high-res source" is therefore
impossible here**, and exp31 does not pretend otherwise — upsampling 64 px
invents no detail. A higher-resolution corpus is a separate prerequisite
(Test 3, e.g. the Ravensburger motif CDN at 1024 px).

What exp31 *does* do is establish the **asymmetry direction** and break
**matched MTF**: the overview is degraded on its own coarser chain, with its own
sampling grid, resample kernel and phase, so the two views share neither a
modulation transfer function nor a pixel lattice even though absolute resolution
stays low. Be warned by finding 2 in the gate section, though: on this corpus
that buys **fidelity, not §8 headroom** — lowpassing the overview slightly
*raises* masked NCC, because subtracting detail cannot manufacture the
decorrelated high-frequency band a genuinely higher-resolution piece would
have. The direction is the real one — at the model input, real
pieces are downsampled ~3× from native (crisp) while real overviews are
downsampled ~2× (softer), so the piece is the finer-resolved view. exp30's
baseline had it backwards: the piece was *upsampled* 91→128 (soft) while the
overview was passed through 256→256 (identity).

**The knob.** `ViewChainConfig.target_px` is an absolute sampling budget in
pixels (the longer side), overriding `scale_min`/`scale_max`. With a ≥1024 px
corpus, the same code produces the doc's full intended asymmetry:

```bash
# 1024 px sources, 4x4 grid: 125 px/cell overview = 500 px, piece keeps native
uv run python -m experiments.exp31_pixel_identity.train \
    --overview-target-px 500     # piece-target-px stays unset
```

Nothing else has to change: the piece then genuinely carries ≥350 px of detail
the overview does not have, instead of merely a different MTF.

## Config knobs and ablations

`CaptureConfig` extends exp26's `AugmentConfig`, so every exp26 field and
range is inherited unchanged — with exactly three flipped off, each superseded
for a **measured** reason:

| exp26 flag | exp31 default | Superseded by |
| --- | --- | --- |
| `halo` | `False` | `rim` + `seg_slop`. exp26's alpha erosion drove the rim ratio to 0.69 where real is 1.08. |
| `noise` | `False` | The per-view chains, which apply noise at the *sampled* resolution. |
| `jpeg` | `False` | The per-view chains, which use an *independent block grid*. |

`CaptureConfig.ablation_flags()` reports the **effective** state of every
component (ANDed with `enabled` and `capture`), so a preset that switches
`capture` off cannot read as "ARP: on" in `results.json`.

Presets (`--capture-preset`), each isolating one lever:

| Preset | What it does |
| --- | --- |
| `full` | Everything on (the run). |
| `exp30` | `capture=False` and exp26's halo/noise/jpeg back on — reproduces exp30 through exp31's code, the control. With `capture` off the piece path hands its composite to exp26's *own* `_augment_appearance`, so the control really does get exp26's noise, blur and JPEG; an earlier version only applied the colour jitter and was a measurably weaker baseline (0.957 vs the correct 0.937). |
| `no_chains` | **One** chain over **both** views from **one** RNG state — same phase, scale, kernels, noise, tone curve and JPEG grid. The honest shared-pass reproduction; the ablation that has to fail. |
| `no_asymmetry` | The overview keeps its own chain but borrows the piece's sampling budget: matched MTF, independent draws. |
| `no_substrate` | Both chains drop the substrate/illumination field. |
| `no_arp` / `no_seg_slop` / `no_rim` / `no_box_photo` / `no_piece_lighting` / `no_crop_jitter` | Drop exactly one component. |
| `chains_only` / `arp_only` | Whole-family controls. |

Every preset also has a CLI switch (`--no-arp`, `--no-rim`, …) so a preset can
be narrowed further without editing code, plus `--overview-target-px` /
`--piece-target-px` for the absolute sampling budgets.

`ncc_probe.py` has no ablation flag of its own, so `augmentation.py` reads the
preset from an environment variable — that is how every row of the ablation
table below was produced:

```bash
EXP31_CAPTURE_PRESET=no_arp uv run python -m experiments.exp31_pixel_identity.ncc_probe \
    --pipeline exp31 --sample 200 --seed 0
```

## Validation: cheap segmentation slop vs real rembg

`segmentation_validation.py` runs three mask sources over the same synthetic
pieces — the exact lattice cut, exp31's cheap model, and real `rembg`/u2net
(the deployed path from `exp24_piece_classifier/build_positives.py` and
`exp25_north_star_eval/evaluate.py`) — and compares boundary statistics.

```bash
cd network
uv run python -m experiments.exp31_pixel_identity.segmentation_validation --sample 240
```

Result on 240 pieces (medians; `soft_px` = boundary transition width in px,
`alpha_grad` = mean |∇α| over the transition band, `area_ratio` = mask area ÷
exact mask area, `rim_ratio` = boundary luminance ÷ interior luminance):

| Source | soft_px | alpha_grad | area_ratio | rim_ratio |
| --- | --- | --- | --- | --- |
| `exact` (generator lattice cut) | 0.78 | 0.543 | 1.002 | 0.928 |
| `cheap` (exp31, fitted) | **1.25** | **0.413** | **1.010** | 0.938 |
| `rembg` (real u2net) | **1.15** | **0.493** | **1.008** | 0.930 |

The fitted cheap model lands within **9%** of rembg on transition width, **16%**
on boundary gradient and **0.2%** on mask area — and both sit clearly apart from
the exact lattice cut, which is crisper (0.78) and tighter (1.002) than either.
The direction of rembg's error is reproduced too: it errs *outward*, keeping
about 1% extra area at the rim, which is why the slop model's dilate bias is
0.7 rather than 0.5.

The first ranges tried (blur 0.6–2.2, softness 0.8–3.0 px) gave `soft_px` 3.18
and `alpha_grad` 0.219 — **3.4× too soft** — which is why these ranges are
fitted rather than assumed.

**Decision: the cheap model, not a rembg pre-pass.** rembg costs
0.9–1.9 s/piece on this machine (→ 47–99 h for a single 192k pre-pass; maybe
2–5 h on a GPU). More decisively, a pre-pass would freeze **one mask per
piece**, which both violates the per-sample randomisation rule §7 takes from the
face-anti-spoofing literature and hands the model a memorisable per-piece
fingerprint on the training set. The cheap model costs well under a millisecond,
is re-randomised every epoch, and is measurably in the right place.

## Measured effect on the pixel-identity shortcut — the §8 gate

Run through the sibling's `ncc_probe.py`, which implements the doc's own
protocol (9 scales × 7 rotations, mask = strict eroded interior), at
`--sample 200 --seed 0`. **The gate PASSES.**

| Domain | GT median NCC | frac > 0.8 | decoy | GT−decoy margin |
| --- | --- | --- | --- | --- |
| exp20 raw | 1.000 | 0.990 | 0.435 | 0.565 |
| exp26 | 0.933 | 0.830 | 0.432 | 0.501 |
| exp30 | 0.937 | 0.840 | 0.453 | 0.484 |
| **exp31** (n=800, seed 0) | **0.763** | **0.429** | 0.443 | 0.320 |
| real (north_star) | 0.679 | 0.340 | 0.303 | 0.377 |
| gate band | [0.579, 0.779] | [0.220, 0.460] | — | [0.277, 0.527] |

exp31's row clears every band, but by 0.016 on GT median and 0.031 on the
above-0.8 fraction. The other rows are n=200 measurements, kept because the
comparison across pipelines is what matters and the gaps are large; exp31's is
re-measured at n=800 because it is the only row close enough to a threshold for
sampling noise to change its verdict.

The `exp30` **control run through exp31's own code** lands at 0.931 / 0.830 /
0.454 / 0.477 — within noise of the probe's independent exp30 reference
(0.937 / 0.840), which is what validates that the exp31 numbers are measuring
the data change and not a plumbing difference.

### Per-component ablation, all through the gate

`EXP31_CAPTURE_PRESET=<preset>` (see "Config knobs"), same sample and seed:

| Preset | GT median | frac > 0.8 | margin | Verdict | Δ GT vs full |
| --- | --- | --- | --- | --- | --- |
| `exp30` (control) | 0.931 | 0.830 | 0.477 | FAIL | +0.194 |
| `chains_only` | 0.895 | 0.720 | 0.414 | FAIL | +0.158 |
| `no_arp` | 0.819 | 0.538 | 0.359 | **FAIL** | +0.082 |
| `arp_only` | 0.813 | 0.520 | 0.379 | FAIL | +0.076 |
| `no_rim` | 0.801 | 0.505 | 0.335 | **FAIL** | +0.064 |
| `no_substrate` | 0.800 | 0.497 | 0.368 | **FAIL** | +0.063 |
| `no_seg_slop` | 0.781 | 0.460 | 0.326 | **FAIL** | +0.044 |
| `no_piece_lighting` | 0.769 | 0.447 | 0.330 | PASS | +0.032 |
| `no_crop_jitter` | 0.763 | 0.406 | 0.303 | PASS | +0.026 |
| **`full`** (n=200; 0.763 at n=800) | **0.737** | **0.381** | **0.300** | **PASS** | — |
| `no_box_photo` | 0.735 | 0.365 | 0.292 | PASS | −0.002 |
| `no_asymmetry` | 0.723 | 0.371 | 0.288 | PASS | −0.014 |
| `no_chains` (shared pass) | 0.717 | 0.365 | 0.298 | PASS | −0.020 |

Four components are **individually load-bearing** — drop any one and the gate
fails: ARP (+0.082), the bright rim (+0.064), the substrate/illumination field
(+0.063) and the segmentation slop (+0.044). Every one of them works the same
way: it puts structure into one view that provably is not in the other, or
modulates the two views by fields that cannot be cancelled. That matches the
stereo literature, where ARP carried most of the 28% → 4% improvement.

### Three findings that were not expected, and one is a warning

**1. The degradation chains alone do almost nothing** (`chains_only` 0.895 vs
the `exp30` control 0.931 — 0.036 out of the 0.194 total). Resample kernels,
phases, sensor noise and independent JPEG grids are all present in
`chains_only`, and they move the metric barely at all.

**2. Lowpassing the overview *raises* masked NCC.** `no_asymmetry` and
`no_chains` both score *lower* (better) than `full`. Masked NCC is dominated by
low frequencies, so removing the overview's high-frequency band suppresses
exactly the uncorrelated component and leaves the strongly-correlated
remainder. The first draft of this experiment used an aggressive overview range
(0.45–0.75, i.e. 29–48 px/cell) and scored 0.764; tightening it to the
physically-faithful 0.62–0.88 improved the gate to 0.737.

This is the 256 px corpus limitation showing up as a *measurement*, and it is
the sharpest available argument for prioritising the higher-resolution corpus
(Test 3). Real resolution asymmetry decorrelates because the piece gains
genuine detail the overview never had; simulated asymmetry by *removing* detail
from the overview correlates, because subtraction cannot create the
decorrelated band. **Resolution asymmetry is kept on by default anyway** — it
is the real MTF relationship and the brief for this experiment — but it earns
its place on domain fidelity, not on this metric, and `no_asymmetry` is the
first ablation to try if a trained exp31 underperforms.

**3. Warning — the margin is now the binding constraint, and 0.737 is close to
the floor.** exp31's GT−decoy margin is 0.300 against a lower bound of 0.277.
The decoy median is stuck at ~0.44 across *every* pipeline (exp20 raw 0.435,
exp26 0.432, exp30 0.453, exp31 0.436) because it is a property of the
**corpus**, not of the augmentation: at 64 px/cell, Unsplash landscape photos
have genuinely self-similar cells, so a wrong cell still correlates at 0.44.
Real content reaches 0.303. Since no per-view degradation lowers the decoy,
pushing GT median from 0.737 down to real's 0.679 would drive the margin to
≈0.24 and **fail the gate on the margin criterion instead**.

Arithmetic: with the decoy pinned at 0.436 and the margin floor at 0.277, the
lowest admissible GT median on this corpus is **0.713**. exp31 is at 0.737,
i.e. within 0.024 of that floor. The remaining distance to real's 0.679 is not
reachable by making the synthetic pair more different — it needs the *content*
to be less self-similar (Test 3's corpus swap) so the decoy floor drops too.
Treat 0.737 as this corpus's practical optimum rather than as headroom left on
the table.

### Domain-gap side metrics

Measured with a local estimator over 160 pieces (medians; this estimator is
*not* the doc's, so compare columns rather than absolute values against §5):

| Metric | exp30 | exp31 | Real (§5) |
| --- | --- | --- | --- |
| Piece flat-region σ | 2.06 | 3.98 | 0.77 |
| Overview flat-region σ | 0.87 | 2.00 | 0.84 |
| Boundary rim ratio | 0.901 | **1.087** | **1.08** |

The rim ratio is the one calibrated directly against §5's number, and it lands
on it (1.087 vs 1.08) from the wrong side of both prior pipelines (raw
synthetic 0.98, exp26 0.69). The overview noise goes from ~nothing to
comparable with the piece, closing the direction of §5's 0.07-vs-0.84 gap.

## How to run

exp31 **reuses exp30's stored pieces byte-for-byte** — the on-disk format is
unchanged, because everything exp31 adds happens at load time. The dataset
root is therefore exp30's, `datasets/realistic_4x4_rgba_v2`.

> **The full dataset does not exist locally.** The 11,998-puzzle v2 root was
> generated on a RunPod container disk that has since been shut down. What a
> checkout has is a **60-puzzle smoke sample** under
> `datasets/realistic_4x4_rgba_v2` (enough for the tests, the probes and a
> 1-epoch smoke train, and every number in this README was measured on it),
> plus the older 1,200-puzzle exp26/v1 root `datasets/realistic_4x4_rgba`
> (**not** interchangeable — its base rotations are the lossy ones exp30
> fixed). Source JPEGs live only in the main checkout at
> `network/datasets/puzzles`, so a worktree run needs
> `--puzzle-root /path/to/main/network/datasets/puzzles`. Full generation
> happens on the pod, inside `runpod/setup_and_train.sh`.

**1. Generate the dataset** (exp30's generator, unchanged — on the pod this is
done for you by `setup_and_train.sh`):

```bash
cd network
uv run python -m experiments.exp30_generator_fixes.generate_dataset \
    --source-dir datasets/puzzles \
    --output-dir datasets/realistic_4x4_rgba_v2 \
    --n-puzzles 100000 --workers 8 --skip-existing
```

**2. Run the acceptance gate BEFORE retraining.** §8 requires the NCC-headroom
probe to pass — median masked NCC at the ground-truth location must fall from
0.99 toward the real 0.73 while the wrong-cell decoy stays near 0.407:

```bash
cd network
uv run python -m experiments.exp31_pixel_identity.ncc_probe
uv run python -m experiments.exp31_pixel_identity.segmentation_validation --sample 240
# and exp30's probes, which must keep passing (border touch, classical parity):
uv run python -m experiments.exp30_generator_fixes.probes --pipeline exp30
```

**3. Train:**

```bash
cd network
uv run python -m experiments.exp31_pixel_identity.train --epochs 50 --eval-test
```

**RunPod (the real run)** — generation and 50-epoch training happen on the
pod; generation is CPU-parallel and resumable:

```bash
cd network/experiments/exp31_pixel_identity
./runpod/prepare_package.sh          # code + frozen split + source puzzles
# scp runpod_package_exp31/runpod_training.tar.gz to the pod, then:
#   cd /workspace && tar -xzf runpod_training.tar.gz && ./setup_and_train.sh
# CAPTURE_PRESET=no_arp EPOCHS=50 ./setup_and_train.sh   # for an ablation
```

Keep the RunPod gotchas in mind, all inherited from exp30 and preserved in
`setup_and_train.sh`: `pip install --break-system-packages` (Ubuntu 24.04 is
PEP 668 externally-managed and the image's torch lives in the system Python),
never reinstall torch/torchvision, generate to the **container disk** via
`RGBA_DIR=/root/...` (a MooseFS network volume charges ~4× quota for 192k
small PNGs), `--skip-existing` to resume, and `opencv-python-headless` for
cv2.

**4. North-star evaluation (ONCE, after training):**

```bash
cd network
uv run python experiments/exp25_north_star_eval/evaluate.py \
    --dataset-root datasets/north_star/v1 \
    --checkpoint experiments/exp31_pixel_identity/outputs/pixel_identity/checkpoint_best_state_dict.pt
```

`train.py` exports `checkpoint_best_state_dict.pt` as a raw state_dict, in the
same format and under the same name as exp26/exp30, so exp25's exact call
(`load_state_dict(torch.load(..., weights_only=True))`) runs unchanged.

**Tests:**

```bash
cd network
uv run pytest experiments/exp31_pixel_identity/test_exp31.py -q
```

## Training recipe — deliberately identical to exp26/exp30

`FastBackboneModel` / ShuffleNetV2_x0.5, AdamW (backbone 1e-4, head 1e-3,
wd 0.01), 128 px piece / 256 px overview, batch 64 (128 on the pod), 50
epochs, the frozen exp20 split `splits/realistic_4x4_v1.json`, and the exp20
harness. Val selects the checkpoint; the synthetic test set is touched **once**
with `--eval-test`; north_star is evaluated exactly once at the very end. Any
difference in the results is therefore attributable to the two-capture data
model and nothing else.

**val/test stay clean and unaugmented**, exactly as in exp26 and exp30. The
pixel-identity shortcut is a property of the *training* pair, and keeping the
evaluation protocol byte-compatible is the only way the synthetic accuracy
numbers stay comparable across exp20/exp26/exp30/exp31. Known limitation,
inherited: checkpoint selection therefore happens on a distribution where the
shortcut still works, and there is no *real* validation set. For a reproducible
"new synthetic test set" (classical parity, proxy 𝒜-distance), use
`capture_dataset.view_pair(..., seed=deterministic_seed(path, rot))`.

## Success criteria

- **Gate, before any retraining:** the §8 NCC-headroom probe. Median masked NCC
  at the ground-truth overview location must drop **from 0.99 toward the real
  0.73**, while the decoy (wrong-cell) NCC stays near the real **0.407** — the
  *contrast* between true and decoy must not be destroyed, only the
  pixel-identity headroom. A run that destroys the contrast has made the task
  impossible, not realistic. **Status: PASS, narrowly** — 0.763 GT median /
  0.429 above 0.8 / 0.320 margin at n=800 seed 0, against ceilings of 0.779 and
  0.460 (see the gate section, and the provenance note at the top: the verdict
  was seed-dependent at the old n=200 default). Note the margin is the binding
  constraint and the corpus's decoy floor puts the lowest admissible GT median
  at ≈0.720, so do not read the remaining distance to real's 0.679 as headroom.
- **Classical parity (§8):** if the SIFT→NCC hybrid's accuracy on the new
  synthetic test set goes **up** (it is 82.2% synthetic / 76.7% real today),
  the data got *easier* rather than more real, and the change is wrong however
  good the NCC number looks. A large drop is also a warning — that means
  artificially harder, not more real.
- **Border-touch probe must keep reading ~0** (exp30's §4.1 fix must survive
  the crop jitter). Asserted by a unit test and re-checkable with exp30's
  `probes.py`.
- **Primary:** north_star both-correct moves off the 12.7–13.2% floor.
- **Secondary:** synthetic test accuracy is *expected to fall* from exp30's
  78.5%. That is the point — part of 78.5% was the pixel-identity shortcut
  being read off. A synthetic score that stays at 78.5% means the shortcut is
  still there.
- Beating the SIFT→NCC hybrid's 76.7% is **not** expected from this alone. §5's
  content gap (colorfulness 69.9 real vs 20.1 synthetic; 23% of synthetic
  pieces nearly featureless) and the resolution gap both need the corpus swap.

## Files

- `capture.py` — the whole two-capture model: `ViewChainConfig` /
  `CaptureConfig`, `ChainDraw` + `draw_chain` / `apply_chain_draw` /
  `apply_view_chain`, `apply_arp` + `PatchSource`, `apply_scene_surface`,
  `apply_segmentation_slop`, `apply_die_cut_edge` + `composite_with_shadow`,
  `apply_box_photo` + `apply_piece_lighting`, `frame_rgba_jittered`, and
  `augment_view_pair` — the one function that guarantees the two chains run
  separately and that ARP touches exactly one view.
- `capture_dataset.py` — `CapturePieceDataset` (train) over exp30's eval
  datasets and the frozen split; plus **`view_pair()`**, the probe contract:
  both views at native *and* model-input scale, the piece mask at both scales,
  and the un-degraded baseline views, from a reproducible
  `deterministic_seed()`.
- `train.py` — training entry point (exp26 recipe, exp20 harness, exp25's
  checkpoint format).
- `augmentation.py` — a thin adapter exposing `capture.py` under the names
  `ncc_probe.py` discovers (`augment_piece_rgba`, `augment_overview`,
  `Exp31Config`), plus the `EXP31_CAPTURE_PRESET` hook. It adds no behaviour;
  read its docstring for the two things to know when reading probe output
  through it (ARP has to be per-view there, and why `AugmentConfig` is
  deliberately *not* exported).
- `segmentation_validation.py` — the cheap-slop-vs-real-rembg comparison above.
- `ncc_probe.py` — the §8 NCC-headroom acceptance gate (written alongside this
  experiment; not modified here).
- `test_exp31.py` — 32 unit tests: no shared chain draws, the substrate field is
  a spatially varying field and not a global gain, ARP hits exactly one view and
  scales with the view, slop moves alpha only, the scene surface replaces the
  generator's black fill, the rim brightens, border non-contact under crop
  jitter over 15 draws × 4 rotations, the shared-pass control draws its
  parameters exactly once (counted), every preset builds and runs, and the
  `view_pair` probe contract.
- `runpod/` — `prepare_package.sh` + `setup_and_train.sh`.

## Results

_Not yet run._
