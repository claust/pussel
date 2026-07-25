# Puzzle Rectification Quality Benchmark

## Objective

When a puzzle is added, the app fuses a five-shot burst into one composite
(`scripts/stitch_quality/` measures *that* step) and the backend then detects
the puzzle's quadrilateral and perspective-warps it into the overview image
the solver matches pieces against. On real captures that overview still comes
out visibly skewed, with single-digit detection confidence.

This is the **offline measurement tool for the rectification step**: it runs
the shipped detection + warp over a capture dump and turns "it looks tilted"
into numbers. See `docs/puzzle-image-rectification.html` for the pipeline
being measured and where its three defects come from.

## Files

- `rectify.py` — the geometry: detector-with-path-attribution, true-aspect
  recovery from an image quad, plane tilt, residual skew, IoU, corner errors,
  and the warp (backend-style or at a specified aspect).
- `score_rectification.py` — the CLI: scores a dump, prints a table, writes
  `rectify_scores/rectify_metrics.json` plus annotated diagnostics.
- `label_quad.py` — click-four-corners ground-truth labeller, writes
  `truth.json` next to the images.
- `test_rectify_quality.py` — synthetic self-tests that validate the
  *metrics* against known answers (a rectangle of known aspect rendered
  through a known camera).

## Usage

All commands run from the repo root.

```bash
uv run --project backend python scripts/rectify_quality/score_rectification.py <dump_dir>
```

```bash
uv run --project backend python scripts/rectify_quality/score_rectification.py <dump_dir> --all-frames
```

The `--all-frames` form also scores `reference.jpg` and `corner_1..4.jpg`,
which is how you see whether an individual raw shot rectifies better than the
composite does.

Ground truth unlocks the strongest metrics (IoU, corner error, and a true
aspect ratio measured independently of the detector). Label once per dump:

```bash
uv run --project backend python scripts/rectify_quality/label_quad.py <dump_dir> --all-frames
```

Click the four corners of the **puzzle picture** — the artwork's rectangle,
not the cardboard's rounded outer edge — in order TL, TR, BR, BL. Keys: `u`
undo, `r` restart, `s` save and advance, `n` skip, `q` quit. Labels land in
`<dump_dir>/truth.json` and `score_rectification.py` picks them up
automatically.

Other flags: `--image <file>` scores one loose photo, `--focal-px` supplies a
known focal length instead of solving for one, `--out` redirects the output
directory, `--no-diagnostics` skips the annotated images.

Self-tests:

```bash
uv run --project backend pytest scripts/rectify_quality/test_rectify_quality.py
```

## What each metric means

### `path` — which generator's candidate won

`mask` (color residual), `grabcut` (fragmented-mask recovery), `edge` (Canny
contours), `lines` (intersected Hough lines), or `none` (full-frame
fallback). All four generators run on every photo and their candidates
compete on one score, so this reports the *winner*, read straight off
`detect_candidates`. The JSON additionally carries the winning candidate's
score `components` and the top `runners_up`, which say how close the call
was — a clear winner and a photo finish between two very different quads are
different risks.

### `conf` — the shipped confidence heuristic

Exactly what the app shows on the confirm screen. Below 0.4 the app warns
"Detection looks uncertain" (`low_confidence` in the JSON). It is the
winning candidate's boundary evidence — rectangularity, edge support and
border contrast — and deliberately excludes the coverage term that helps
rank candidates, since how much of the frame the puzzle fills says nothing
about whether the quad is right.

On the labelled captures it runs 0.18 (a correct detection with thin
evidence on busy carpet) to 0.92 (a clean box shot), with a photo of no
puzzle at all scoring 0.24. Treat it as separating "found a rectangle" from
"found nothing"; on this set it tracks corner accuracy only loosely.

### `IoU` and `corner_rms` — agreement with the hand-labelled quad

Needs `truth.json`. IoU is rasterized rather than analytic, so degenerate
self-intersecting quads can't break it. Corner errors compare TL-to-TL after
both quads are ordered, reported in pixels at a normalized 2048 px long side
so dumps at different resolutions are comparable.

### `true_ar` — the aspect ratio the puzzle actually has

The heart of the tool. A rectangle's four image corners constrain both the
camera's focal length and the rectangle's aspect ratio: opposite edge pairs
give two vanishing points whose directions must be orthogonal in space, which
is one equation in `f`; with `f` known the aspect follows. (Zhang & He's
whiteboard-rectification formulation.) `aspect.focal_source` says where the
focal length came from:

| value | meaning |
| --- | --- |
| `known` | supplied via `--focal-px` |
| `estimated` | solved for from the quad — the normal case |
| `parallel` | the quad is a parallelogram (square-on shot); `f` drops out and the aspect is exact anyway |
| `assumed` | **the quad couldn't solve for `f`**, so a 69.4° horizontal FOV was assumed |

Dumps written since the plane-rectification work carry the camera's real
intrinsics, and the scorer reads the focal length straight out of them for
`reference.jpg` and `corner_N.jpg` — those frames now report `known` without
`--focal-px`. The composite still can't: its pixels have been rectified into
a metric top-down space and no longer have a focal length. `--focal-px`
remains the escape hatch for older dumps and loose photos.

`assumed` is worth noticing: it happens when the phone is tilted about a
single axis, leaving the horizontal edges parallel and only one finite
vanishing point. The self-tests pin what it costs — an assumed focal 23% off
the true one moved the aspect by 7%. That is the concrete argument for
dumping the camera's real intrinsics with each capture.

`true_ar` is measured from the **hand-labelled** quad when one exists
(`aspect.source: "truth"`), isolating the warp's aspect error from the
detector's corner error. Without labels it falls back to the detected quad
(`"detected"`) and the number folds both errors together.

### `crop_ar` and `ar_err%` — what the warp actually produced

`crop_ar` is the shipped warp's output aspect, sized from the quad's own edge
lengths. `ar_err%` is how far that is from `true_ar`. This is defect 2 in the
design doc, quantified: under perspective the two horizontal edges are
foreshortened by different amounts, so max-edge-length sizing systematically
mis-shapes the crop. A test pins the effect — a square photographed at 35°
warps to a crop more than 10% off square.

### `tilt°` — how oblique the shot was

The angle between the camera's optical axis and the puzzle plane's normal,
from decomposing the plane-to-image homography. 0° is perfectly square-on.
Context for the other numbers: a 12% aspect error at 26° tilt and the same
error at 3° tilt are different bugs.

### `skew°` — residual skew, no ground truth needed

How far the straight lines *inside* the warped crop still sit off the
horizontal/vertical axes, length-weighted. This is the number that
corresponds to what the user sees: when the detector picks the wrong quad,
the puzzle's real borders end up inside the crop at an angle, and this is
that angle. On a correct crop the box's edges coincide with the crop's own
borders and it reads ~0°.

Two exclusions matter. Segments hugging the crop boundary are dropped — a
warp fills its output edge to edge, so Canny fires along the border itself and
those segments are axis-aligned by construction; counting them would drag
every measurement to 0 and make a bad crop look perfect. Segments more than
20° off both axes are dropped as artwork content.

`—` means no qualifying straight structure was found, which is honest rather
than a failure: an assembled jigsaw of a cartoon scene, cropped tightly, has
no long straight lines in it. The metric carries real signal on box shots and
on any crop that swallowed part of the background.

### `ideal` (JSON only, needs ground truth)

The crop the same pixels would have produced from the correct quad at the
correct aspect — the ceiling this capture allows, and the way to tell a
detector fix from a warp fix. Written as `<name>_ideal.jpg`.

## Diagnostics written

Into `<dump_dir>/rectify_scores/`:

- `rectify_metrics.json` — every number above, per image.
- `<name>_quads.jpg` — the photo with the detected quad (orange) and the
  hand-labelled quad (green) drawn on it, corners lettered TL/TR/BR/BL.
- `<name>_crop.jpg` — what the app would show on the confirm screen.
- `<name>_ideal.jpg` — the ground-truth crop, when labelled.

## Notes

- Images are never committed. Dumps live in
  `~/Pictures/puzzles/glare_stitch_dumps/GlareFreeDumps/`; pull them off a
  device with the recipe in `scripts/stitch_quality/README.md`.
- The tool imports the real `app.services.puzzle_detector`, so it measures
  the shipped behavior and tracks changes to it automatically.
- The scorer prints one line before the table saying whether the dump carries
  camera geometry. Read it first: without it the composite was **not**
  plane-rectified on device (a pre-2026-07-25 dump, the Simulator path, or a
  burst that never found the surface), and its skew is the capture's rather
  than the detector's.
