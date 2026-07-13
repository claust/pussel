# North Star Test Set — Capture Guide

**Date:** July 2026
**Status:** PLAN (v1 not yet captured)

The critical review's conclusion (item #4 in `CRITICAL_REVIEW.md`, confirmed by
exp23): no result so far says anything about *photographed* pieces, and the
synthetic benchmark is largely solvable by pixel matching. This guide defines
how to build the real benchmark: a small, frozen test set of **iPhone photos of
physical puzzle pieces**, each labeled with its true position and rotation
against a reference image of the box.

Design goals, in order:

1. **Trustworthy labels** — after exp20, label correctness is the top priority.
   The protocol below makes labels a mechanical consequence of capture order,
   not of human judgment per piece.
2. **Cheap to capture** — one afternoon for v1 (~150 pieces), no rig, no app.
3. **Frozen and versioned in git** — small processed JPEGs committed to the
   repo, originals archived outside it.

---

## 1. What v1 looks like

| Item | Target |
| --- | --- |
| Puzzles | 5–8, varied artwork |
| Pieces per puzzle | every piece of the puzzle (so pick 16–48 piece puzzles) |
| Total pieces | 120–200 |
| Photos per piece | 1 (upright; rotations applied digitally at eval, same as exp20 protocol) |
| Reference images per puzzle | 2 (raw box photo + rectified crop of the artwork) |
| Backgrounds | ≥2 per puzzle (e.g. wood table, white paper, dark cloth) |
| Committed size budget | < 50 MB total (plain git, no LFS) |

Even 100 real pieces will tell us more than another 20K synthetic puzzles — do
not let scope creep delay v1. Anything not captured in v1 becomes v2 in a new
subfolder; v1 is never edited after it is committed (see §7).

### Choosing puzzles

- **Small piece counts are ideal** (24–100 pieces): fast to assemble and
  disassemble, and each piece is large enough that its grid cell is
  unambiguous. Kids' puzzles are perfect.
- **Vary the artwork**: at least one photographic landscape, one busy
  illustration/collage, and one *hard* low-texture puzzle (lots of sky or
  flat color). exp23 showed SIFT fails exactly on low-texture pieces — the
  north star must contain that failure mode.
- **The box art must actually match the puzzle.** Check that the lid picture
  is the full artwork without heavy overlaid logos, borders, or a cropped/
  stylized rendition. Skip puzzles where it isn't.
- **Every piece of every chosen puzzle is photographed** — no sampling. This
  keeps position labels mechanical (capture order = label, §3) and avoids
  biasing the set toward pieces that are easy to place by eye. Large puzzles
  (500+) are out of scope for v1 for exactly this reason; they can join a
  later version once a labeling story for them exists.

---

## 2. Ground truth scheme

Every jigsaw puzzle is a grid of `rows × cols` pieces. A piece's label is:

- `row`, `col` — integer grid position, `(0, 0)` = top-left when the puzzle is
  viewed upright (as on the box).
- `rotation` — degrees in {0, 90, 180, 270}, the rotation *applied to* the
  piece relative to upright. Photographed upright ⇒ `rotation = 0`.
- Per puzzle: `rows`, `cols`, and the reference image.

Everything else is derived at eval time: normalized piece center is
`((col + 0.5) / cols, (row + 0.5) / rows)`, and mapping to any coarser grid
(e.g. the current model's 4×4 head) is
`cell = floor(y_norm * 4) * 4 + floor(x_norm * 4)`.

**Rotation convention:** identical to exp20/exp23 — the label is the rotation
the eval harness applies digitally to the upright photo, so labels are exact by
construction (`(0 + applied) % 360`). This keeps v1 cheap and directly
comparable to the synthetic benchmark. A physically-rotated subset (place the
piece at a real 90°/180°/270° and record it) is a good v2 addition to check
that digital rotation isn't flattering, but is deliberately out of scope for
v1.

**Why labels can't be wrong here:** position comes from disassembling in
raster order (§3), not from anyone deciding per-photo where a piece belongs;
rotation is 0 by protocol. The only failure mode is losing your place in the
raster order — which the count check in §4 catches.

---

## 3. Capture protocol (iPhone)

### One-time setup (5 minutes)

1. **Marker for "up":** put a piece of tape or a drawn arrow on the shooting
   surface pointing *away from you*. That direction is "puzzle top". Every
   piece is placed with its puzzle-top edge facing the arrow.
2. **Camera settings:** turn **Live Photos off**; use the **1x main lens**
   (never ultra-wide — distortion; avoid macro-mode auto-switching by not
   going closer than ~15 cm). HEIC format is fine — conversion happens on the
   Mac. Clean the lens.
3. **Light:** daylight or a bright room light that is *not* directly behind
   you (your own shadow is the most common spoiler). Some shadow/glare
   variation across pieces is good — it's the realism we're testing — but the
   piece must be clearly visible.

### Per puzzle (~30–45 min for a 24–50 piece puzzle)

1. **Photograph the box lid** straight-on, artwork filling the frame. This is
   the reference "puzzle image". Take 2–3 shots, keep the best.
2. **Assemble the puzzle** (or start from an already-assembled one).
3. **Photograph the assembled puzzle** top-down once — this is the audit
   record that lets anyone re-verify labels later.
4. **Note `rows × cols`** (count the pieces along each edge).
5. **Disassemble in raster order, photographing as you go:** remove piece
   (0,0) (top-left), place it on the background *upright* (its top edge toward
   the arrow), photograph it top-down with the piece filling roughly ⅓–½ of
   the frame, set it aside. Then (0,1), (0,2), … row by row. **Capture order
   is the label** — if you misfire, delete the photo on the phone immediately
   so the sequence stays clean, or note the frame number on paper.
6. **Switch background** between puzzles (or halfway through a puzzle) so no
   background is confounded with one puzzle.
7. **AirDrop the batch to the Mac** right away, into a folder per puzzle
   (§4). Batching per puzzle keeps the raster-order → filename mapping
   trivially auditable.

Variation in distance, angle (±15° from top-down), and lighting across pieces
is *desirable* — don't build a copy stand. The one thing that must be
consistent is the upright orientation against the arrow.

---

## 4. Getting the images into the repo

### Transfer: AirDrop (recommended)

Select the batch in Photos → AirDrop to the Mac → files land as
`IMG_XXXX.HEIC` with sequential numbers and EXIF timestamps, both of which
preserve capture order. Drop each puzzle's batch into its own raw folder:

```
~/pussel_north_star_raw/          # NOT in the repo — this is the archive
  puzzle01_ravensburger_alps/
    box.HEIC                      # renamed by hand: the box lid shot
    assembled.HEIC                # renamed by hand: the audit shot
    IMG_4501.HEIC … IMG_4524.HEIC # pieces, raster order
  puzzle02_.../
```

Alternatives if AirDrop is fiddly: iCloud Photos on the Mac, or USB cable +
the built-in **Image Capture** app. Avoid WhatsApp/email/Messages — they
recompress.

**Archive the raw folder** (iCloud Drive or an external disk). Full-res HEICs
are the master copy; the repo gets processed derivatives only.

### Processing: downscale + convert

Committed images are **sRGB JPEG, max side 1024 px, quality ~85** — roughly
150–350 KB each, so 200 pieces + references ≈ 40–60 MB worst case, fine for
plain git. (Model inputs are 256×256/128×128 and exp23's classical methods
used ~110–130 px pieces, so 1024 px retains ample headroom for future
higher-res work.) macOS can do the conversion natively, including EXIF
auto-rotation:

```bash
sips --resampleHeightWidthMax 1024 --setProperty format jpeg \
     --setProperty formatOptions 85 IMG_4501.HEIC --out piece_r00_c00.jpg
```

An `ingest.py` script (to be written alongside the first real batch) wraps
this: it takes a raw puzzle folder plus `rows`/`cols`, sorts piece shots by
filename/EXIF time, **verifies the count matches `rows × cols`**, converts,
renames to `piece_rRR_cCC.jpg`, and appends to `metadata.csv`. The count
check is the safety net for the raster-order protocol.

The box reference additionally gets a **rectified variant**: perspective-crop
the raw box shot to exactly the artwork rectangle (Preview.app crop is fine
for v1; a 4-corner-click OpenCV `getPerspectiveTransform` helper can join
`ingest.py` later). Both raw and rectified are committed; evaluation uses the
rectified one.

### Repo location and git strategy

`network/datasets/*` is gitignored (line 181 of the root `.gitignore`), which
is correct for the 12K-puzzle synthetic sets — but the north star set is
small, precious, and *must* be versioned. Add an exception:

```gitignore
# .gitignore — after the existing network/datasets/* line
!network/datasets/north_star/
```

Layout:

```
network/datasets/north_star/
  README.md                       # provenance: puzzles used, capture dates, device
  v1/
    metadata.csv                  # one row per piece — the single source of truth
    puzzle01_alps/
      box_raw.jpg                 # box lid photo, downscaled
      box_rectified.jpg           # perspective-cropped artwork — the eval reference
      assembled.jpg               # audit shot
      pieces/
        piece_r00_c00.jpg
        piece_r00_c01.jpg
        ...
    puzzle02_.../
```

`metadata.csv` columns:

```csv
puzzle_id,piece_file,rows,cols,row,col,rotation,background,captured_date,device
puzzle01_alps,puzzle01_alps/pieces/piece_r00_c00.jpg,6,4,0,0,0,wood_table,2026-07-19,iphone15pro
```

Following exp22/exp23's lesson: ground truth lives in an explicit CSV, not
(only) encoded in filenames.

**No Git LFS.** At < 50 MB of derivatives, plain git is simpler and avoids the
LFS bandwidth quota being drained by CI clones. If a future version balloons
past ~150 MB, revisit.

---

## 5. Evaluation protocol

One eval script (a sibling of `exp20_realistic_pieces/reevaluate_checkpoint.py`
and `exp23_classical_baselines/evaluate.py`) consumes `metadata.csv`:

- **Samples:** each piece × 4 digital rotations (PIL `rotate(expand=False)`
  on the upright photo, exactly as the synthetic protocol) — v1 gives
  ~480–800 samples.
- **Reference input:** the puzzle's `box_rectified.jpg`, resized to whatever
  each method expects (256×256 for the CNN).
- **Metrics:** report at two granularities —
  1. **4×4 cell** (map the piece's normalized center through the 4×4 grid) —
     directly comparable to the exp20/exp23 numbers, and what the current
     16-way CNN head can express;
  2. **native piece grid** (`rows × cols`) for methods that output
     coordinates (NCC, SIFT, future heatmap models) — the honest product
     metric.
  Plus rotation accuracy and both-correct, as always.
- **First run:** the exp20 checkpoint, masked NCC, and the SIFT→NCC hybrid,
  side by side. This immediately answers the program's biggest open question:
  *does anything survive the camera?* Expect all three to drop hard — the
  interesting number is by how much, and which degrades most gracefully.

**Sanity checks before trusting any number** (the exp20 lesson):

- Perfect-model check: an "oracle" that reads labels from `metadata.csv` must
  score 100%.
- Human spot check: open 10 random `(piece photo, box_rectified, label)`
  triples and confirm by eye that the piece belongs at that cell, upright.
- Confusion matrix on rotation: a flat ~25% row again means a labeling bug,
  not a model property.

---

## 6. Background-only realism (parallel, not blocking)

The north star measures reality; it must never be trained on. The synthetic
training data gets its own realism push separately (item #5 of the review:
independent photometric jitter, perspective warps, real backgrounds, sensor
noise). Photos of your empty tabletops/backgrounds taken during the capture
session are cheap and useful as synthetic-data backgrounds — take a dozen.

---

## 7. Usage discipline

- **Frozen:** once `v1/` is committed, its images and `metadata.csv` never
  change. Extensions land as `v2/`; eval scripts pin a version explicitly.
- **Test-only, low-frequency:** this is the *north star*, not a dev set —
  evaluate on it at milestones (new architecture, new training data regime),
  not every epoch or hyperparameter tweak. The review's lesson #3 (the test
  set steered development from exp7 on) applies doubly here, because this set
  is tiny.
- **If a dev-real set is needed** for iterating on camera robustness,
  photograph 1–2 *additional* puzzles under the same protocol into
  `dev_real/` — never promote north-star puzzles into it.

---

## 8. Pitfalls checklist

- [ ] EXIF orientation: normalize during ingest (`sips`/Pillow handle it);
      verify a few portrait-shot photos display upright after processing.
- [ ] HEIC color: convert to sRGB JPEG so OpenCV/PIL loads are consistent.
- [ ] Glare on glossy pieces: reposition the light, don't use flash.
- [ ] Ultra-wide/macro lens distortion: stay on 1x, ≥15 cm away.
- [ ] Box art with logos/borders: rectify-crop to artwork only, or skip the
      puzzle.
- [ ] Losing raster order: delete misfires on the phone immediately; the
      ingest count check must pass before anything is committed.
- [ ] Screenshots/edited photos sneaking into the batch: AirDrop from a
      per-puzzle album, not from All Photos.
- [ ] Don't commit originals: raw HEICs stay in the archive folder outside
      the repo.

---

## 9. v1 milestone checklist

1. [ ] Pick 5–8 puzzles (≥1 low-texture); verify box art matches artwork.
2. [ ] Capture session: box + assembled + pieces per puzzle, ≥2 backgrounds,
       plus a dozen empty-background shots.
3. [ ] AirDrop batches to `~/pussel_north_star_raw/`, archive it.
4. [ ] Write `ingest.py`; process all batches; count checks pass.
5. [ ] Rectify box references.
6. [ ] Add `.gitignore` exception; commit `network/datasets/north_star/v1/`.
7. [ ] Write the eval script; oracle scores 100%; human spot check passes.
8. [ ] Run exp20 CNN + NCC + SIFT→NCC hybrid on v1; log results as a new
       experiment entry in `EXPERIMENT_LOG.md`.
