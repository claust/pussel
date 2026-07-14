# North Star v1 — real-photo test set

The first real (non-synthetic) benchmark for the puzzle solver, per
[`NORTH_STAR_GUIDE.md`](../NORTH_STAR_GUIDE.md). iPhone photos of physical
puzzle pieces, labeled by capture protocol, evaluated against a photo of each
puzzle's box artwork.

## Capture session (2026-07-13)

14 kids' puzzles, 236 pieces, 958 photos (14 overviews + 236 × 4 piece shots).
Protocol per puzzle: photograph the overview (box poster or assembled puzzle),
then remove pieces in raster order (top-left → right, row by row), placing each
piece upright on four backgrounds in a fixed cycle — **red carpet → gray fabric
→ cardboard → wood** — with a yellow/red paper arrow in frame pointing to
"puzzle top". Capture order *is* the label: no per-photo human labeling.

| # | slug | overview | grid |
|---|------|----------|------|
| 01 | frozen_scene | IMG_1093 | 3×4 |
| 02 | frozen_closeup | IMG_1142 | 3×4 |
| 03 | dumbo | IMG_1191 | 3×4 |
| 04 | bambi | IMG_1240 | 4×4 |
| 05 | lion_king | IMG_1305 | 5×5 |
| 06 | jungle_book | IMG_1406 | 5×4 |
| 07 | peppa_kitchen | IMG_1487 | 2×3 |
| 08 | peppa_forest | IMG_1512 | 3×4 |
| 09 | peppa_aquarium | IMG_1561 | 3×3 |
| 10 | peppa_family | IMG_1598 | 4×4 |
| 11 | paw_patrol_a | IMG_1663 (overview: IMG_2052¹) | 4×6 |
| 12 | paw_patrol_b | IMG_1760 | 4×6 |
| 13 | unicorn_pink | IMG_1857 | 4×6 |
| 14 | unicorn_night | IMG_1954 | 4×6 |

¹ The shot taken during the session (IMG_1663) turned out to be puzzle 12's
poster — the box shipped two puzzles. The correct artwork was photographed
afterwards as IMG_2052 (`OVERVIEW_OVERRIDES` in `ingest.py`); IMG_1663 still
marks where puzzle 11's piece shots start.

Raw HEICs (the master copy) live outside the repo in `~/Pictures/puzzles/`,
archived separately. **Neither raw photos nor processed images are committed**
(`network/datasets/*` is gitignored); the dataset is reproduced locally with
`ingest.py`.

## Ingest

```bash
cd network
uv run python experiments/north_star/ingest.py   # ~10 min, macOS only (uses sips)
```

Produces `network/datasets/north_star/v1/`:

- `puzzleNN_<slug>/overview.jpg` — the reference artwork photo (1024 px sRGB JPEG)
- `puzzleNN_<slug>/pieces/piece_rRR_cCC_<background>.jpg` — upright piece photos
- `metadata.csv` — one row per piece photo: grid position, background, source
  image, piece bounding box, orientation provenance, review flag
- `review/puzzleNN_<slug>.jpg` — contact sheets (one row per piece, its 4
  upright crops) for human verification

Key mechanics (details in the `ingest.py` docstring):

- **Count check**: hard-fails unless each puzzle has exactly `rows × cols × 4`
  shots after its overview — the safety net for capture-order labels.
- **Upright rotation**: EXIF orientation is unreliable for top-down photos, so
  each shot is rotated by the in-frame arrow (yellow body → red tip direction),
  cross-checked by correlating the piece crop against its three siblings over
  all four rotations. Conflicts prefer the arrow and set `flagged=1` (the
  matcher drifts on low-texture crops).
- **Arrow exclusion**: the arrow is a capture aid, not part of the sample.
  `metadata.csv` carries the piece bbox (`piece_x1..y2`); evaluation must crop
  to it so no method ever sees the arrow.
- `rotation_overrides.csv` / `bbox_overrides.csv` (next to the script) pin the
  rotation (CCW 90° steps) or piece bbox (final-image pixel coords) for shots
  where detection fails; add rows after checking the contact sheets — manual
  entries beat both automatic signals.

## Usage discipline

Test-only, low-frequency (guide §7): evaluate at milestones, never train on it,
never promote these puzzles into a dev set. Extensions go to `v2/`, `v1` is
frozen once adopted.
