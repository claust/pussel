# Exp24: Binary piece / not-a-piece classifier

## Motivation

The live piece-preview endpoint (`POST /api/v1/piece/preview`) segments the
most salient region of a camera frame with rembg and reported a **heuristic**
confidence (area/aspect/skin-tone bands). In practice the heuristic passed
false positives — a face in profile on a webcam showed "Piece detected" — and
the skin gate cannot generalize to the mobile use case, where false positives
are table objects (cups, keys, phones).

This experiment trains a small binary classifier ("puzzle piece" vs "not a
piece") that runs on the rembg-segmented crop. Its probability **replaces**
the heuristic confidence in the API response; the cheap area/aspect gates stay
as pre-filters, and the skin gate remains only as the fallback when no
checkpoint is available (e.g. CI).

## Model

- MobileNetV3-Small backbone + small binary head — **1,000,993 parameters**
- 128×128 input, few-ms single-image CPU inference
- Mobile-friendly from day one: the classifier will later run on-device

## Input protocol

Shared between training and the backend
(`backend/app/services/piece_classifier.py` — keep in sync with
`data_prep.py`):

1. rembg RGBA output → largest opaque connected component
2. composite on black, crop to the component bbox + 8% margin
3. pad to square (black), resize to 128×128, scale to [0, 1] (no ImageNet
   normalization, matching the repo's other models)

## Data

All crops live in `network/datasets/piece_classifier/` (gitignored; build in
the main checkout so worktrees can share it by absolute path).

**Positives**
- `positives/synthetic/` — exp20 realistic-piece generator (Bezier tab/blank
  edges) over 400 source puzzles, seed 2024 → ~6.3k crops
- `positives/real/` — the 944 north-star piece photos (4 background types),
  segmented with rembg exactly like the preview pipeline. The photos
  themselves are never committed.

**Negatives** (things rembg actually segments, built through the same
downscale-to-320 + rembg pipeline as the backend)
- `negatives/caltech101/` — `Faces`/`Faces_easy` (webcam case) plus household
  and hand-held object categories: camera, cellphone, chair, cup, headphone,
  lamp, laptop, pizza, scissors, soccer_ball, stapler, umbrella, watch, wrench
  (~80 per category)
- `negatives/coco128/` — 128 real COCO scenes (people, cups, phones on tables)

Known gap: no dedicated bare-hands negatives (faces/persons cover skin tones).
Real-world false positives can be harvested via the backend's dev-only
`SAVE_PREVIEW_CROPS=true` setting, which saves accepted preview crops to
`uploads/preview_crops/` for use as hard negatives in retraining.

## Splits and augmentation

- Group-level splits (70/15/15) to prevent near-duplicate leakage: positives
  group by puzzle, negatives by chunks of 10 consecutive files per category
  (Caltech-101 orders same-subject shots consecutively); stratified by source.
- Train augmentation: full-circle rotation, H/V flips, color jitter
  (lighting), random downscale-upscale (simulates the 320px preview pipeline's
  low-resolution crops).

## Usage

```bash
# from network/ — build data (downloads Caltech-101 ~137 MB + COCO128 ~7 MB);
# paths are relative to network/ (pass an absolute --output-root from a worktree)
uv run python -m experiments.exp24_piece_classifier.build_positives \
    --synthetic-root datasets/piece_classifier/synthetic_raw \
    --real-root ~/Pictures/puzzles \
    --output-root datasets/piece_classifier
uv run python -m experiments.exp24_piece_classifier.build_negatives \
    --output-root datasets/piece_classifier

# train (checkpoint selection on val balanced accuracy; test evaluated once);
# --data-root defaults to network/datasets/piece_classifier
uv run python -m experiments.exp24_piece_classifier.train --epochs 12
```

Outputs land in `outputs/` (`checkpoint_best.pt`, `results.json`). The backend
loads `outputs/checkpoint_best.pt` automatically (override with
`PIECE_CLASSIFIER_CHECKPOINT_PATH`) and falls back to the heuristic when it is
missing.

## Results

See `outputs/results.json` and the EXPERIMENT_LOG.md entry (test metrics,
confusion matrix, per-category breakdown including faces and table objects).
