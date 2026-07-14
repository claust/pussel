"""Ingest the north-star capture session into a labeled dataset.

Turns the raw iPhone photos (see ``NORTH_STAR_GUIDE.md`` §3-4) into
``network/datasets/north_star/v1/``: downscaled sRGB JPEGs plus a
``metadata.csv`` with one row per piece photo.

Capture protocol this script assumes (July 2026 session, 14 puzzles):

* One overview photo per puzzle (box poster or assembled puzzle), then every
  piece in raster order (top-left, row by row), each photographed on four
  backgrounds in a fixed cycle: red carpet, gray fabric, cardboard, wood.
* A yellow/red paper arrow lies in every piece frame pointing to "puzzle top".
  EXIF orientation is unreliable for top-down shots, so each image is rotated
  in 90-degree steps until the arrow points up; pieces are then upright and
  ``rotation = 0`` by construction.

Orientation is decided by two independent signals and cross-checked:

1. **Arrow reading** — the arrow is segmented as the yellow+red foreground
   object; direction = red-tip centroid vs yellow-body centroid.
2. **Sibling matching** — the same piece appears in all 4 shots of its group,
   so each shot's piece crop (masked gradient magnitude) is correlated against
   its siblings' upright crops over all four rotations.

When both agree the shot is confident; on conflict the matcher wins (measured
more reliable) and the shot is flagged; arrow-less shots (mostly wood, where
the yellow paper matches the wood tones) use the matcher alone. Flagged rows
land in ``metadata.csv`` (``flagged=1``) and per-puzzle contact sheets are
written to ``<output>/review/`` for visual verification.

Labels are a mechanical consequence of capture order (group index -> row,
col); the script hard-fails when a puzzle's shot count does not match
``rows * cols * 4``.

Usage (from ``network/``)::

    uv run python experiments/north_star/ingest.py
        [--source ~/Pictures/puzzles] [--output datasets/north_star/v1] [--puzzle SLUG]
"""

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# (anchor/overview image number, rows, cols, slug) — capture-order manifest.
PUZZLES: list[tuple[int, int, int, str]] = [
    (1093, 3, 4, "frozen_scene"),
    (1142, 3, 4, "frozen_closeup"),
    (1191, 3, 4, "dumbo"),
    (1240, 4, 4, "bambi"),
    (1305, 5, 5, "lion_king"),
    (1406, 5, 4, "jungle_book"),
    (1487, 2, 3, "peppa_kitchen"),
    (1512, 3, 4, "peppa_forest"),
    (1561, 3, 3, "peppa_aquarium"),
    (1598, 4, 4, "peppa_family"),
    (1663, 4, 6, "paw_patrol_a"),
    (1760, 4, 6, "paw_patrol_b"),
    (1857, 4, 6, "unicorn_pink"),
    (1954, 4, 6, "unicorn_night"),
]

# slug -> image number whose photo is the overview, when it isn't the segment anchor.
# paw_patrol_a's anchor shot (IMG_1663) turned out to be puzzle_b's poster; the correct
# artwork was photographed later as IMG_2052. Override numbers never count as piece shots.
OVERVIEW_OVERRIDES: dict[str, int] = {"paw_patrol_a": 2052}

BACKGROUNDS = ["red_carpet", "gray_fabric", "cardboard", "wood"]
SHOTS_PER_PIECE = len(BACKGROUNDS)

DETECT_WIDTH = 480  # analysis resolution for masks (matches the resolution the detector was validated on)
CROP_SIZE = 96  # sibling-matching crop resolution
MATCH_MARGIN = 0.05  # matcher must beat the runner-up rotation by this much

# k = number of 90-degree counter-clockwise rotations that turn the arrow upright
K_FOR_DIRECTION = {"up": 0, "right": 1, "down": 2, "left": 3}
CV2_ROTATE = {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}
PIL_TRANSPOSE = {1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}


@dataclass
class ShotDetection:
    """Per-shot analysis extracted in the parallel phase."""

    tmp_jpg: str  # converted, unrotated 1024px JPEG
    captured_at: str
    device: str
    width: int
    height: int
    bbox: tuple[int, int, int, int] | None  # piece bbox in full-resolution unrotated coords
    arrow_k: int | None  # arrow-implied CCW rotations to upright, if arrow found
    crop: np.ndarray | None  # masked gradient crop (CROP_SIZE^2), unrotated
    crop_mask: np.ndarray | None


def convert_heic(src: Path, dst: Path, max_side: int, quality: int, match_srgb: bool = True) -> None:
    """Convert a HEIC to an EXIF-upright JPEG using macOS ``sips``.

    ``match_srgb`` converts colours to the sRGB profile — used for the images
    that land in the dataset. The detection pass runs on a plain conversion
    instead, because all thresholds were tuned and validated on those colours.
    """
    cmd = ["sips", "--resampleHeightWidthMax", str(max_side)]
    cmd += ["--setProperty", "format", "jpeg", "--setProperty", "formatOptions", str(quality)]
    if match_srgb:
        cmd += ["--matchTo", "/System/Library/ColorSync/Profiles/sRGB Profile.icc"]
    proc = subprocess.run(cmd + [str(src), "--out", str(dst)], capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"sips failed for {src.name}: {proc.stderr.strip()}")


def components(mask: np.ndarray, min_area: int) -> list[tuple[int, np.ndarray]]:
    """Connected components of ``mask`` at least ``min_area`` px, largest first."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = [
        (int(stats[i, cv2.CC_STAT_AREA]), labels == i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    out.sort(key=lambda t: -t[0])
    return out


def solid_yellow_bodies(hsv: np.ndarray, frame_area: int) -> np.ndarray:
    """Mask of blobs shaped like the arrow's paper body: solid, elongated yellow rectangles.

    Detected bottom-up from colour alone (no red tip required) so the arrow can
    be erased from the object mask even when its tip reading fails — otherwise
    a merged arrow ends up inside the piece's bounding box.
    """
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = ((hue > 10) & (hue < 45) & (sat > 120) & (val > 100)).astype(np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = np.zeros(yellow.shape, bool)
    for area, comp in components(yellow, frame_area // 4000):
        if area > frame_area // 15:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.findNonZero(comp.astype(np.uint8)))
        long_len, short_len = max(rw, rh), min(rw, rh)
        if short_len < 3 or not 1.5 < long_len / short_len < 8:
            continue
        if area / (rw * rh) < 0.7:
            continue
        out |= comp
    return out


def grow_piece(piece: np.ndarray, weak: np.ndarray, forbid: np.ndarray, iters: int = 5) -> np.ndarray:
    """Hysteresis growth: extend the piece into adjacent weakly-non-background pixels.

    Tabs on low-contrast pieces often fall just below the mask threshold and get
    clipped from the bounding box. Bounded dilation (≈5 px per iteration) into the
    weak mask recovers them without reaching detached background junk.
    """
    grown = piece.astype(np.uint8)
    allowed = (weak & ~forbid).astype(np.uint8)
    for _ in range(iters):
        grown = cv2.dilate(grown, np.ones((5, 5), np.uint8)) & allowed | grown
    return grown.astype(bool)


def arrow_score(
    comp: np.ndarray,
    area: int,
    yellowish: np.ndarray,
    reddish: np.ndarray,
    frame_area: int,
    require_red: bool = True,
) -> float | None:
    """Score a component as the paper arrow, or None if it can't be it.

    Colour alone is not enough — a mostly-yellow puzzle piece with red artwork
    (e.g. the Peppa Pig kitchen) mimics the arrow's palette. The arrow is also
    small, solid (rectangle + triangle fills its minimum-area rectangle) and
    elongated, while pieces are large, knobby and near-square.
    """
    if area > frame_area // 6:
        return None
    yfrac = float(yellowish[comp].mean())
    if yfrac < 0.2:
        return None
    if require_red:
        # the red tip may sit inside this comp or as a neighbouring blob split off by a mask gap
        near = cv2.dilate(comp.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        if float(reddish[comp].mean()) < 0.02 and int((reddish & near).sum()) < 30:
            return None
    # shape-test the yellow paper body itself (a solid elongated rectangle): attached
    # shadows don't distort it, and irregular artwork yellows fail its fill check
    body = cv2.morphologyEx((comp & yellowish).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bodies = components(body, frame_area // 4000)
    if not bodies:
        return None
    body_area, body_comp = bodies[0]
    if body_area > frame_area // 15:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.findNonZero(body_comp.astype(np.uint8)))
    long_len, short_len = max(rw, rh), min(rw, rh)
    if short_len < 3 or not 1.5 < long_len / short_len < 8:
        return None
    if body_area / (rw * rh) < 0.7:
        return None
    return float(body_area)


def find_arrow_component(
    comps: list[tuple[int, np.ndarray]], yellowish: np.ndarray, reddish: np.ndarray, frame_area: int
) -> np.ndarray | None:
    """Pick the arrow among the foreground components, splitting merged blobs if needed."""

    def with_tip(comp: np.ndarray) -> np.ndarray:
        """Merge a red tip that a mask gap split off from the body."""
        near = cv2.dilate(comp.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        return comp | (reddish & near)

    best: tuple[float, np.ndarray] | None = None
    for area, comp in comps:
        score = arrow_score(comp, area, yellowish, reddish, frame_area)
        if score is not None and (best is None or score > best[0]):
            best = (score, comp)
    if best is not None:
        return with_tip(best[1])
    # the arrow may have merged with the piece: erode the biggest blobs to split bridges
    for _area, comp in comps[:2]:
        eroded = cv2.erode(comp.astype(np.uint8), np.ones((7, 7), np.uint8))
        for _sub_area, sub in components(eroded, frame_area // 4000):
            grown = cv2.dilate(sub.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & comp
            if arrow_score(grown, int(grown.sum()), yellowish, reddish, frame_area) is not None:
                return with_tip(grown)
    return None


def arrow_direction(arrow: np.ndarray, yellowish: np.ndarray, reddish: np.ndarray) -> str | None:
    """Direction the arrow points: from the yellow body's centroid toward the red tip's."""
    ys, xs = np.nonzero(arrow & yellowish)
    body = np.array([xs.mean(), ys.mean()])
    ys, xs = np.nonzero(arrow & reddish)
    if len(xs) == 0:
        return None
    tip = np.array([xs.mean(), ys.mean()])
    dx, dy = tip - body
    return ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")


def detect_objects(
    bgr: np.ndarray,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None, np.ndarray | None, str | None]:
    """Segment the frame into piece and arrow.

    The background is modelled as the two dominant border colours (k-means) so
    that patterned backgrounds (carpet, wood grain) don't flood the mask. The
    arrow is the foreground component that is mostly yellow with some red; if
    it merged with the piece, the blob is eroded to split the bridge.

    Returns (piece bbox, piece mask, arrow mask, arrow direction).
    """
    height, width = bgr.shape[:2]
    frame_area = height * width
    # distance to the background in LAB with down-weighted lightness: dark-but-differently-hued
    # pieces (navy on red carpet) stay separable, while shadows (pure L shifts) stay quiet
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= 0.5
    border = np.concatenate(
        [lab[:15].reshape(-1, 3), lab[-15:].reshape(-1, 3), lab[:, :15].reshape(-1, 3), lab[:, -15:].reshape(-1, 3)]
    )
    _, _, centers = cv2.kmeans(border, 2, None, (cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    dist = np.stack([np.linalg.norm(lab - c, axis=2) for c in centers]).min(axis=0)
    mask = cv2.morphologyEx((dist > 18).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellowish = (hue > 10) & (hue < 45) & (sat > 80) & (val > 90)
    reddish = ((hue < 12) | (hue > 168)) & (sat > 110) & (val > 60)

    comps = components(mask, frame_area // 3000)
    if not comps:
        return None, None, None, None

    arrow = find_arrow_component(comps, yellowish, reddish, frame_area)
    direction: str | None = None
    if arrow is not None:
        direction = arrow_direction(arrow, yellowish, reddish)
        if direction is None:
            arrow = None

    # keep-out zone for the piece: the identified arrow, plus anything shaped like
    # the arrow's paper body — so a merged or unrecognised arrow can never leak
    # into the piece's bounding box
    forbid = cv2.dilate(solid_yellow_bodies(hsv, frame_area).astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    if arrow is not None:
        forbid |= cv2.dilate(arrow.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)

    bbox, piece = pick_piece(comps, forbid, yellowish, reddish, frame_area)
    # pale pieces can vanish at the default threshold (e.g. tan artwork on wood):
    # retry more sensitively with the keep-out area removed before giving up
    if bbox is None or (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < 0.025 * frame_area:
        mask2 = cv2.morphologyEx((dist > 12).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask2[forbid] = 0
        bbox2, piece2 = pick_piece(components(mask2, frame_area // 3000), forbid, yellowish, reddish, frame_area)
        if bbox2 is not None:
            bbox, piece = bbox2, piece2

    if piece is not None:
        # recover clipped tabs: grow into weakly-non-background neighbours, then pad
        piece = grow_piece(piece, dist > 10, forbid)
        ys, xs = np.nonzero(piece)
        pad_x = round(0.03 * (xs.max() - xs.min()))
        pad_y = round(0.03 * (ys.max() - ys.min()))
        bbox = (
            max(0, int(xs.min()) - pad_x),
            max(0, int(ys.min()) - pad_y),
            min(width - 1, int(xs.max()) + pad_x),
            min(height - 1, int(ys.max()) + pad_y),
        )
    return bbox, piece, arrow, direction


def pick_piece(
    comps: list[tuple[int, np.ndarray]],
    arrow_zone: np.ndarray,
    yellowish: np.ndarray,
    reddish: np.ndarray,
    frame_area: int,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray | None]:
    """Piece = largest plausible non-arrow component, plus its nearby fragments."""
    rest = []
    for _area, comp in comps:
        c = comp & ~arrow_zone
        area = int(c.sum())
        if area <= frame_area // 3000 or area > 0.4 * frame_area:
            continue
        # a blob shaped like the arrow body (solid yellow rectangle) is never the piece,
        # even when it wasn't confidently identified as the arrow (missing red tip)
        if arrow_score(c, area, yellowish, reddish, frame_area, require_red=False) is not None:
            continue
        rest.append((area, c))
    if not rest:
        return None, None
    rest.sort(key=lambda t: -t[0])
    _, piece = rest[0]
    ys, xs = np.nonzero(piece)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    for _area, c in rest[1:]:
        ys, xs = np.nonzero(c)
        if xs.min() < x2 + 12 and xs.max() > x1 - 12 and ys.min() < y2 + 12 and ys.max() > y1 - 12:
            piece = piece | c
    ys, xs = np.nonzero(piece)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), piece


def gradient_crop(
    bgr: np.ndarray, bbox: tuple[int, int, int, int], piece_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Masked gradient-magnitude crop of the piece, padded square — background-invariant signature."""
    x1, y1, x2, y2 = bbox
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    mag[~piece_mask] = 0
    crop = mag[y1 : y2 + 1, x1 : x2 + 1]
    mask = piece_mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.float32)
    ch, cw = crop.shape
    side = max(ch, cw)
    pc = np.zeros((side, side), np.float32)
    pm = np.zeros((side, side), np.float32)
    pc[(side - ch) // 2 : (side - ch) // 2 + ch, (side - cw) // 2 : (side - cw) // 2 + cw] = crop
    pm[(side - ch) // 2 : (side - ch) // 2 + ch, (side - cw) // 2 : (side - cw) // 2 + cw] = mask
    return cv2.resize(pc, (CROP_SIZE, CROP_SIZE)), cv2.resize(pm, (CROP_SIZE, CROP_SIZE))


def rot90(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate an array by ``k`` 90-degree CCW steps."""
    return img if k % 4 == 0 else cv2.rotate(img, CV2_ROTATE[k % 4])


def masked_ncc(a: np.ndarray, ma: np.ndarray, b: np.ndarray, mb: np.ndarray) -> float:
    """Normalised cross-correlation over the intersection of two masks (-1 when too small)."""
    joint = (ma > 0.5) & (mb > 0.5)
    if joint.sum() < 200:
        return -1.0
    x, y = a[joint], b[joint]
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def analyze_shot(args: tuple[Path, Path, int, int]) -> ShotDetection:
    """Parallel phase: convert one HEIC and run detection on it (no rotation yet)."""
    src, tmp_jpg, max_side, quality = args
    convert_heic(src, tmp_jpg, max_side, quality)
    exif = Image.open(tmp_jpg).getexif()
    img = Image.open(tmp_jpg)
    # detection runs on a plain (non-colour-matched) conversion at DETECT_WIDTH —
    # the exact data the detector's thresholds were tuned and validated on
    detect_jpg = Path(tmp_jpg).with_suffix(".detect.jpg")
    convert_heic(src, detect_jpg, DETECT_WIDTH, 85, match_srgb=False)
    small = cv2.cvtColor(np.asarray(Image.open(detect_jpg).convert("RGB")), cv2.COLOR_RGB2BGR)
    detect_jpg.unlink()
    bbox, piece_mask, _arrow_mask, direction = detect_objects(small)
    if bbox is not None:  # a plausible piece fills at least ~2.5% of the frame
        x1, y1, x2, y2 = bbox
        if (x2 - x1) * (y2 - y1) < 0.025 * small.shape[0] * small.shape[1]:
            bbox = None
    crop = crop_mask = None
    if bbox is not None and piece_mask is not None:
        crop, crop_mask = gradient_crop(small, bbox, piece_mask)
        scale = img.width / small.shape[1]  # scale bbox to full-resolution (unrotated) coords
        bbox = tuple(round(v * scale) for v in bbox)
    return ShotDetection(
        tmp_jpg=str(tmp_jpg),
        captured_at=str(exif.get(306, "")),
        device=str(exif.get(272, "")).strip(),
        width=img.width,
        height=img.height,
        bbox=bbox,
        arrow_k=K_FOR_DIRECTION[direction] if direction else None,
        crop=crop,
        crop_mask=crop_mask,
    )


def decide_rotations(group: list[ShotDetection]) -> tuple[list[int], list[str], list[bool]]:
    """Choose the upright rotation for each shot in a 4-shot piece group.

    Arrow reading and sibling matching are combined: agreement wins, the
    matcher overrides a disagreeing arrow (and flags the shot), arrow-less
    shots use the matcher alone.
    """
    refs = [(i, d) for i, d in enumerate(group) if d.arrow_k is not None and d.crop is not None]
    ks: list[int] = []
    sources: list[str] = []
    flags: list[bool] = []
    for i, det in enumerate(group):
        match_k = None
        margin = 0.0
        others = [(j, d) for j, d in refs if j != i]
        if det.crop is not None and others:
            scores = []
            for k in range(4):
                rc, rm = rot90(det.crop, k), rot90(det.crop_mask, k)
                vals = [masked_ncc(rc, rm, rot90(d.crop, d.arrow_k), rot90(d.crop_mask, d.arrow_k)) for _j, d in others]
                vals = [x for x in vals if x > -1]
                scores.append(float(np.mean(vals)) if vals else -1.0)
            order = np.argsort(scores)
            match_k = int(order[-1])
            margin = scores[order[-1]] - scores[order[-2]]

        if det.arrow_k is not None and (match_k is None or match_k == det.arrow_k):
            ks.append(det.arrow_k)
            sources.append("arrow+match" if match_k is not None else "arrow")
            flags.append(False)
        elif det.arrow_k is not None:
            # conflict: the shape-tested arrow reading beats the matcher (the matcher
            # drifts on low-texture crops); keep the arrow but flag for the sheets
            ks.append(det.arrow_k)
            sources.append("arrow_over_match")
            flags.append(True)
        elif match_k is not None:
            ks.append(match_k)
            sources.append("match")
            flags.append(margin < MATCH_MARGIN)
        else:
            ks.append(0)
            sources.append("none")
            flags.append(True)
    return ks, sources, flags


def transform_bbox(bbox: tuple[int, int, int, int], k: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Map a bbox through ``k`` CCW 90-degree rotations of a width x height image."""
    x1, y1, x2, y2 = bbox
    pts = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for _ in range(k % 4):
        pts = [(y, width - 1 - x) for x, y in pts]
        width, height = height, width
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def contact_sheet(rows: list[list[np.ndarray]], labels: list[str], path: Path, tile: int = 128) -> None:
    """Write a review sheet: one row per piece, its 4 upright crops side by side."""
    if not rows:
        return
    per_row = max(len(r) for r in rows)
    sheet = np.full((len(rows) * tile, per_row * tile + 90, 3), 30, np.uint8)
    for r, (crops, label) in enumerate(zip(rows, labels)):
        cv2.putText(sheet, label, (4, r * tile + tile // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for c, crop in enumerate(crops):
            h, w = crop.shape[:2]
            side = max(h, w)
            pad = np.zeros((side, side, 3), np.uint8)
            pad[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop
            sheet[r * tile : (r + 1) * tile, 90 + c * tile : 90 + (c + 1) * tile] = cv2.resize(pad, (tile, tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def source_numbers(source: Path) -> dict[int, Path]:
    """Map image number -> HEIC path for every ``IMG_NNNN.HEIC`` in the source folder."""
    out: dict[int, Path] = {}
    for p in source.glob("IMG_*.HEIC"):
        m = re.fullmatch(r"IMG_(\d+)", p.stem)
        if m:
            out[int(m.group(1))] = p
    return out


CSV_FIELDS = [
    "puzzle_id",
    "piece_file",
    "rows",
    "cols",
    "row",
    "col",
    "rotation",
    "background",
    "source_image",
    "captured_at",
    "device",
    "image_w",
    "image_h",
    "piece_x1",
    "piece_y1",
    "piece_x2",
    "piece_y2",
    "applied_k",
    "orientation_source",
    "flagged",
    "bbox_suspect",
]


def load_overrides() -> dict[int, int]:
    """Manual rotation decisions from ``rotation_overrides.csv`` next to this script.

    Columns: ``image`` (number, e.g. 1353) and ``k`` (CCW 90-degree steps to
    upright, 0-3). Add a row after visually reviewing a flagged shot on the
    contact sheet; the override wins over both the arrow and the matcher.
    """
    path = Path(__file__).parent / "rotation_overrides.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {int(row["image"]): int(row["k"]) % 4 for row in csv.DictReader(f)}


def load_bbox_overrides() -> dict[int, tuple[int, int, int, int]]:
    """Manual piece bboxes from ``bbox_overrides.csv`` next to this script.

    Columns: ``image`` (number) and ``x1,y1,x2,y2`` in FINAL upright-image pixel
    coordinates. Used verbatim, replacing the detected bbox — the fix for shots
    where segmentation caught background or only part of the piece.
    """
    path = Path(__file__).parent / "bbox_overrides.csv"
    if not path.exists():
        return {}
    with path.open() as f:
        return {
            int(row["image"]): (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            for row in csv.DictReader(f)
        }


def resolve_bbox(
    det: ShotDetection, n: int, k: int, bbox_overrides: dict[int, tuple[int, int, int, int]]
) -> tuple[int, int, int, int] | None:
    """Return the piece bbox in final upright-image coords, or None if unknown.

    An explicit human override beats the detected box; detected boxes are in
    full-resolution unrotated coords and must be rotated along with the image.
    """
    if n in bbox_overrides:
        return bbox_overrides[n]
    if det.bbox is not None:
        return transform_bbox(det.bbox, k, det.width, det.height)
    return None


def ingest_puzzle(
    index: int,
    anchor: int,
    rows: int,
    cols: int,
    slug: str,
    seg: list[int],
    byno: dict[int, Path],
    opts: argparse.Namespace,
    pool: ProcessPoolExecutor,
) -> tuple[list[dict[str, object]], list[str]]:
    """Ingest one puzzle: overview + all piece groups; returns (metadata rows, warnings)."""
    puzzle_id = f"puzzle{index + 1:02d}_{slug}"
    pdir = opts.output / puzzle_id
    (pdir / "pieces").mkdir(parents=True, exist_ok=True)
    convert_heic(byno[OVERVIEW_OVERRIDES.get(slug, anchor)], pdir / "overview.jpg", opts.max_side, opts.quality)
    overrides = load_overrides()
    bbox_overrides = load_bbox_overrides()

    with tempfile.TemporaryDirectory() as td:
        jobs = [(byno[n], Path(td) / f"{n}.jpg", opts.max_side, opts.quality) for n in seg]
        detections = list(pool.map(analyze_shot, jobs))

        out_rows: list[dict[str, object]] = []
        warnings: list[str] = []
        sheet_rows: list[list[np.ndarray]] = []
        sheet_labels: list[str] = []
        for gi in range(rows * cols):
            r, c = divmod(gi, cols)
            group = detections[gi * SHOTS_PER_PIECE : (gi + 1) * SHOTS_PER_PIECE]
            numbers = seg[gi * SHOTS_PER_PIECE : (gi + 1) * SHOTS_PER_PIECE]
            ks, sources, flags = decide_rotations(group)
            for pos, n in enumerate(numbers):  # explicit human decisions beat both signals
                if n in overrides:
                    ks[pos], sources[pos], flags[pos] = overrides[n], "manual", False
            crops: list[np.ndarray] = []
            for det, n, k, src_name, flag in zip(group, numbers, ks, sources, flags):
                bgname = BACKGROUNDS[numbers.index(n)]
                rel = f"{puzzle_id}/pieces/piece_r{r:02d}_c{c:02d}_{bgname}.jpg"
                img = Image.open(det.tmp_jpg)
                if k:
                    img = img.transpose(PIL_TRANSPOSE[k])
                img.save(opts.output / rel, quality=opts.quality)

                bbox_full = resolve_bbox(det, n, k, bbox_overrides)
                if bbox_full is not None:
                    x1, y1, x2, y2 = bbox_full
                    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    crops.append(bgr[max(0, y1) : y2, max(0, x1) : x2])
                if flag:
                    warnings.append(f"{puzzle_id} r{r}c{c} IMG_{n}: orientation uncertain ({src_name})")
                if bbox_full is None:
                    warnings.append(f"{puzzle_id} r{r}c{c} IMG_{n}: piece bbox not found")
                out_rows.append(
                    {
                        "puzzle_id": puzzle_id,
                        "piece_file": rel,
                        "rows": rows,
                        "cols": cols,
                        "row": r,
                        "col": c,
                        "rotation": 0,
                        "background": bgname,
                        "source_image": f"IMG_{n}.HEIC",
                        "captured_at": det.captured_at,
                        "device": det.device,
                        "image_w": img.width,
                        "image_h": img.height,
                        "piece_x1": bbox_full[0] if bbox_full else "",
                        "piece_y1": bbox_full[1] if bbox_full else "",
                        "piece_x2": bbox_full[2] if bbox_full else "",
                        "piece_y2": bbox_full[3] if bbox_full else "",
                        "applied_k": k,
                        "orientation_source": src_name,
                        "flagged": int(flag),
                    }
                )
            mark_bbox_suspects(out_rows[-len(group) :])
            sheet_rows.append(crops)
            sheet_labels.append(f"r{r} c{c}")
        contact_sheet(sheet_rows, sheet_labels, opts.output / "review" / f"{puzzle_id}.jpg")
    return out_rows, warnings


def mark_bbox_suspects(group_rows: list[dict[str, object]]) -> None:
    """Set ``bbox_suspect`` on a 4-shot group's metadata rows.

    The same physical piece should occupy a similar bbox in all 4 shots, so a
    much smaller (or missing) box means detection only caught part of it; and a
    bbox reaching the frame edge means the mask ran into background (the piece
    is never at the edge — the arrow and margins surround it).
    """

    def box_area(gr: dict[str, object]) -> int:
        return (int(gr["piece_x2"]) - int(gr["piece_x1"])) * (int(gr["piece_y2"]) - int(gr["piece_y1"]))

    areas = [box_area(gr) for gr in group_rows if gr["piece_x1"] != ""]
    median_area = float(np.median(areas)) if areas else 0.0
    for gr in group_rows:
        ok = gr["piece_x1"] != ""
        at_edge = ok and (
            int(gr["piece_x1"]) <= 1
            or int(gr["piece_y1"]) <= 1
            or int(gr["piece_x2"]) >= int(gr["image_w"]) - 2
            or int(gr["piece_y2"]) >= int(gr["image_h"]) - 2
        )
        small = ok and median_area > 0 and box_area(gr) < 0.55 * median_area
        gr["bbox_suspect"] = int(not ok or at_edge or small)


def main() -> int:
    """Run the ingest over all (or selected) puzzles; return a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=Path.home() / "Pictures/puzzles", help="folder with IMG_*.HEIC")
    ap.add_argument("--output", type=Path, default=Path(__file__).parents[2] / "datasets/north_star/v1")
    ap.add_argument("--puzzle", action="append", help="only ingest these puzzle slugs (repeatable)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=85)
    opts = ap.parse_args()

    byno = source_numbers(opts.source)
    if not byno:
        print(f"no IMG_*.HEIC found in {opts.source}", file=sys.stderr)
        return 1
    nums = sorted(n for n in byno if n not in OVERVIEW_OVERRIDES.values())
    known = {p[3] for p in PUZZLES}
    if opts.puzzle and set(opts.puzzle) - known:
        print(f"unknown puzzle slug(s): {set(opts.puzzle) - known}", file=sys.stderr)
        return 1

    bounds = [p[0] for p in PUZZLES] + [nums[-1] + 1]
    all_rows: list[dict[str, object]] = []
    all_warnings: list[str] = []
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        for i, (anchor, rows, cols, slug) in enumerate(PUZZLES):
            if opts.puzzle and slug not in opts.puzzle:
                continue
            seg = [n for n in nums if bounds[i] < n < bounds[i + 1]]
            expected = rows * cols * SHOTS_PER_PIECE
            if len(seg) != expected:
                print(
                    f"FATAL {slug}: expected {expected} piece shots after IMG_{anchor}, found {len(seg)} — "
                    "capture-order labels cannot be trusted; fix the manifest or the folder",
                    file=sys.stderr,
                )
                return 1
            print(f"puzzle{i + 1:02d}_{slug}: overview IMG_{anchor} + {rows * cols} pieces x {SHOTS_PER_PIECE}")
            rows_out, warns = ingest_puzzle(i, anchor, rows, cols, slug, seg, byno, opts, pool)
            all_rows.extend(rows_out)
            all_warnings.extend(warns)

    if not all_rows:
        print("nothing ingested", file=sys.stderr)
        return 1

    csv_path = opts.output / "metadata.csv"
    if opts.puzzle and csv_path.exists():  # partial run: keep other puzzles' rows
        done = {str(r["puzzle_id"]) for r in all_rows}
        with csv_path.open() as f:
            kept = [row for row in csv.DictReader(f) if row["puzzle_id"] not in done]
        all_rows = sorted(
            kept + [{k: str(v) for k, v in r.items()} for r in all_rows], key=lambda r: str(r["piece_file"])
        )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nwrote {len(all_rows)} rows -> {csv_path}")
    n_flag = sum(1 for r in all_rows if str(r["flagged"]) == "1")
    print(f"flagged for review: {n_flag}  (contact sheets in {opts.output / 'review'})")
    for w in all_warnings:
        print(f"  warn: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
