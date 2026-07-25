"""Service for detecting a puzzle picture's boundary in a photo and perspective-correcting it."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
from PIL import Image

Point = Tuple[float, float]

# Corners covering the whole image, used as fallback when no quadrilateral is found
FULL_IMAGE_CORNERS: List[Point] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

# Detection runs on a downscaled copy for speed; warping uses the full-resolution image
WORKING_MAX_DIM = 1000

# A candidate region must cover at least this fraction of the photo
MIN_AREA_RATIO = 0.15

# Area ratio at (or above) which the confidence heuristic saturates
FULL_CONFIDENCE_AREA_RATIO = 0.5

# Fraction of the shorter working dimension used as the border ring for background sampling
BORDER_FRACTION = 0.05

# If the border ring's 99th-percentile color residual (RGB units, see _foreground_mask)
# exceeds this, there is no uniform background around the subject (e.g. an edge-to-edge
# image) and detection bails out. Calibrated so a textured carpet with mild illumination
# tint (~20) passes while a colored surround (~100+) bails.
MAX_BORDER_COLOR_SPREAD = 30.0

# Minimum color residual (RGB units) for a pixel to count as foreground
COLOR_RESIDUAL_FLOOR = 12.0

# GrabCut refinement runs on a further-downscaled copy: cost grows quadratically and
# region shape, not pixel accuracy, is what the quad fit needs
GRABCUT_MAX_DIM = 750

# Seed components smaller than this fraction of the frame are dropped before GrabCut
# (stray specks — e.g. a photographer's shadow patch — would otherwise grow into the region)
GRABCUT_SEED_COMPONENT_RATIO = 0.005

# GrabCut is skipped when the surviving seeds cover less than this fraction of the frame:
# with almost no foreground evidence it would hallucinate a region from noise
GRABCUT_MIN_SEED_RATIO = 0.02

GRABCUT_ITERATIONS = 3

# Auto-Canny spread: thresholds are set to (1 ± sigma) × the image's median intensity,
# so the edge detector adapts to each photo's exposure rather than using fixed levels
EDGE_CANNY_SIGMA = 0.33

# Kernel size for dilating the edge map, closing small gaps in the puzzle's border
# so its outline forms a single traceable contour
EDGE_DILATE_KERNEL = 5

# Polygon-approximation tolerances (as a fraction of contour perimeter) tried in order
# when reducing an edge contour to a quadrilateral
EDGE_APPROX_EPSILONS = (0.02, 0.04, 0.06)

# Widest across-border probe (working pixels, full width) the edge-support term may
# look through when deciding whether a candidate side coincides with a detected edge.
# `_probe_reach` narrows it further on busy edge maps — the wider the window, the more
# likely a side "hits" an edge purely by chance — so this is the ceiling, not the
# window: a side is sampled at points along its length, and each sample searches at
# most this far across the border for a set pixel.
EDGE_SUPPORT_THICKNESS = 2 * EDGE_DILATE_KERNEL + 1

# --- Line-intersection candidates (see PuzzleFrameDetector._line_quads) ---

# A Hough segment must span at least this fraction of the shorter working dimension.
# The puzzle's own borders are the longest straight structures in a well-framed photo.
LINE_MIN_LENGTH_FRACTION = 0.25

# Hough accumulator votes, as a fraction of the minimum segment length.
LINE_VOTE_FRACTION = 0.6

# Segments this many degrees from a family's dominant direction still belong to it.
LINE_FAMILY_WINDOW_DEG = 18.0

# Two lines in the same family whose offsets differ by less than this fraction of the
# working long side are the same border seen twice, and get merged.
LINE_MERGE_OFFSET_FRACTION = 0.03

# How many lines per family are carried into the pairwise enumeration. Bounds the
# candidate count at (K choose 2)^2 quads — 225 at K=6, which sampled scoring
# (see _side_edge_support) handles in a few milliseconds.
LINE_CANDIDATES_PER_FAMILY = 6

# The two families must be at least this far from parallel for their intersections to
# be a stable quad rather than a sliver.
LINE_MIN_FAMILY_SEPARATION_DEG = 45.0

# --- Unified candidate scoring (see PuzzleFrameDetector._score_quad) ---

# Exponent applied to the rectangularity term (see _score_quad).
RECTANGULARITY_POWER = 2.0

# Candidates scoring below this are dropped rather than returned as a best-of-a-bad-lot.
MIN_CANDIDATE_SCORE = 0.02

# Chance of a spurious edge hit that the probe window is sized to keep (see _probe_reach).
CHANCE_TARGET = 0.5

# Chance level above which the edge-support statistic carries no information (see
# _side_edge_support). Only reachable when the texture is so dense that even the
# narrowest window is saturated.
MAX_INFORMATIVE_CHANCE = 0.85

# Points sampled along each side when scoring it. Sampling rather than rasterizing keeps
# scoring cheap enough to run over every candidate from every generator.
SIDE_SAMPLE_COUNT = 128

# How far (working pixels) either side of a border the contrast term reads its two
# brightness samples — outside the blurred transition, inside the object and background.
CONTRAST_PROBE_DISTANCE = 8.0

# Gray-level difference a sample must show, in the quad's own polarity, to count as
# sitting on a real boundary. Low on purpose: the term is about the *fraction* of a side
# that behaves like a boundary, so a subtle but consistent border should pass while a
# side crossing open background or the object's own interior should not.
CONTRAST_MIN_LEVEL = 6.0


@dataclass
class QuadCandidate:
    """One detected quadrilateral, with the score that decides between candidates.

    Attributes:
        corners: Four (x, y) points normalized to [0, 1], ordered TL, TR, BR, BL.
        source: Which generator produced it — "mask", "grabcut", "edge" or "lines".
        confidence: The unified score in [0, 1]; comparable across sources.
        components: The score's individual terms, for diagnostics.
    """

    corners: List[Point]
    source: str
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)


def _order_corners(points: "np.ndarray") -> "np.ndarray":
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left].

    Uses the standard sum/diff heuristic: top-left has the smallest x+y,
    bottom-right the largest x+y, top-right the smallest y-x, bottom-left
    the largest y-x.

    Args:
        points: Array of shape (4, 2) with (x, y) coordinates.

    Returns:
        Array of shape (4, 2) ordered TL, TR, BR, BL.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()  # y - x
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _chroma(pixels: "np.ndarray") -> "np.ndarray":
    """Compute brightness-invariant chromaticity (each channel divided by the channel sum).

    Shadows keep the background's chroma while differing in brightness, so
    chroma distance separates the subject from shadows on the background.

    Args:
        pixels: Float array of RGB values, last axis of size 3.

    Returns:
        Array of the same shape with channels normalized to sum to 1.
    """
    return pixels / (pixels.sum(axis=-1, keepdims=True) + 1e-6)


def _side_samples(start: "np.ndarray", end: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray", int]:
    """Sample points along a quad side, with the side's unit normal.

    Args:
        start: The side's first corner.
        end: The side's second corner.

    Returns:
        A tuple of (points, normal, count); count is 0 for a degenerate side.
    """
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1:
        return np.zeros((0, 2)), np.zeros(2), 0
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    count = max(8, min(SIDE_SAMPLE_COUNT, int(length)))
    steps = np.linspace(0.0, 1.0, count)
    points = np.asarray(start, dtype=np.float64)[None, :] + (direction * length)[None, :] * steps[:, None]
    return points, normal, count


def _edge_map_is_informative(edge_density: float) -> bool:
    """Whether "is this side on an edge?" can still separate candidates.

    Args:
        edge_density: Fraction of the edge map that is set.

    Returns:
        False when even the narrowest probe window is saturated.
    """
    reach = _probe_reach(edge_density)
    return bool(1.0 - (1.0 - edge_density) ** (2 * reach + 1) <= MAX_INFORMATIVE_CHANCE)


def _probe_reach(edge_density: float) -> int:
    """How far across a border to look for a supporting edge, given the texture.

    The window trades tolerance against discrimination: wide enough to forgive
    the pixel or two a fitted line sits off the real border, narrow enough
    that finding an edge in it still means something. On a busy edge map a
    wide window contains an edge no matter where it is placed, so the width is
    chosen to keep the chance of a spurious hit at `CHANCE_TARGET` — the
    denser the texture, the more precisely a real border must be located to
    earn credit for it.

    Args:
        edge_density: Fraction of the edge map that is set.

    Returns:
        The half-width in pixels, at least 1.
    """
    if edge_density <= 0.0:
        return EDGE_SUPPORT_THICKNESS // 2
    affordable = math.log(1.0 - CHANCE_TARGET) / math.log(1.0 - min(edge_density, 0.99))
    return int(max(1, min(EDGE_SUPPORT_THICKNESS // 2, (affordable - 1) // 2)))


def _side_edge_support(start: "np.ndarray", end: "np.ndarray", edges: "np.ndarray", edge_density: float) -> float:
    """Fraction of a side backed by detected edges, in excess of chance.

    Sampled along the side rather than rasterized: the scorer runs over every
    candidate from four generators, and a full-frame mask per side would make
    that cost dominate detection. A sample counts as backed when any edge
    pixel lies within `EDGE_SUPPORT_THICKNESS` across the border, which
    tolerates the pixel or two a fitted line can sit off the real one.

    Chance level is computed for that same "any within the window" statistic,
    not for a single pixel: on a texture-saturated edge map a window of a
    dozen pixels contains an edge essentially always, so a per-pixel chance
    level would score random noise as perfect support.

    Args:
        start: The side's first corner.
        end: The side's second corner.
        edges: The dilated binary edge map.
        edge_density: Fraction of the edge map that is set.

    Returns:
        Support in [0, 1].
    """
    points, normal, count = _side_samples(start, end)
    if count == 0 or edge_density >= 1.0:
        return 0.0
    height, width = edges.shape[:2]
    reach = _probe_reach(edge_density)
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    probes = points[None, :, :] + normal[None, None, :] * offsets[:, None, None]
    xs = np.clip(np.rint(probes[..., 0]), 0, width - 1).astype(np.int32)
    ys = np.clip(np.rint(probes[..., 1]), 0, height - 1).astype(np.int32)
    hit = float(np.mean(edges[ys, xs].any(axis=0)))
    chance = 1.0 - (1.0 - edge_density) ** offsets.size
    if chance > MAX_INFORMATIVE_CHANCE:
        # Almost every window contains an edge, so "is this side on an edge?" no longer
        # separates anything. Report no support rather than a ratio of two numbers that
        # are both ~1 — which is how random texture scores as a perfect border.
        return 0.0
    return max(0.0, (hit - chance) / (1.0 - chance))


def _side_contrast_profile(
    start: "np.ndarray", end: "np.ndarray", work: "np.ndarray", centroid: "np.ndarray"
) -> "np.ndarray":
    """Per-sample colour just inside a side minus just outside it.

    Probed a short way off the border on both sides so the transition itself,
    blurred by the working-image blur, doesn't dominate.

    Returned as full RGB differences rather than a brightness difference: a
    colourful puzzle on gray carpet can step hard in hue while barely changing
    in luminance, and a brightness-only reading calls that border invisible.
    The caller projects these onto a single colour axis to recover a signed
    quantity (see `_score_quad`).

    Kept as a profile rather than reduced to a mean: a side that is a real
    boundary for most of its length and runs through the object's own interior
    for the rest — which is exactly what a quad that has cut a corner looks
    like — still averages to a healthy contrast. The fraction of the side that
    behaves like a boundary does not.

    Args:
        start: The side's first corner.
        end: The side's second corner.
        work: Working RGB image.
        centroid: The quad's centroid, which fixes which way is "inside".

    Returns:
        An (N, 3) array of inside-minus-outside colour differences; empty for
        a degenerate side.
    """
    points, normal, count = _side_samples(start, end)
    if count == 0:
        return np.zeros((0, 3))
    height, width = work.shape[:2]
    # Orient the normal outward, then read symmetric probes either side of it.
    if float(np.dot(normal, centroid - points[count // 2])) > 0:
        normal = -normal

    def sample(distance: float) -> "np.ndarray":
        probes = points + normal[None, :] * distance
        xs = np.clip(np.rint(probes[:, 0]), 0, width - 1).astype(np.int32)
        ys = np.clip(np.rint(probes[:, 1]), 0, height - 1).astype(np.int32)
        return work[ys, xs].astype(np.float64)

    return cast("np.ndarray", sample(-CONTRAST_PROBE_DISTANCE) - sample(CONTRAST_PROBE_DISTANCE))


@dataclass
class _Line:
    """A straight line in normal form, with the segment evidence behind it.

    Attributes:
        angle: Direction in degrees, in [0, 180).
        offset: Signed distance from the origin along the line's normal.
        weight: Total length of the segments supporting it.
    """

    angle: float
    offset: float
    weight: float

    @property
    def normal(self) -> Tuple[float, float]:
        """The line's unit normal, so that normal · point = offset.

        Returns:
            The (a, b) normal components.
        """
        radians = math.radians(self.angle)
        return -math.sin(radians), math.cos(radians)


def _segment_to_line(segment: "np.ndarray") -> _Line:
    """Convert a Hough segment into normal form, weighted by its length.

    Args:
        segment: The (x1, y1, x2, y2) endpoints.

    Returns:
        The corresponding line.
    """
    x1, y1, x2, y2 = (float(value) for value in segment)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    line = _Line(angle=angle, offset=0.0, weight=math.hypot(x2 - x1, y2 - y1))
    first, second = line.normal
    line.offset = first * x1 + second * y1
    return line


def _mean_angle(lines: List[_Line]) -> float:
    """Length-weighted mean direction of a family, in degrees in [0, 180).

    Averaged over doubled angles: directions live on a half-circle, where 179°
    and 1° are 2° apart, and a plain arithmetic mean of the two returns 90° —
    perpendicular to both, and enough to make a family look parallel to its
    own perpendicular.

    Args:
        lines: The family's lines.

    Returns:
        The mean direction in degrees.
    """
    doubled = np.radians([2.0 * line.angle for line in lines])
    weights = np.array([line.weight for line in lines])
    mean = math.atan2(float(np.sum(weights * np.sin(doubled))), float(np.sum(weights * np.cos(doubled))))
    return math.degrees(mean / 2.0) % 180.0


def _split_into_families(lines: List[_Line]) -> Optional[Tuple[List[_Line], List[_Line]]]:
    """Group lines into two roughly-orthogonal direction families.

    A rectangle's borders come in two perpendicular pairs, so the dominant
    direction (by supporting length) and its perpendicular define the two
    families. Lines belonging to neither are scene content and are dropped.

    Args:
        lines: All detected lines.

    Returns:
        The two families, or None when either is empty or they are too close
        to parallel to intersect stably.
    """
    if not lines:
        return None
    dominant = max(lines, key=lambda line: line.weight).angle

    def separation(angle: float, reference: float) -> float:
        difference = abs(angle - reference) % 180.0
        return min(difference, 180.0 - difference)

    first = [line for line in lines if separation(line.angle, dominant) <= LINE_FAMILY_WINDOW_DEG]
    perpendicular = (dominant + 90.0) % 180.0
    second = [line for line in lines if separation(line.angle, perpendicular) <= LINE_FAMILY_WINDOW_DEG]
    if len(first) < 2 or len(second) < 2:
        return None
    if separation(_mean_angle(first), _mean_angle(second)) < LINE_MIN_FAMILY_SEPARATION_DEG:
        return None
    return first, second


def _merge_lines(family: List[_Line], long_side: int) -> List[_Line]:
    """Merge near-duplicate lines in a family, strongest first.

    One border produces many Hough segments at slightly different offsets;
    without merging, the pairwise enumeration would spend all its candidates
    on variations of the same two lines.

    Args:
        family: Lines sharing a direction.
        long_side: Working image long side, setting the merge tolerance.

    Returns:
        The merged lines, sorted by descending supporting length.
    """
    tolerance = LINE_MERGE_OFFSET_FRACTION * long_side
    merged: List[_Line] = []
    for line in sorted(family, key=lambda candidate: candidate.weight, reverse=True):
        for existing in merged:
            if abs(existing.offset - line.offset) < tolerance:
                existing.weight += line.weight
                break
        else:
            merged.append(_Line(line.angle, line.offset, line.weight))
    return merged


def _line_intersection(first: _Line, second: _Line) -> Optional[Point]:
    """Intersect two lines given in normal form.

    Args:
        first: The first line.
        second: The second line.

    Returns:
        The (x, y) intersection, or None when they are parallel.
    """
    matrix = np.array([first.normal, second.normal])
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-9:
        return None
    solution = np.linalg.solve(matrix, np.array([first.offset, second.offset]))
    return float(solution[0]), float(solution[1])


def _quad_from_lines(
    first: _Line, second: _Line, third: _Line, fourth: _Line, work_w: int, work_h: int
) -> Optional["np.ndarray"]:
    """Intersect two pairs of lines into a quadrilateral.

    Args:
        first: One line of the first family.
        second: The other line of the first family.
        third: One line of the second family.
        fourth: The other line of the second family.
        work_w: Working image width.
        work_h: Working image height.

    Returns:
        The quad, shape (4, 2), or None when any corner is missing or falls
        far outside the frame.
    """
    margin = 0.15 * max(work_w, work_h)
    corners: List[Point] = []
    for along in (first, second):
        for across in (third, fourth):
            point = _line_intersection(along, across)
            if point is None:
                return None
            if not (-margin <= point[0] <= work_w + margin and -margin <= point[1] <= work_h + margin):
                return None
            corners.append(point)
    return _order_corners(np.array(corners, dtype=np.float32))


class PuzzleFrameDetector:
    """Detects the puzzle picture in a photo and warps it to a flat crop.

    Detection is a **fusion of four generators**, not a fallback chain. Each
    proposes quadrilaterals from a different kind of evidence:

    * `mask` — segments the subject from a roughly uniform background
      (sampled from the photo's border) using a color residual, which ignores
      shadows, plus a brighter-than-background luminance rule.
    * `grabcut` — when that mask has real evidence but fragments (a washed-out
      puzzle under warm light), grows the fragments to the subject's true
      color boundary.
    * `edge` — traces closed contours in the edge map and reduces each to a
      convex quad.
    * `lines` — intersects pairs of long straight Hough lines.

    Every candidate is then scored on one scale by `_score_quad` and the best
    wins. The layered "first path that answers, wins" arrangement this
    replaced had no way to tell a confident wrong answer from a tentative
    right one: on a box photographed on carpet, the contour trace wrapped the
    box *and its cast shadow* into one region and returned it, and nothing
    downstream ever got to disagree.

    What rejects that quad is `_score_quad`'s border-contrast term, not the
    choice of generator. Under directional light a cast shadow's boundary is
    every bit as long and as straight as the box's own edge — the line
    generator can and does build quads from it — so edge evidence cannot
    separate the two. The direction of the colour step across each side can:
    a real object differs from its surroundings the same way along its whole
    outline, while the shadow quad steps toward shadow on one side and toward
    bright cardboard on the others.

    The puzzle picture is a strong quadrilateral with straight edges, so every
    generator looks for that rectangle explicitly rather than asking a
    salient-object model — a salient-object model segments whatever the
    artwork depicts (an elephant, a lion) instead of the puzzle's outline.
    """

    def detect_corners(self, image: Image.Image) -> Tuple[List[Point], float]:
        """Detect the puzzle picture's quadrilateral in the image.

        The image should already be RGB and EXIF-corrected. Detection runs on a
        downscaled copy; returned corners are normalized to [0, 1] so they apply
        to the original image as well.

        Args:
            image: The photo to analyze.

        Returns:
            A tuple of (corners, confidence). Corners are ordered
            [top-left, top-right, bottom-right, bottom-left], normalized 0-1.
            Falls back to the full-image corners with confidence 0.0 when no
            suitable region is found.
        """
        candidates = self.detect_candidates(image)
        if not candidates:
            return list(FULL_IMAGE_CORNERS), 0.0
        best = candidates[0]
        return best.corners, best.confidence

    def detect_candidates(self, image: Image.Image) -> List[QuadCandidate]:
        """Generate and score every quadrilateral candidate, best first.

        Each generator answers a different question — "what is not background?"
        (mask), "what grows from those fragments?" (GrabCut), "what closed
        outline is there?" (edge contours), "what pairs of long straight lines
        are there?" (lines) — and each is unreliable on the scenes the others
        handle. Rather than taking the first one that answers, all four run and
        their candidates compete under one score (`_score_quad`), so the choice
        is made on evidence instead of on generator order.

        Args:
            image: The photo to analyze.

        Returns:
            Scored candidates sorted by descending confidence; empty when
            nothing scored above `MIN_CANDIDATE_SCORE`.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        scale = min(1.0, WORKING_MAX_DIM / max(width, height))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))))
        work = cv2.GaussianBlur(rgb, (7, 7), 0).astype(np.float32)
        work_h, work_w = work.shape[:2]
        edges = self._edge_map(work)
        quads: List[Tuple["np.ndarray", str]] = []
        mask = self._foreground_mask(work)
        if mask is not None:
            from_mask = self._quad_from_mask(mask, work_w, work_h)
            if from_mask is not None:
                quads.append((from_mask, "mask"))
            # Runs even when the mask already yielded a quad, and deliberately:
            # this is a fusion, not a chain, so "the cheaper generator answered"
            # is not evidence that its answer is the better one. On 20 real
            # captures GrabCut wins 5, and in 2 of those a mask quad existed
            # too and lost — skipping it there would take the worse quad on 10%
            # of captures.
            #
            # It is the expensive one: ~69% of detection time, up to 650ms on a
            # 2048px image. Affordable because detection runs once per puzzle
            # added, not per frame or per piece. If that ever stops being true,
            # make GrabCut itself cheaper rather than conditional.
            from_grabcut = self._grabcut_quad(work, mask, work_w, work_h)
            if from_grabcut is not None:
                quads.append((from_grabcut, "grabcut"))
        quads.extend((quad, "edge") for quad in self._edge_quads(edges, work_w, work_h))
        quads.extend((quad, "lines") for quad in self._line_quads(edges, work_w, work_h))

        candidates: List[QuadCandidate] = []
        for quad, source in quads:
            confidence, components = self._score_quad(quad, edges, work_w, work_h, work)
            if components["rank_score"] < MIN_CANDIDATE_SCORE:
                continue
            ordered = _order_corners(quad)
            corners: List[Point] = [
                (float(np.clip(x / work_w, 0.0, 1.0)), float(np.clip(y / work_h, 0.0, 1.0))) for x, y in ordered
            ]
            candidates.append(QuadCandidate(corners, source, confidence, components))
        candidates.sort(key=lambda candidate: candidate.components["rank_score"], reverse=True)
        return candidates

    def warp(self, image: Image.Image, corners: List[Point]) -> Image.Image:
        """Perspective-warp and crop the image using 4 normalized corners.

        Args:
            image: The full-resolution photo to warp.
            corners: Four (x, y) points normalized to [0, 1], in any consistent
                order; they are re-ordered TL, TR, BR, BL internally.

        Returns:
            A new image containing only the quadrilateral region, straightened
            to a rectangle sized from the corner-to-corner distances.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        pixels = np.array([(x * width, y * height) for x, y in corners], dtype=np.float32)
        src = _order_corners(pixels)
        tl, tr, br, bl = src

        dst_w = max(1, round(max(math.dist(tl, tr), math.dist(bl, br))))
        dst_h = max(1, round(max(math.dist(tl, bl), math.dist(tr, br))))
        dst = np.array([(0, 0), (dst_w - 1, 0), (dst_w - 1, dst_h - 1), (0, dst_h - 1)], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(rgb, matrix, (dst_w, dst_h))
        return Image.fromarray(warped)

    def _foreground_mask(self, work: "np.ndarray") -> Optional["np.ndarray"]:
        """Segment the subject from the background sampled along the photo's border.

        Deviation from the background color is measured as the residual
        ``||pixel - bg_chroma * luminance||`` in RGB units — the pixel compared
        against the background's chroma scaled to the pixel's own brightness.
        This equals chroma distance times luminance, which keeps the shadow
        invariance of chroma but stays robust where raw chroma is not: in dark
        pixels (a carpet's fiber shadows) the channel sum is small, so sensor
        and JPEG noise dominate the chroma ratio, while the residual stays
        near the noise amplitude.

        Args:
            work: Blurred float RGB working image.

        Returns:
            A uint8 mask (0/255) of foreground pixels, or None when the border
            is not uniform enough to represent a background.
        """
        work_h, work_w = work.shape[:2]
        margin = max(2, round(BORDER_FRACTION * min(work_w, work_h)))
        border_mask = np.zeros((work_h, work_w), dtype=bool)
        border_mask[:margin] = border_mask[-margin:] = True
        border_mask[:, :margin] = border_mask[:, -margin:] = True

        pix_chroma = _chroma(work)
        bg_chroma = np.median(pix_chroma[border_mask], axis=0)

        luminance = work.sum(axis=2)
        residual = np.linalg.norm(work - bg_chroma[None, None, :] * luminance[..., None], axis=2)
        border_spread = float(np.percentile(residual[border_mask], 99))
        if border_spread > MAX_BORDER_COLOR_SPREAD:
            # Border pixels vary too much: no uniform background (e.g. edge-to-edge picture)
            return None

        residual_thresh = max(COLOR_RESIDUAL_FLOOR, 1.5 * border_spread)

        # Brighter-than-background pixels are foreground too (a shadow is never brighter)
        border_lum = luminance[border_mask]
        bg_lum = float(np.median(border_lum))
        lum_spread = float(np.percentile(np.abs(border_lum - bg_lum), 99))
        lum_thresh = max(60.0, 1.5 * lum_spread)

        mask = ((residual > residual_thresh) | (luminance > bg_lum + lum_thresh)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), dtype=np.uint8))
        return cast("np.ndarray", mask)

    def _grabcut_quad(
        self, work: "np.ndarray", seed_mask: "np.ndarray", work_w: int, work_h: int
    ) -> Optional["np.ndarray"]:
        """Recover the subject region with GrabCut when the color mask is fragmented.

        A washed-out puzzle under warm light can be so close to the background
        color that thresholding only picks up scattered colorful fragments,
        none large enough to pass the area gate. GrabCut — seeded with those
        fragments as probable foreground and the border ring as definite
        background — models both sides as color mixtures with spatial
        smoothness, growing the fragments out to the subject's true color
        boundary. The recovered foreground is often still fragmented — a ring
        with washed-out holes, or patches along the subject's frame — so the
        quad is fitted (and the area gate applied) on the convex hull of all
        significant recovered components, treating them as one convex subject.

        Args:
            work: Blurred float RGB working image.
            seed_mask: uint8 foreground mask (0/255) from _foreground_mask.
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            The quad, shape (4, 2), or None when the seeds are too sparse or
            no recovered region covers at least MIN_AREA_RATIO.
        """
        total_area = work_w * work_h
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
        seeds = np.zeros_like(seed_mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= GRABCUT_SEED_COMPONENT_RATIO * total_area:
                seeds[labels == label] = 255
        if np.count_nonzero(seeds) < GRABCUT_MIN_SEED_RATIO * total_area:
            return None

        scale = min(1.0, GRABCUT_MAX_DIM / max(work_w, work_h))
        grab_w = max(1, round(work_w * scale))
        grab_h = max(1, round(work_h * scale))
        image = cv2.resize(np.clip(work, 0, 255).astype(np.uint8), (grab_w, grab_h))
        seeds_small = cv2.resize(seeds, (grab_w, grab_h), interpolation=cv2.INTER_NEAREST)

        margin = max(2, round(BORDER_FRACTION * min(grab_w, grab_h)))
        grab = np.full((grab_h, grab_w), cv2.GC_PR_BGD, dtype=np.uint8)
        grab[seeds_small > 0] = cv2.GC_PR_FGD
        grab[:margin] = grab[-margin:] = cv2.GC_BGD
        grab[:, :margin] = grab[:, -margin:] = cv2.GC_BGD

        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        try:
            # The rect argument is unused in GC_INIT_WITH_MASK mode
            cv2.grabCut(
                image,
                grab,
                (0, 0, 0, 0),
                background_model,
                foreground_model,
                GRABCUT_ITERATIONS,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return None

        foreground = ((grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        foreground = cv2.resize(foreground, (work_w, work_h), interpolation=cv2.INTER_NEAREST)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((11, 11), dtype=np.uint8))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        kept = np.zeros_like(foreground)
        kept_area = 0
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= GRABCUT_SEED_COMPONENT_RATIO * total_area:
                kept[labels == label] = 255
                kept_area += int(stats[label, cv2.CC_STAT_AREA])
        points = cv2.findNonZero(kept)
        if points is None:
            return None
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        if hull_area / total_area < MIN_AREA_RATIO:
            return None
        return self._quad_from_hull(hull)

    def _edge_map(self, work: "np.ndarray") -> "np.ndarray":
        """Build the dilated Canny edge map every candidate is scored against.

        Auto-thresholded from the image's median intensity so the detector
        adapts to each photo's exposure, then dilated to close the small gaps
        that stop a border tracing as one contour.

        Args:
            work: Blurred float RGB working image.

        Returns:
            A uint8 binary edge map.
        """
        gray = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        median = float(np.median(gray))
        lower = int(max(0, (1.0 - EDGE_CANNY_SIGMA) * median))
        upper = int(min(255, (1.0 + EDGE_CANNY_SIGMA) * median))
        edges = cv2.Canny(gray, lower, upper)
        return cast("np.ndarray", cv2.dilate(edges, np.ones((EDGE_DILATE_KERNEL, EDGE_DILATE_KERNEL), dtype=np.uint8)))

    def _edge_quads(self, edges: "np.ndarray", work_w: int, work_h: int) -> List["np.ndarray"]:
        """Trace closed edge contours and reduce each to a convex quadrilateral.

        Answers "what closed outline is there?". Strong when the puzzle's
        border traces as one unbroken loop; weak when a cast shadow or an
        adjacent object welds itself onto that loop, which is what the other
        generators and the scorer are there to catch.

        Args:
            edges: The dilated binary edge map.
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            Every large convex quadrilateral found, unscored.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = work_w * work_h
        quads: List["np.ndarray"] = []
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA_RATIO * image_area:
                continue
            quad = self._quad_from_contour(contour)
            if quad is not None:
                quads.append(quad)
        return quads

    def _line_quads(self, edges: "np.ndarray", work_w: int, work_h: int) -> List["np.ndarray"]:
        """Build quadrilaterals from pairs of long straight lines.

        Answers "what pairs of long straight lines are there?", and is the
        generator that survives what breaks the others. A puzzle's four
        borders are the longest straight structures in a well-framed photo,
        and they stay straight even when the region around them does not
        segment cleanly: a cast shadow beside the box has a soft, curved
        boundary that produces no long line, so a quad including the shadow
        cannot be built here at all — where a contour trace merrily wraps
        around both.

        Segments are grouped into two roughly-orthogonal families by
        direction, near-duplicate lines within a family are merged, and the
        strongest few per family are enumerated pairwise: two lines from each
        family intersect into one quad.

        Args:
            edges: The dilated binary edge map.
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            Candidate quads, shape (4, 2) each, unscored and unordered.
        """
        short_side = min(work_w, work_h)
        minimum_length = int(LINE_MIN_LENGTH_FRACTION * short_side)
        segments = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=max(20, int(LINE_VOTE_FRACTION * minimum_length)),
            minLineLength=minimum_length,
            maxLineGap=int(0.02 * short_side),
        )
        if segments is None:
            return []

        lines = [_segment_to_line(segment) for segment in segments.reshape(-1, 4)]
        families = _split_into_families(lines)
        if families is None:
            return []

        quads: List["np.ndarray"] = []
        first_family, second_family = (
            _merge_lines(family, max(work_w, work_h))[:LINE_CANDIDATES_PER_FAMILY] for family in families
        )
        for first_index in range(len(first_family)):
            for second_index in range(first_index + 1, len(first_family)):
                for third_index in range(len(second_family)):
                    for fourth_index in range(third_index + 1, len(second_family)):
                        quad = _quad_from_lines(
                            first_family[first_index],
                            first_family[second_index],
                            second_family[third_index],
                            second_family[fourth_index],
                            work_w,
                            work_h,
                        )
                        if quad is not None:
                            quads.append(quad)
        return quads

    def _quad_from_contour(self, contour: "np.ndarray") -> Optional["np.ndarray"]:
        """Reduce an edge contour to a convex quadrilateral, if it is one.

        Tries increasingly coarse polygon approximations; accepts the first
        that yields exactly four convex points. Requiring a clean four-sided
        approximation is what rejects the artwork's subject: a puzzle border
        traces a quadrilateral, an elephant does not.

        A real box's outline is often ragged rather than clean — a glare seam,
        or a slightly larger box stacked underneath, welds extra vertices onto
        the contour that no approximation tolerance removes. The convex hull
        strips those concavities while preserving the outer quadrilateral, so
        when the raw contour fails, its hull gets the same four-convex-points
        test (mirroring how the mask path quad-fits hulls).

        Args:
            contour: A contour from cv2.findContours.

        Returns:
            Array of shape (4, 2), float32, unordered, or None if neither the
            contour nor its convex hull approximates a convex quadrilateral.
        """
        for candidate in (contour, cv2.convexHull(contour)):
            perimeter = cv2.arcLength(candidate, closed=True)
            for epsilon in EDGE_APPROX_EPSILONS:
                approx = cv2.approxPolyDP(candidate, epsilon * perimeter, closed=True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    return approx.reshape(4, 2).astype(np.float32)
        return None

    def _score_quad(
        self, quad: "np.ndarray", edges: "np.ndarray", work_w: int, work_h: int, work: Optional["np.ndarray"] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Score how puzzle-like a candidate quad is, in [0, 1].

        The one scale every generator's candidates are compared on. Four terms
        multiply together, each answering something a wrong quad tends to fail
        while a right one does not.

        **Rectangularity**, squared. A rectangle stays highly rectangular
        under any plausible viewing tilt, so real quads cluster near 1.0 and
        the term mostly exists to reject sprawling lopsided ones — which needs
        amplifying, since the honest gap between 0.98 and 0.69 is otherwise
        smaller than the coverage a sprawling quad gains.

        **Coverage**, saturating at `FULL_CONFIDENCE_AREA_RATIO`.

        **Edge support**, per side, counting only the excess over the edge
        map's overall density: on a densely textured background (a carpet) any
        outline hits plenty of edge pixels by chance.

        **Border contrast consistency**, which is what separates the puzzle
        from its own cast shadow. Under directional light a box throws a
        shadow whose boundary is every bit as long and straight as the box's
        own edges, so edge support alone happily accepts a quad whose left
        side is the shadow's outer boundary — the real failure seen on a
        1000-piece box photographed on carpet. But a real object is brighter
        (or darker) than its surroundings on *all four* sides, whereas that
        quad steps down into shadow on one side and up onto the bright box on
        the others.

        The term is the *fraction* of each side that behaves like a boundary
        of the quad's own polarity, scored on the weakest side — not the mean
        contrast along it. A side that is a genuine border for most of its
        length and cuts through the object's own interior for the rest, which
        is what a quad that has clipped a corner looks like, keeps a perfectly
        healthy mean.

        Args:
            quad: The candidate quadrilateral, shape (4, 2).
            edges: The (dilated) binary edge map the quad was found in.
            work_w: Working image width.
            work_h: Working image height.
            work: Working RGB image for the contrast term. Omitting it skips
                that term (the term scores 1.0).

        Returns:
            A (confidence, components) tuple; components carries the
            individual terms for diagnostics.
        """
        rect_w, rect_h = cv2.minAreaRect(quad.reshape(-1, 1, 2))[1]
        rect_area = rect_w * rect_h
        quad_area = cv2.contourArea(quad.reshape(-1, 1, 2))
        rectangularity = quad_area / rect_area if rect_area > 0 else 0.0
        area_ratio = quad_area / (work_w * work_h)
        coverage = min(1.0, area_ratio / FULL_CONFIDENCE_AREA_RATIO)
        if area_ratio < MIN_AREA_RATIO:
            return 0.0, {
                "rectangularity": float(rectangularity),
                "coverage": float(coverage),
                "edge_support": 0.0,
                "weakest_side_support": 0.0,
                "border_contrast": 0.0,
                "evidence": 0.0,
                "rank_score": 0.0,
            }

        edge_density = float(np.count_nonzero(edges)) / edges.size
        ordered = _order_corners(quad)
        sides = [(ordered[index], ordered[(index + 1) % 4]) for index in range(4)]
        if _edge_map_is_informative(edge_density):
            supports = [_side_edge_support(start, end, edges, edge_density) for start, end in sides]
            edge_support = float(np.mean(supports))
        else:
            # A saturated edge map cannot separate any candidate from any other — it is a
            # property of the photo, not of this quad. Abstaining (scoring every candidate
            # 1.0 on this term) leaves the ranking to the terms that still carry signal;
            # scoring 0 would instead veto every candidate and return nothing at all.
            supports = [1.0] * 4
            edge_support = 1.0

        contrast = 1.0
        if work is not None:
            centroid = ordered.mean(axis=0)
            profiles = [_side_contrast_profile(start, end, work, centroid) for start, end in sides]
            populated = [profile for profile in profiles if profile.size]
            combined = np.concatenate(populated) if populated else np.zeros((0, 3))
            axis = combined.mean(axis=0) if combined.size else np.zeros(3)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-6:
                # One colour axis for the whole quad, so "steps the same way as the
                # other sides" is what counts — an object differs from its
                # surroundings in one direction, not toward cream on three sides and
                # toward shadow on the fourth.
                axis = axis / norm
                fractions = [
                    float(np.mean(profile @ axis > CONTRAST_MIN_LEVEL)) if profile.size else 0.0 for profile in profiles
                ]
                contrast = float(min(fractions))

        # Rectangularity is squared: a rectangle stays highly rectangular under any
        # plausible viewing tilt (0.95+ for a handheld overview shot), so the gap between
        # a real quad and a lopsided one is small on this term and needs amplifying.
        # Without it, a sprawling near-frame-sized quad at 0.69 outscores a correct one
        # at 0.98 on the strength of its coverage alone.
        # Evidence that *this* quad is a real boundary, which is the reported
        # confidence. Measured on the labelled captures in scripts/rectify_quality/, it
        # spans 0.18 (a correct but poorly-evidenced detection on busy carpet) to 0.92
        # (a clean box shot), with a photo containing no puzzle at all scoring 0.24 —
        # a usable scale as it stands. It remains a heuristic, not a probability: it
        # separates "found a rectangle" from "found nothing" far better than it tracks
        # how accurate the corners are.
        evidence = rectangularity**RECTANGULARITY_POWER * edge_support * contrast
        confidence = float(np.clip(evidence, 0.0, 1.0))
        # Coverage ranks but does not get reported: it is a prior about what we are
        # looking for (the big rectangle, not the smaller one inside it), not evidence
        # about whether this quad is right. Folding it into the reported number told the
        # user a pixel-perfect detection of a puzzle filling a fifth of the frame was
        # "uncertain", which is a statement about the framing, not the detection.
        components = {
            "rectangularity": float(rectangularity),
            "coverage": float(coverage),
            "edge_support": float(edge_support),
            "weakest_side_support": float(min(supports)),
            "border_contrast": float(contrast),
            "evidence": float(evidence),
            "rank_score": float(evidence * coverage),
        }
        return confidence, components

    def _quad_from_mask(self, mask: "np.ndarray", work_w: int, work_h: int) -> Optional["np.ndarray"]:
        """Fit a quadrilateral to the largest region of a foreground mask.

        Args:
            mask: uint8 foreground mask (0/255).
            work_w: Working image width.
            work_h: Working image height.

        Returns:
            The quad, shape (4, 2), or None when the mask has no region
            covering at least MIN_AREA_RATIO.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) / (work_w * work_h) < MIN_AREA_RATIO:
            return None
        return self._quad_from_hull(cv2.convexHull(contour))

    def _quad_from_hull(self, hull: "np.ndarray") -> "np.ndarray":
        """Approximate a convex hull with a quadrilateral.

        Tries increasingly coarse polygon approximations until one has exactly
        4 points; falls back to the hull's minimum-area bounding rectangle.

        Args:
            hull: Convex hull contour from cv2.convexHull.

        Returns:
            Array of shape (4, 2), float32, unordered.
        """
        perimeter = cv2.arcLength(hull, closed=True)
        for epsilon in (0.02, 0.04, 0.06, 0.08, 0.1):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, closed=True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
        return cast("np.ndarray", cv2.boxPoints(cv2.minAreaRect(hull)))


# Singleton instance
_puzzle_detector: Optional[PuzzleFrameDetector] = None


def get_puzzle_detector() -> PuzzleFrameDetector:
    """Get or create the singleton PuzzleFrameDetector instance.

    Returns:
        The shared PuzzleFrameDetector instance.
    """
    global _puzzle_detector
    if _puzzle_detector is None:
        _puzzle_detector = PuzzleFrameDetector()
    return _puzzle_detector
