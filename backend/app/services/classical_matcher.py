"""Classical SIFT->NCC hybrid matcher for puzzle pieces.

Ported from ``network/experiments/exp25_north_star_eval/evaluate.py`` (the
north-star real-photo evaluation), which found this hybrid roughly 5x more
accurate than the served CNN on real photos. SIFT keypoint matching with
FLANN + partial-affine RANSAC runs first; when it fails to find enough
matches, a masked multi-scale normalized cross-correlation (NCC) template
match is used as a fallback.

The algorithm (constants, thresholds, matching logic) is ported verbatim from
exp25; only the return values are adapted from the evaluation's discrete grid
cell index to a continuous normalized position, to match the backend's
``PieceResponse`` contract.
"""

import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

from app.config import settings
from app.models.puzzle_model import PieceResponse, Position
from app.services.background_remover import get_background_remover
from app.services.piece_detector import crop_to_alpha_region

# Pixels whose grayscale value is above this count as piece content (the
# segmented crops have pure-black backgrounds).
MASK_THRESHOLD = 8

# NCC: the piece template is resized so its content spans cell_size * scale.
# Real pieces with tabs span ~1.1-1.5 cells; the range covers loose crops too.
NCC_SCALES = (0.9, 1.1, 1.3, 1.5)
NCC_OVERVIEW_SIDE = 256  # match the synthetic benchmark's puzzle resolution

# SIFT (exp23 settings; FLANN instead of brute force for the larger images)
LOWE_RATIO = 0.75
RANSAC_REPROJ_THRESHOLD = 5.0
MIN_GOOD_MATCHES = 4
MIN_INLIERS = 3
SIFT_OVERVIEW_SIDE = 768
SIFT_PIECE_SIDE = 384


def _resize_max_side(rgb: "np.ndarray", side: int) -> "np.ndarray":
    """Resize an image so its longer side equals ``side`` (aspect preserved).

    Never upscales: images already smaller than ``side`` are returned as-is.

    Args:
        rgb: Image array.
        side: Target size of the longer side.

    Returns:
        Resized array (unchanged when already smaller).
    """
    h, w = rgb.shape[:2]
    scale = side / max(h, w)
    if scale >= 1.0:
        return rgb
    return cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


@dataclass
class PuzzleFeatures:
    """Cached per-puzzle features for classical matching."""

    sift_keypoints: Any  # tuple[cv2.KeyPoint, ...]; cv2 stubs don't expose a precise type
    sift_descriptors: Optional["np.ndarray"]
    sift_shape: tuple[int, int]  # (height, width) of the SIFT overview used for detection
    ncc_overview_bgr: "np.ndarray"


class ClassicalMatcher:
    """SIFT->NCC hybrid puzzle-piece matcher (exp25 recipe)."""

    def __init__(self) -> None:
        """Create the SIFT detector, FLANN matcher, and the per-puzzle feature cache."""
        self._detector = cv2.SIFT_create()  # type: ignore[attr-defined]
        self._flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
        self._puzzle_cache: dict[str, PuzzleFeatures] = {}

    def _load_puzzle_features(self, puzzle_id: str) -> PuzzleFeatures:
        """Load and cache SIFT/NCC features for a puzzle overview image.

        Args:
            puzzle_id: The ID of the puzzle to load.

        Returns:
            The cached (or newly computed) PuzzleFeatures.

        Raises:
            FileNotFoundError: If the puzzle image doesn't exist.
        """
        if puzzle_id not in self._puzzle_cache:
            try:
                puzzle_uuid = UUID(puzzle_id)
                if str(puzzle_uuid) != puzzle_id:
                    raise ValueError
            except ValueError as exc:
                raise FileNotFoundError(f"Puzzle image not found: {puzzle_id}") from exc

            # Build the filename from the parsed UUID rather than the raw request
            # string, so the path component can only ever be a canonical UUID.
            puzzle_path = os.path.join(settings.UPLOAD_DIR, f"{puzzle_uuid}.jpg")
            # Normalize and ensure the puzzle path stays within the upload directory
            base_dir = os.path.realpath(settings.UPLOAD_DIR)
            normalized_path = os.path.realpath(puzzle_path)
            # Ensure the resolved puzzle path is contained within the upload directory
            if os.path.commonpath([base_dir, normalized_path]) != base_dir:
                raise FileNotFoundError(f"Puzzle image not found: {normalized_path}")
            if not os.path.exists(normalized_path):
                raise FileNotFoundError(f"Puzzle image not found: {normalized_path}")

            # PIL handles the case where the upload isn't actually a JPEG despite the extension.
            puzzle_img = Image.open(normalized_path).convert("RGB")
            rgb = np.array(puzzle_img)

            sift_rgb = _resize_max_side(rgb, SIFT_OVERVIEW_SIDE)
            sift_gray = cv2.cvtColor(sift_rgb, cv2.COLOR_RGB2GRAY)
            sift_kp, sift_des = self._detector.detectAndCompute(sift_gray, None)

            ncc_overview_bgr = cv2.cvtColor(_resize_max_side(rgb, NCC_OVERVIEW_SIDE), cv2.COLOR_RGB2BGR)

            self._puzzle_cache[puzzle_id] = PuzzleFeatures(
                sift_keypoints=sift_kp,
                sift_descriptors=sift_des,
                sift_shape=(sift_gray.shape[0], sift_gray.shape[1]),
                ncc_overview_bgr=ncc_overview_bgr,
            )

        return self._puzzle_cache[puzzle_id]

    def clear_puzzle_cache(self, puzzle_id: Optional[str] = None) -> None:
        """Clear cached puzzle features.

        Args:
            puzzle_id: If provided, only clear this puzzle. Otherwise clear all.
        """
        if puzzle_id is not None:
            self._puzzle_cache.pop(puzzle_id, None)
        else:
            self._puzzle_cache.clear()

    def _predict_sift(
        self, piece_rgb: "np.ndarray", features: PuzzleFeatures
    ) -> Optional[tuple[float, float, int, float]]:
        """Predict (position, rotation, confidence) via SIFT + FLANN + partial-affine RANSAC.

        Args:
            piece_rgb: Observed piece image, RGB uint8, black background.
            features: Cached puzzle features.

        Returns:
            (nx, ny, rotation_degrees, confidence), or None when matching fails.
        """
        piece_resized = _resize_max_side(piece_rgb, SIFT_PIECE_SIDE)
        gray = cv2.cvtColor(piece_resized, cv2.COLOR_RGB2GRAY)
        mask = (gray > MASK_THRESHOLD).astype(np.uint8) * 255
        piece_kp, piece_des = self._detector.detectAndCompute(gray, mask)
        puzzle_des = features.sift_descriptors
        if piece_des is None or puzzle_des is None or len(piece_kp) < MIN_GOOD_MATCHES:
            return None
        pairs = self._flann.knnMatch(piece_des, puzzle_des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_GOOD_MATCHES:
            return None
        puzzle_kp = features.sift_keypoints
        src = np.array([piece_kp[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
        dst = np.array([puzzle_kp[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD
        )
        if transform is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
            return None
        theta = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
        rotation_idx = int(round(-theta / 90)) % 4
        inlier_flags = inliers.ravel().astype(bool)
        centroid = dst.reshape(-1, 2)[inlier_flags].mean(axis=0)
        puz_h, puz_w = features.sift_shape
        nx = float(centroid[0]) / puz_w
        ny = float(centroid[1]) / puz_h
        confidence = min(1.0, int(inliers.sum()) / 30.0)
        return nx, ny, rotation_idx * 90, confidence

    def _predict_ncc(
        self, piece_rgb: "np.ndarray", features: PuzzleFeatures
    ) -> Optional[tuple[float, float, int, float]]:
        """Predict (position, rotation, confidence) via masked multi-scale NCC.

        Tries all 4 candidate un-rotations of the piece x each of NCC_SCALES,
        matched against the NCC overview with a content mask.

        Args:
            piece_rgb: Observed piece image, RGB uint8, black background.
            features: Cached puzzle features.

        Returns:
            (nx, ny, rotation_degrees, confidence), or None when no scale/rotation
            candidate produced a usable template (e.g. an all-black piece).
        """
        puzzle_bgr = features.ncc_overview_bgr
        puz_h, puz_w = puzzle_bgr.shape[:2]
        nominal = max(puz_w / settings.CLASSICAL_GRID_COLS, puz_h / settings.CLASSICAL_GRID_ROWS)
        best_score = -np.inf
        best: Optional[tuple[float, float, int]] = None
        for candidate_idx in range(4):
            unrot = np.rot90(piece_rgb, k=candidate_idx)
            for scale in NCC_SCALES:
                side = int(round(nominal * scale))
                if side < 8 or side > min(puz_h, puz_w):
                    continue
                template = cv2.cvtColor(
                    cv2.resize(unrot, (side, side), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR
                )
                mask = (cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) > MASK_THRESHOLD).astype(np.uint8) * 255
                response = cv2.matchTemplate(puzzle_bgr, template, cv2.TM_CCOEFF_NORMED, mask=mask)
                response = np.nan_to_num(response, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
                _, max_val, _, max_loc = cv2.minMaxLoc(response)
                if max_val > best_score:
                    best_score = max_val
                    nx = (max_loc[0] + side / 2) / puz_w
                    ny = (max_loc[1] + side / 2) / puz_h
                    best = (nx, ny, candidate_idx)
        if best is None:
            return None
        nx, ny, candidate_idx = best
        confidence = max(0.0, min(1.0, float(best_score)))
        return nx, ny, candidate_idx * 90, confidence

    async def process_piece(
        self,
        piece_file: UploadFile,
        puzzle_id: str,
        remove_background: bool = True,
    ) -> PieceResponse:
        """Process a puzzle piece image and predict its position and rotation.

        Runs SIFT first; when it fails to find a confident match, falls back to
        masked multi-scale NCC template matching. When both fail (or an
        unexpected error occurs), returns a neutral fallback response.

        Args:
            piece_file: The puzzle piece image file.
            puzzle_id: The ID of the puzzle to match against.
            remove_background: Whether to remove background from piece image.

        Returns:
            PieceResponse with predicted position, confidence, rotation, and optionally cleaned image.
        """
        try:
            contents = await piece_file.read()

            cleaned_image_b64: Optional[str] = None
            if remove_background and settings.ENABLE_BACKGROUND_REMOVAL:
                remover = get_background_remover(settings.REMBG_MODEL)

                # Get RGBA image with transparent background for frontend display
                rgba_image = remover.remove_background(contents)

                # Crop to the segmented subject so the piece fills the frame,
                # matching the deployed preview path.
                rgba_image = crop_to_alpha_region(rgba_image)

                # Encode as base64 PNG for frontend
                buffer = io.BytesIO()
                rgba_image.save(buffer, format="PNG")
                cleaned_image_b64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

                # Composite onto BLACK (not white) for matching: the exp25 masking
                # convention treats gray > MASK_THRESHOLD as piece content.
                piece_img = Image.new("RGB", rgba_image.size, (0, 0, 0))
                if rgba_image.mode == "RGBA":
                    piece_img.paste(rgba_image, mask=rgba_image.split()[3])
                else:
                    piece_img.paste(rgba_image)
            else:
                piece_img = Image.open(io.BytesIO(contents)).convert("RGB")

            piece_rgb = np.array(piece_img)

            # Load puzzle features (raises FileNotFoundError, caught below and re-raised)
            features = self._load_puzzle_features(puzzle_id)

            sift_result = self._predict_sift(piece_rgb, features)
            result = sift_result if sift_result is not None else self._predict_ncc(piece_rgb, features)

            if result is not None:
                nx, ny, rotation_degrees, confidence = result
                return PieceResponse(
                    position=Position(x=nx, y=ny),
                    position_confidence=confidence,
                    rotation=rotation_degrees,
                    rotation_confidence=confidence,
                    cleaned_image=cleaned_image_b64,
                )

            # Both SIFT and NCC failed to find a usable match.
            return PieceResponse(
                position=Position(x=0.5, y=0.5),
                position_confidence=0.0,
                rotation=0,
                rotation_confidence=0.0,
                cleaned_image=cleaned_image_b64,
            )

        except FileNotFoundError:
            # Re-raise file not found errors
            raise
        except Exception as e:  # noqa: BLE001 - degrade gracefully, mirrors ImageProcessor
            # Log error and return fallback response
            print(f"Error processing image (classical matcher): {str(e)}")
            return PieceResponse(
                position=Position(x=0.5, y=0.5),
                position_confidence=0.0,
                rotation=0,
                rotation_confidence=0.0,
                cleaned_image=None,
            )


# Singleton instance
_matcher: Optional[ClassicalMatcher] = None


def get_classical_matcher() -> ClassicalMatcher:
    """Get the singleton ClassicalMatcher instance.

    Returns:
        The shared ClassicalMatcher instance.
    """
    global _matcher
    if _matcher is None:
        _matcher = ClassicalMatcher()
    return _matcher
