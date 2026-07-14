"""Tests for the piece detector service."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.services.piece_detector import (
    DetectedRegion,
    PieceDetector,
    _band_score,
    crop_to_alpha_region,
    get_piece_detector,
    largest_alpha_component,
    skin_fraction,
)

# Opaque region drawn into test RGBA images: left, top, right, bottom.
# Sized like a hand-held piece: ~10% of a 320x240 frame, mild aspect ratio.
REGION = (60, 40, 160, 120)

# A blue-ish color far outside the YCrCb skin-tone box
PIECE_COLOR = (80, 120, 200)

# A color squarely inside the YCrCb skin-tone box
SKIN_COLOR = (200, 140, 110)


def make_rgba_with_region(
    size: tuple[int, int] = (320, 240),
    region: tuple[int, int, int, int] = REGION,
    color: tuple[int, int, int] = PIECE_COLOR,
) -> Image.Image:
    """Create a transparent RGBA image with one opaque rectangle."""
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    left, top, right, bottom = region
    rgba[top:bottom, left:right] = (*color, 255)
    return Image.fromarray(rgba, mode="RGBA")


class TestCropToAlphaRegion:
    """Tests for crop_to_alpha_region."""

    def test_crops_to_opaque_region_with_margin(self) -> None:
        """The crop covers the opaque rectangle plus the requested margin."""
        left, top, right, bottom = REGION

        cropped = crop_to_alpha_region(make_rgba_with_region(), margin=0.1)

        expected_w = round((right - left) * 1.2)
        expected_h = round((bottom - top) * 1.2)
        assert abs(cropped.width - expected_w) <= 2
        assert abs(cropped.height - expected_h) <= 2

    def test_fully_transparent_image_is_unchanged(self) -> None:
        """An image with no opaque pixels is returned as-is."""
        rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))

        assert crop_to_alpha_region(rgba) is rgba

    def test_non_rgba_image_is_unchanged(self) -> None:
        """A non-RGBA image is returned as-is."""
        rgb = Image.new("RGB", (100, 100), (10, 20, 30))

        assert crop_to_alpha_region(rgb) is rgb

    def test_crop_is_clamped_to_image_bounds(self) -> None:
        """A region touching the edge does not produce an out-of-bounds crop."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[0:100, 0:100] = (10, 10, 10, 255)

        cropped = crop_to_alpha_region(Image.fromarray(rgba, mode="RGBA"), margin=0.2)

        assert cropped.size == (100, 100)


def test_largest_alpha_component_picks_biggest_blob() -> None:
    """With multiple opaque blobs, the largest one is selected."""
    import cv2

    rgba = np.zeros((200, 200, 4), dtype=np.uint8)
    rgba[10:30, 10:30, 3] = 255  # small blob
    rgba[80:180, 60:190, 3] = 255  # large blob

    contour = largest_alpha_component(Image.fromarray(rgba, mode="RGBA"))

    assert contour is not None
    x, y, w, h = cv2.boundingRect(contour)
    assert (x, y) == (60, 80)
    assert (w, h) == (130, 100)


class TestSkinFraction:
    """Tests for skin_fraction."""

    def test_skin_colored_region_scores_high(self) -> None:
        """A region filled with a skin tone is measured as almost all skin."""
        rgba = make_rgba_with_region(color=SKIN_COLOR)

        assert skin_fraction(rgba) > 0.95

    def test_non_skin_region_scores_low(self) -> None:
        """A blue region contains no skin-tone pixels."""
        rgba = make_rgba_with_region(color=PIECE_COLOR)

        assert skin_fraction(rgba) < 0.05

    def test_fully_transparent_image_scores_zero(self) -> None:
        """An image with no opaque pixels yields 0.0 rather than dividing by zero."""
        rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))

        assert skin_fraction(rgba) == 0.0

    def test_contour_restricts_measurement(self) -> None:
        """With a contour, only pixels inside that region are measured."""
        rgba = np.zeros((240, 320, 4), dtype=np.uint8)
        rgba[40:120, 60:160] = (*PIECE_COLOR, 255)  # blue blob (the contour target)
        rgba[150:230, 60:160] = (*SKIN_COLOR, 255)  # separate skin-colored blob
        image = Image.fromarray(rgba, mode="RGBA")
        contour = largest_alpha_component(Image.fromarray(rgba[:130], mode="RGBA"))
        assert contour is not None

        assert skin_fraction(image, contour) < 0.05
        assert skin_fraction(image) > 0.4


class TestBandScore:
    """Tests for the _band_score confidence helper."""

    def test_full_confidence_inside_band(self) -> None:
        """Values in the full-confidence band score 1.0."""
        assert _band_score(0.5, 0.0, 0.2, 0.8, 1.0) == 1.0

    def test_zero_outside_hard_limits(self) -> None:
        """Values at or beyond the hard limits score 0.0."""
        assert _band_score(0.0, 0.0, 0.2, 0.8, 1.0) == 0.0
        assert _band_score(1.5, 0.0, 0.2, 0.8, 1.0) == 0.0

    def test_linear_taper_between_band_and_hard_limit(self) -> None:
        """Values between the band and a hard limit taper linearly."""
        assert _band_score(0.1, 0.0, 0.2, 0.8, 1.0) == pytest.approx(0.5)
        assert _band_score(0.9, 0.0, 0.2, 0.8, 1.0) == pytest.approx(0.5)


def make_classifier(available: bool = False, score: Optional[float] = None) -> MagicMock:
    """Create a mock piece classifier service.

    Args:
        available: Whether the mock reports a loaded model.
        score: The probability returned by score().

    Returns:
        A MagicMock standing in for PieceClassifierService.
    """
    classifier = MagicMock()
    classifier.available = available
    classifier.score.return_value = score
    return classifier


class TestPieceDetector:
    """Tests for PieceDetector.detect_region with a mocked background remover.

    The piece classifier is pinned to "unavailable" so these tests exercise
    the heuristic fallback regardless of whether a trained checkpoint exists
    on the machine running them.
    """

    @pytest.fixture
    def frame(self) -> Image.Image:
        """A camera frame; content is irrelevant since rembg is mocked."""
        return Image.new("RGB", (640, 480), (50, 60, 70))

    def detect(
        self,
        frame: Image.Image,
        rgba: Image.Image,
        classifier: Optional[MagicMock] = None,
    ) -> Optional[DetectedRegion]:
        """Run detect_region with the background remover mocked to return rgba."""
        remover = MagicMock()
        remover.remove_background.return_value = rgba
        if classifier is None:
            classifier = make_classifier(available=False)

        with (
            patch("app.services.piece_detector.get_background_remover", return_value=remover),
            patch("app.services.piece_detector.get_piece_classifier", return_value=classifier),
        ):
            return PieceDetector().detect_region(frame)

    def test_detects_region_from_rembg_alpha(self, frame: Image.Image) -> None:
        """The polygon and bbox reflect the mocked rembg alpha mask."""
        result = self.detect(frame, make_rgba_with_region())

        assert result is not None
        left, top, right, bottom = REGION
        assert result.bbox[0] == pytest.approx(left / 320, abs=0.02)
        assert result.bbox[1] == pytest.approx(top / 240, abs=0.02)
        assert result.bbox[2] == pytest.approx((right - left) / 320, abs=0.02)
        assert result.bbox[3] == pytest.approx((bottom - top) / 240, abs=0.02)
        assert len(result.polygon) >= 4
        for x, y in result.polygon:
            assert 0.0 <= x <= 1.0
            assert 0.0 <= y <= 1.0

    def test_piece_like_region_has_full_confidence(self, frame: Image.Image) -> None:
        """A piece-sized, compact, non-skin region gets confidence 1.0."""
        result = self.detect(frame, make_rgba_with_region())

        assert result is not None
        assert result.confidence == 1.0

    def test_no_alpha_returns_none(self, frame: Image.Image) -> None:
        """A fully transparent rembg result yields no detection."""
        assert self.detect(frame, Image.new("RGBA", (320, 240), (0, 0, 0, 0))) is None

    def test_tiny_region_returns_none(self, frame: Image.Image) -> None:
        """A region below the minimum area ratio is rejected."""
        rgba = make_rgba_with_region(region=(100, 100, 104, 104))

        assert self.detect(frame, rgba) is None

    def test_face_sized_skin_region_returns_none(self, frame: Image.Image) -> None:
        """A large skin-toned region (a face at webcam distance) is rejected."""
        rgba = make_rgba_with_region(region=(60, 40, 260, 200), color=SKIN_COLOR)

        assert self.detect(frame, rgba) is None

    def test_oversized_region_returns_none(self, frame: Image.Image) -> None:
        """A region covering most of the frame is rejected regardless of color."""
        rgba = make_rgba_with_region(region=(10, 10, 310, 230))

        assert self.detect(frame, rgba) is None

    def test_elongated_region_returns_none(self, frame: Image.Image) -> None:
        """A long thin region (arm, table edge) is rejected by the aspect gate."""
        rgba = make_rgba_with_region(region=(20, 100, 300, 112))

        assert self.detect(frame, rgba) is None

    def test_piece_sized_skin_region_returns_none(self, frame: Image.Image) -> None:
        """A skin-dominant region is rejected even when its size is piece-like."""
        rgba = make_rgba_with_region(color=SKIN_COLOR)

        assert self.detect(frame, rgba) is None

    def test_borderline_area_lowers_confidence(self, frame: Image.Image) -> None:
        """A region larger than the full-confidence band gets reduced confidence."""
        # 160x120 of 320x240 = 25% of the frame: between the 15% full-confidence
        # edge and the 35% hard limit, so the area score tapers to roughly 0.5
        rgba = make_rgba_with_region(region=(80, 60, 240, 180))

        result = self.detect(frame, rgba)

        assert result is not None
        assert 0.3 < result.confidence < 0.7

    def test_rembg_failure_returns_none(self, frame: Image.Image) -> None:
        """A rembg exception degrades to no detection instead of raising."""
        remover = MagicMock()
        remover.remove_background.side_effect = RuntimeError("boom")

        with patch("app.services.piece_detector.get_background_remover", return_value=remover):
            assert PieceDetector().detect_region(frame) is None


class TestPieceDetectorWithClassifier:
    """Tests for detect_region when a trained piece classifier is available."""

    @pytest.fixture
    def frame(self) -> Image.Image:
        """A camera frame; content is irrelevant since rembg is mocked."""
        return Image.new("RGB", (640, 480), (50, 60, 70))

    def detect(
        self,
        frame: Image.Image,
        rgba: Image.Image,
        classifier: MagicMock,
    ) -> Optional[DetectedRegion]:
        """Run detect_region with mocked background remover and classifier."""
        remover = MagicMock()
        remover.remove_background.return_value = rgba

        with (
            patch("app.services.piece_detector.get_background_remover", return_value=remover),
            patch("app.services.piece_detector.get_piece_classifier", return_value=classifier),
        ):
            return PieceDetector().detect_region(frame)

    def test_classifier_probability_becomes_confidence(self, frame: Image.Image) -> None:
        """The classifier probability replaces the heuristic confidence."""
        result = self.detect(frame, make_rgba_with_region(), make_classifier(available=True, score=0.87))

        assert result is not None
        assert result.confidence == pytest.approx(0.87)

    def test_classifier_receives_the_rembg_output(self, frame: Image.Image) -> None:
        """The classifier scores the segmented RGBA frame."""
        rgba = make_rgba_with_region()
        classifier = make_classifier(available=True, score=0.9)

        self.detect(frame, rgba, classifier)

        classifier.score.assert_called_once_with(rgba)

    def test_skin_gate_is_skipped_when_classifier_scores(self, frame: Image.Image) -> None:
        """A skin-toned region the heuristic would reject passes with a high score."""
        rgba = make_rgba_with_region(color=SKIN_COLOR)

        result = self.detect(frame, rgba, make_classifier(available=True, score=0.95))

        assert result is not None
        assert result.confidence == pytest.approx(0.95)

    def test_low_probability_region_is_still_reported(self, frame: Image.Image) -> None:
        """A low classifier score yields found=True with low confidence (frontend gates)."""
        result = self.detect(frame, make_rgba_with_region(), make_classifier(available=True, score=0.02))

        assert result is not None
        assert result.confidence == pytest.approx(0.02)

    def test_area_gate_still_rejects_before_classifier(self, frame: Image.Image) -> None:
        """The cheap area pre-filter rejects oversized regions regardless of score."""
        rgba = make_rgba_with_region(region=(10, 10, 310, 230))

        assert self.detect(frame, rgba, make_classifier(available=True, score=0.99)) is None

    def test_classifier_failure_falls_back_to_heuristic(self, frame: Image.Image) -> None:
        """score() returning None degrades to the heuristic confidence."""
        result = self.detect(frame, make_rgba_with_region(), make_classifier(available=True, score=None))

        assert result is not None
        assert result.confidence == 1.0

    def test_zero_probability_reports_floor_confidence(self, frame: Image.Image) -> None:
        """A 0.0 score is floored so found=True always carries confidence > 0."""
        result = self.detect(frame, make_rgba_with_region(), make_classifier(available=True, score=0.0))

        assert result is not None
        assert result.confidence == 0.001


class TestSavePreviewCrops:
    """Tests for the SAVE_PREVIEW_CROPS hard-negative harvesting setting."""

    def detect_with_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        save_crops: bool,
    ) -> None:
        """Run one detection with SAVE_PREVIEW_CROPS configured."""
        from app.services import piece_detector as module

        monkeypatch.setattr(module.settings, "SAVE_PREVIEW_CROPS", save_crops)
        monkeypatch.setattr(module.settings, "UPLOAD_DIR", str(tmp_path))

        remover = MagicMock()
        remover.remove_background.return_value = make_rgba_with_region()
        with (
            patch("app.services.piece_detector.get_background_remover", return_value=remover),
            patch("app.services.piece_detector.get_piece_classifier", return_value=make_classifier()),
        ):
            result = PieceDetector().detect_region(Image.new("RGB", (640, 480), (50, 60, 70)))
        assert result is not None

    def test_accepted_crop_is_saved_when_enabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the setting on, an accepted region's crop lands on disk."""
        self.detect_with_settings(tmp_path, monkeypatch, save_crops=True)

        crops = list(tmp_path.glob("preview_crops/*.png"))
        assert len(crops) == 1
        saved = Image.open(crops[0])
        assert saved.width == saved.height  # classifier-input protocol: square crop

    def test_nothing_is_saved_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the setting off, no crops directory appears."""
        self.detect_with_settings(tmp_path, monkeypatch, save_crops=False)

        assert not (tmp_path / "preview_crops").exists()


def test_get_piece_detector_is_singleton() -> None:
    """get_piece_detector returns the same instance on repeated calls."""
    assert get_piece_detector() is get_piece_detector()
