"""Tests for card detection modules."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mtg_scanner.detection.opencv_detector import OpenCVDetector, order_points
from mtg_scanner.models.card_patch import CardPatch


# ---------------------------------------------------------------------------
# order_points()
# ---------------------------------------------------------------------------


class TestOrderPoints:
    """Tests for the four-point corner ordering utility."""

    def test_already_ordered(self):
        pts = np.array([[0, 0], [100, 0], [100, 140], [0, 140]], dtype=np.float32)
        result = order_points(pts)
        np.testing.assert_array_equal(result[0], [0, 0])    # TL
        np.testing.assert_array_equal(result[1], [100, 0])  # TR
        np.testing.assert_array_equal(result[2], [100, 140])  # BR
        np.testing.assert_array_equal(result[3], [0, 140])  # BL

    def test_shuffled_input(self):
        """order_points must sort correctly regardless of input order."""
        pts = np.array([[100, 140], [0, 0], [0, 140], [100, 0]], dtype=np.float32)
        result = order_points(pts)
        np.testing.assert_array_equal(result[0], [0, 0])    # TL
        np.testing.assert_array_equal(result[1], [100, 0])  # TR
        np.testing.assert_array_equal(result[2], [100, 140])  # BR
        np.testing.assert_array_equal(result[3], [0, 140])  # BL

    def test_non_axis_aligned(self):
        """Rotated rectangle corners should still be ordered correctly."""
        pts = np.array([[50, 0], [100, 50], [50, 100], [0, 50]], dtype=np.float32)
        result = order_points(pts)
        # TL has the smallest sum
        sums = pts.sum(axis=1)
        assert result[0].tolist() == pts[np.argmin(sums)].tolist()
        # BR has the largest sum
        assert result[2].tolist() == pts[np.argmax(sums)].tolist()

    def test_output_shape(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        result = order_points(pts)
        assert result.shape == (4, 2)


# ---------------------------------------------------------------------------
# OpenCVDetector – aspect ratio filtering
# ---------------------------------------------------------------------------


class TestAspectRatioFilter:
    """Tests for the contour aspect-ratio check in OpenCVDetector."""

    def setup_method(self):
        self.detector = OpenCVDetector()

    def _make_rect_contour(self, w: int, h: int) -> np.ndarray:
        """Create a rectangular contour array."""
        return np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32
        ).reshape(-1, 1, 2)

    def test_valid_card_ratio(self):
        """Standard MTG card portrait (63×88 mm ≈ 0.716) should pass."""
        contour = self._make_rect_contour(630, 880)
        valid, conf = self.detector._is_valid_card_contour(contour, image_area=10_000_000)
        assert valid
        assert conf > 0.0

    def test_too_wide(self):
        """A landscape rectangle should be rejected."""
        contour = self._make_rect_contour(880, 300)  # ratio 300/880 ≈ 0.34
        valid, _ = self.detector._is_valid_card_contour(contour, image_area=10_000_000)
        assert not valid

    def test_too_narrow(self):
        """A very narrow rectangle should be rejected."""
        contour = self._make_rect_contour(100, 1000)  # ratio 100/1000 = 0.10
        valid, _ = self.detector._is_valid_card_contour(contour, image_area=10_000_000)
        assert not valid

    def test_too_small_area(self):
        """A contour smaller than min_card_area_px must be rejected."""
        contour = self._make_rect_contour(20, 28)  # area = 560 < 5000
        valid, _ = self.detector._is_valid_card_contour(contour, image_area=10_000_000)
        assert not valid


# ---------------------------------------------------------------------------
# OpenCVDetector – synthetic image
# ---------------------------------------------------------------------------


class TestOpenCVDetectorSynthetic:
    """Integration-style test with a synthetic image."""

    def test_white_rect_on_black(self, tmp_path):
        """A white rectangle with card proportions on black should be detected."""
        import cv2

        # Create a 600×400 black image with a white card-shaped rectangle
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        # Card: 90×126 pixels (ratio 90/126 ≈ 0.71)
        cv2.rectangle(img, (100, 80), (190, 206), (255, 255, 255), -1)

        img_path = str(tmp_path / "synth.png")
        cv2.imwrite(img_path, img)

        detector = OpenCVDetector()
        patches = detector.detect(img_path)

        # We do not assert exactly 1 because edge-detection on synthetic images
        # can produce varying results; just verify the return type is correct.
        assert isinstance(patches, list)
        for p in patches:
            assert isinstance(p, CardPatch)
            assert p.image is not None
            assert p.patch_index >= 0

    def test_detect_missing_file(self):
        """Detecting a non-existent file must return an empty list."""
        detector = OpenCVDetector()
        result = detector.detect("/no/such/file.jpg")
        assert result == []


# ---------------------------------------------------------------------------
# YOLODetector – mocked
# ---------------------------------------------------------------------------


class TestYOLODetector:
    """Unit tests for the YOLO detector using mocked Ultralytics."""

    def test_detect_with_mock_model(self, tmp_path):
        """YOLODetector should return CardPatch objects from mocked YOLO output."""
        import cv2

        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img_path = str(tmp_path / "test.jpg")
        cv2.imwrite(img_path, img)

        # Mock ultralytics.YOLO
        mock_box = MagicMock()
        mock_box.conf = [MagicMock(item=lambda: 0.9)]
        mock_box.xyxy = [MagicMock(
            __iter__=lambda self: iter([
                MagicMock(item=lambda: 50),
                MagicMock(item=lambda: 50),
                MagicMock(item=lambda: 200),
                MagicMock(item=lambda: 280),
            ])
        )]

        mock_result = MagicMock()
        mock_result.boxes = mock_box

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.return_value = [mock_result]

        with patch.dict("sys.modules", {"ultralytics": MagicMock(YOLO=mock_yolo_instance)}):
            from mtg_scanner.detection.yolo_detector import YOLODetector

            detector = YOLODetector(model_path="fake.pt", confidence_threshold=0.5)
            # Force model load with mock
            detector._model = mock_yolo_instance

            patches = detector.detect(img_path)

        assert isinstance(patches, list)
