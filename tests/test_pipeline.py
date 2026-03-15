"""End-to-end tests for the Pipeline class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.models.recognized_card import CardData, RecognizedCard
from mtg_scanner.models.scan_result import ScanResult
from mtg_scanner.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patch(index: int = 0, image: np.ndarray | None = None) -> CardPatch:
    if image is None:
        image = np.zeros((280, 200, 3), dtype=np.uint8)
    return CardPatch(
        image=image,
        source_image_path="fake/image.jpg",
        bbox=(0, 0, 200, 280),
        detection_confidence=0.9,
        patch_index=index,
    )


def _make_card_data() -> CardData:
    return CardData(
        name="Lightning Bolt",
        set_code="m21",
        collector_number="167",
        rarity="common",
        type_line="Instant",
        price_eur=0.25,
        price_usd=0.30,
        scryfall_uri="https://scryfall.com/card/m21/167",
        image_uri="https://cards.scryfall.io/normal/m21/167.jpg",
    )


def _build_pipeline_with_mocks(
    patches: list[CardPatch],
    recognition_results: list[tuple],
    card_data: CardData | None,
) -> Pipeline:
    """Build a Pipeline whose internals are all mocked."""
    detector = MagicMock()
    detector.detect.return_value = patches

    primary = MagicMock()
    primary.recognize.side_effect = recognition_results

    scryfall = MagicMock()
    scryfall.lookup.return_value = card_data

    return Pipeline(
        detector=detector,
        primary_recognizer=primary,
        fallback_recognizer=None,
        scryfall_client=scryfall,
        save_patches=False,
    )


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------


class TestProcessImage:
    def test_no_patches(self):
        """When no cards are detected the ScanResult should reflect that."""
        pipeline = _build_pipeline_with_mocks([], [], None)
        result = pipeline.process_image("fake.jpg")

        assert isinstance(result, ScanResult)
        assert result.total_detected == 0
        assert result.total_recognized == 0
        assert result.total_unknown == 0
        assert result.cards == []

    def test_single_recognised_card(self):
        patches = [_make_patch(0)]
        pipeline = _build_pipeline_with_mocks(
            patches=patches,
            recognition_results=[("Lightning Bolt", 0.92)],
            card_data=_make_card_data(),
        )
        result = pipeline.process_image("fake.jpg")

        assert result.total_detected == 1
        assert result.total_recognized == 1
        assert result.total_unknown == 0
        assert result.cards[0].card_name == "Lightning Bolt"
        assert result.cards[0].card_data is not None
        assert result.cards[0].card_data.price_usd == pytest.approx(0.30)

    def test_multiple_cards_mixed(self):
        """Multiple patches with partial recognition."""
        patches = [_make_patch(0), _make_patch(1), _make_patch(2)]
        pipeline = _build_pipeline_with_mocks(
            patches=patches,
            recognition_results=[
                ("Lightning Bolt", 0.95),
                (None, 0.0),
                ("Dark Ritual", 0.80),
            ],
            card_data=_make_card_data(),
        )
        result = pipeline.process_image("fake.jpg")

        assert result.total_detected == 3
        assert result.total_recognized == 2
        assert result.total_unknown == 1

    def test_scan_result_timestamp_is_iso(self):
        """scan_timestamp must be a valid ISO 8601 string."""
        from datetime import datetime

        pipeline = _build_pipeline_with_mocks([], [], None)
        result = pipeline.process_image("fake.jpg")
        # Should not raise
        datetime.fromisoformat(result.scan_timestamp)

    def test_detection_exception_handled(self):
        """If the detector raises, process_image must return an empty ScanResult."""
        pipeline = Pipeline(
            primary_recognizer=MagicMock(),
            scryfall_client=MagicMock(),
            save_patches=False,
        )
        pipeline._detector = MagicMock()
        pipeline._detector.detect.side_effect = RuntimeError("camera error")

        result = pipeline.process_image("fake.jpg")
        assert result.total_detected == 0


# ---------------------------------------------------------------------------
# process_directory
# ---------------------------------------------------------------------------


class TestProcessDirectory:
    def test_non_existent_directory(self):
        pipeline = _build_pipeline_with_mocks([], [], None)
        results = pipeline.process_directory("/no/such/dir")
        assert results == []

    def test_empty_directory(self, tmp_path):
        pipeline = _build_pipeline_with_mocks([], [], None)
        results = pipeline.process_directory(str(tmp_path))
        assert results == []

    def test_directory_with_images(self, tmp_path):
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for name in ["a.jpg", "b.jpg"]:
            cv2.imwrite(str(tmp_path / name), img)

        pipeline = _build_pipeline_with_mocks(
            patches=[],
            recognition_results=[],
            card_data=None,
        )
        results = pipeline.process_directory(str(tmp_path))
        assert len(results) == 2

    def test_progress_callback_called(self, tmp_path):
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "img.jpg"), img)

        pipeline = _build_pipeline_with_mocks([], [], None)
        calls = []
        pipeline.process_directory(str(tmp_path), progress_callback=lambda i, t, p: calls.append((i, t)))
        assert calls == [(1, 1)]


# ---------------------------------------------------------------------------
# ScanResult serialisation
# ---------------------------------------------------------------------------


class TestScanResultSerialisation:
    def _make_result(self) -> ScanResult:
        patch = _make_patch(0)
        card_data = _make_card_data()
        rc = RecognizedCard(
            patch=patch,
            card_name="Lightning Bolt",
            recognition_confidence=0.92,
            recognition_method="ocr",
            card_data=card_data,
        )
        return ScanResult(
            image_path="fake.jpg",
            scan_timestamp="2026-01-01T00:00:00+00:00",
            cards=[rc],
            total_detected=1,
            total_recognized=1,
            total_unknown=0,
        )

    def test_to_json_structure(self):
        result = self._make_result()
        data = result.to_json()
        assert data["image_path"] == "fake.jpg"
        assert data["total_detected"] == 1
        assert len(data["cards"]) == 1
        assert data["cards"][0]["card_name"] == "Lightning Bolt"

    def test_to_json_serialisable(self):
        result = self._make_result()
        data = result.to_json()
        # Must not raise
        json.dumps(data)

    def test_to_csv_rows(self):
        result = self._make_result()
        rows = result.to_csv_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["card_name"] == "Lightning Bolt"
        assert row["set_code"] == "m21"
        assert row["recognition_method"] == "ocr"

    def test_summary_contains_card_name(self):
        result = self._make_result()
        summary = result.summary()
        assert "Lightning Bolt" in summary
        assert "ocr" in summary

    def test_save_results_json(self, tmp_path):
        result = self._make_result()
        result.image_path = "fake"
        pipeline = _build_pipeline_with_mocks([], [], None)
        written = pipeline.save_results([result], output_dir=str(tmp_path), fmt="json")
        assert len(written) == 1
        assert Path(written[0]).exists()

    def test_save_results_csv(self, tmp_path):
        result = self._make_result()
        result.image_path = "fake"
        pipeline = _build_pipeline_with_mocks([], [], None)
        written = pipeline.save_results([result], output_dir=str(tmp_path), fmt="csv")
        assert len(written) == 1
        assert Path(written[0]).exists()
