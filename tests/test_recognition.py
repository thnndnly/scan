"""Tests for card recognition modules."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mtg_scanner.models.card_patch import CardPatch
from mtg_scanner.utils.fuzzy_search import best_match, fuzzy_match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def names_file(tmp_path) -> str:
    """Write a small card-names JSON file and return its path."""
    names = [
        "Lightning Bolt",
        "Counterspell",
        "Dark Ritual",
        "Serra Angel",
        "Llanowar Elves",
        "Black Lotus",
        "Ancestral Recall",
    ]
    p = tmp_path / "card_names.json"
    p.write_text(json.dumps(names), encoding="utf-8")
    return str(p)


@pytest.fixture
def card_patch(sample_card_image) -> CardPatch:
    return CardPatch(
        image=sample_card_image,
        source_image_path="fake/path.jpg",
        bbox=(0, 0, 200, 280),
        detection_confidence=0.85,
        patch_index=0,
    )


# ---------------------------------------------------------------------------
# Title region extraction
# ---------------------------------------------------------------------------


class TestTitleRegionExtraction:
    def test_top_13_percent(self):
        """OCRRecognizer should crop the top 13 % of the card image.

        Uses a 700×500 image so the title strip (91 px) is tall enough that
        the MIN_TITLE_HEIGHT_PX (80 px) upscaling logic does NOT trigger.
        """
        from mtg_scanner.recognition.ocr_recognizer import OCRRecognizer, _TITLE_STRIP_HEIGHT_FRAC

        img = np.zeros((700, 500, 3), dtype=np.uint8)
        rec = OCRRecognizer()
        region = rec._extract_title_region(img)
        expected_h = int(img.shape[0] * _TITLE_STRIP_HEIGHT_FRAC)
        from mtg_scanner.recognition.ocr_recognizer import _TITLE_STRIP_X_END_FRAC
        assert region.shape[0] == expected_h
        assert region.shape[1] == int(img.shape[1] * _TITLE_STRIP_X_END_FRAC)

    def test_title_region_not_empty(self, sample_card_image):
        from mtg_scanner.recognition.ocr_recognizer import OCRRecognizer

        rec = OCRRecognizer()
        region = rec._extract_title_region(sample_card_image)
        assert region.size > 0


# ---------------------------------------------------------------------------
# Fuzzy search
# ---------------------------------------------------------------------------


class TestFuzzySearch:
    def test_exact_match(self, names_file):
        name, conf = best_match("Lightning Bolt", [], names_file=names_file)
        assert name == "Lightning Bolt"
        assert conf > 0.95

    def test_typo_match(self, names_file):
        """Small typos should still yield the correct card name."""
        name, conf = best_match("Lightnng Bolt", [], names_file=names_file)
        assert name == "Lightning Bolt"
        assert conf > 0.7

    def test_no_match_below_cutoff(self, names_file):
        name, conf = best_match("xyzzy_no_card", [], score_cutoff=90.0, names_file=names_file)
        assert name is None
        assert conf == 0.0

    def test_fuzzy_match_returns_list(self, names_file):
        from mtg_scanner.utils.fuzzy_search import load_card_names

        names = load_card_names(names_file)
        results = fuzzy_match("Counterspell", names, score_cutoff=80.0, limit=3)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0][0] == "Counterspell"


# ---------------------------------------------------------------------------
# Hash recognizer – determinism
# ---------------------------------------------------------------------------


class TestHashRecognizer:
    def test_phash_determinism(self, sample_card_image):
        """The same image should always produce the same pHash."""
        try:
            import imagehash  # noqa: F401
        except ImportError:
            pytest.skip("imagehash not installed")

        from mtg_scanner.recognition.hash_recognizer import _compute_phash
        from mtg_scanner.utils.image_utils import crop_region

        artwork = crop_region(sample_card_image, 0.20, 0.65, x_margin_px=5)
        h1 = _compute_phash(artwork)
        h2 = _compute_phash(artwork)
        assert str(h1) == str(h2)

    def test_hash_recognizer_no_db(self, card_patch, tmp_path):
        """HashRecognizer with a non-existent DB should return (None, 0.0)."""
        try:
            import imagehash  # noqa: F401
        except ImportError:
            pytest.skip("imagehash not installed")

        from mtg_scanner.recognition.hash_recognizer import HashRecognizer

        rec = HashRecognizer(db_path=str(tmp_path / "nonexistent.db"))
        name, conf = rec.recognize(card_patch)
        assert name is None
        assert conf == 0.0

    def test_hash_recognizer_with_db(self, card_patch, sample_card_image, tmp_path):
        """HashRecognizer should find a card when an exact hash is in the DB."""
        try:
            import imagehash  # noqa: F401
        except ImportError:
            pytest.skip("imagehash not installed")

        from mtg_scanner.recognition.hash_recognizer import (
            HashRecognizer,
            _compute_phash,
        )
        from mtg_scanner.utils.image_utils import crop_region

        # Build a minimal DB with the same image
        db_path = tmp_path / "card_hashes.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE card_hashes "
            "(id INTEGER PRIMARY KEY, card_name TEXT, set_code TEXT, hash_value TEXT, image_uri TEXT);"
        )
        artwork = crop_region(sample_card_image, 0.20, 0.65, x_margin_px=5)
        hash_val = str(_compute_phash(artwork))
        conn.execute(
            "INSERT INTO card_hashes (card_name, set_code, hash_value, image_uri) VALUES (?, ?, ?, ?);",
            ("Lightning Bolt", "m21", hash_val, ""),
        )
        conn.commit()
        conn.close()

        rec = HashRecognizer(db_path=str(db_path), max_hamming_distance=12)
        name, conf = rec.recognize(card_patch)
        assert name == "Lightning Bolt"
        assert conf == 1.0  # distance 0 → confidence 1.0


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_fallback_invoked_when_primary_fails(self, card_patch):
        """Pipeline should call the fallback recogniser when primary returns None."""
        from mtg_scanner.pipeline import Pipeline

        primary = MagicMock()
        primary.recognize.return_value = (None, 0.0)

        fallback = MagicMock()
        fallback.recognize.return_value = ("Dark Ritual", 0.85)

        scryfall = MagicMock()
        scryfall.lookup.return_value = None

        pipeline = Pipeline(
            primary_recognizer=primary,
            fallback_recognizer=fallback,
            scryfall_client=scryfall,
        )

        # Bypass detection
        with patch.object(pipeline._detector, "detect", return_value=[card_patch]):
            result = pipeline.process_image("fake.jpg")

        assert result.cards[0].card_name == "Dark Ritual"
        assert result.cards[0].recognition_method == "hash"

    def test_unknown_when_all_fail(self, card_patch):
        """When all recognisers fail, method should be 'unknown'."""
        from mtg_scanner.pipeline import Pipeline

        primary = MagicMock()
        primary.recognize.return_value = (None, 0.0)

        fallback = MagicMock()
        fallback.recognize.return_value = (None, 0.0)

        scryfall = MagicMock()
        scryfall.lookup.return_value = None

        pipeline = Pipeline(
            primary_recognizer=primary,
            fallback_recognizer=fallback,
            scryfall_client=scryfall,
        )

        with patch.object(pipeline._detector, "detect", return_value=[card_patch]):
            result = pipeline.process_image("fake.jpg")

        assert result.cards[0].card_name is None
        assert result.cards[0].recognition_method == "unknown"
