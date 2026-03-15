"""Shared pytest fixtures for mtg-card-scanner tests."""

from __future__ import annotations

import numpy as np
import pytest

from mtg_scanner.models.card_patch import CardPatch


@pytest.fixture
def sample_card_image() -> np.ndarray:
    """Return a synthetic 200×280 BGR card image.

    The image has a dark title bar (top ~40 rows), a coloured artwork region,
    and a white background elsewhere.
    """
    img = np.zeros((280, 200, 3), dtype=np.uint8)
    # Title region (dark grey)
    img[0:40, :] = [50, 50, 50]
    # Artwork region
    img[56:182, :] = [100, 150, 200]
    return img


@pytest.fixture
def sample_card_patch(sample_card_image: np.ndarray) -> CardPatch:
    """Return a :class:`CardPatch` wrapping :func:`sample_card_image`."""
    return CardPatch(
        image=sample_card_image,
        source_image_path="tests/fixtures/sample.jpg",
        bbox=(10, 20, 200, 280),
        detection_confidence=0.9,
        patch_index=0,
    )


@pytest.fixture
def mock_scryfall_response() -> dict:
    """Return a minimal Scryfall API response dict for Lightning Bolt."""
    return {
        "name": "Lightning Bolt",
        "set": "m21",
        "collector_number": "167",
        "rarity": "common",
        "type_line": "Instant",
        "prices": {"usd": "0.30", "eur": "0.25"},
        "scryfall_uri": "https://scryfall.com/card/m21/167",
        "image_uris": {"normal": "https://cards.scryfall.io/normal/m21/167.jpg"},
    }
