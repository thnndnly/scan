from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mtg_scanner.models.card_patch import CardPatch


@dataclass
class CardData:
    """Data about a Magic: The Gathering card retrieved from Scryfall.

    Attributes:
        name: The card's full name.
        set_code: The set code (e.g. 'm21').
        collector_number: The collector number within the set.
        rarity: Rarity string ('common', 'uncommon', 'rare', 'mythic').
        type_line: The type line text (e.g. 'Instant').
        price_eur: Current EUR price from Scryfall, or None if unavailable.
        price_usd: Current USD price from Scryfall, or None if unavailable.
        scryfall_uri: URL to the card's Scryfall page.
        image_uri: URL to the card's normal-sized image.
    """

    name: str
    set_code: str
    collector_number: str
    rarity: str
    type_line: str
    price_eur: Optional[float]
    price_usd: Optional[float]
    scryfall_uri: str
    image_uri: str


@dataclass
class RecognizedCard:
    """A detected card patch paired with its recognition result and card data.

    Attributes:
        patch: The original CardPatch from detection.
        card_name: The recognised card name, or None if recognition failed.
        recognition_confidence: Confidence score for the recognition result (0.0 - 1.0).
        recognition_method: Which method produced the result: 'ocr', 'hash', 'llm', or 'unknown'.
        card_data: Full Scryfall card data, or None when lookup returned no result.
    """

    patch: CardPatch
    card_name: Optional[str]
    recognition_confidence: float
    recognition_method: str  # 'ocr' | 'hash' | 'llm' | 'unknown'
    card_data: Optional[CardData]  # None if Scryfall found nothing
